"""Recall: turn a question into a ranked, explained list of memories.

Two rules separate this from a vector search:

* **One live value per functional predicate.** A superseded address is not a
  candidate -- otherwise the model gets told two contradicting things and
  picks one at random.
* **Every hit carries a ``why``.** If the library cannot say why it surfaced a
  memory, it should not surface it.

Scoring is deliberately boring: predicate intent match, character-level
similarity, recency, importance. With 10k memories this is a full scan in pure
Python and still finishes well under the latency budget; a real index can come
later, when a number proves it is needed.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from .maintain import decay_from_fields
from .models import FUNCTIONAL_PREDICATES, Hit, human_ts, parse_ts
from .store.sqlite import SqliteStore

#: Query words -> predicate they imply. Cheap intent detection; an embedding
#: model would do better and would also be less debuggable.
INTENT: dict[str, tuple[str, ...]] = {
    "live_at": ("住", "住在", "搬家", "搬到", "家", "地址", "哪儿", "哪里", "在哪"),
    "moved_to": ("搬到", "搬家", "从哪来", "搬来", "迁"),
    "work_at": ("工作", "上班", "公司", "单位", "实习"),
    "study_at": ("上学", "学校", "读", "专业", "大学"),
    "age": ("几岁", "年龄", "多大", "岁数"),
    "origin": ("哪里人", "哪儿人", "老家", "家乡", "籍贯"),
    "likes": ("喜欢", "爱", "偏好", "口味"),
    "dislikes": ("不喜欢", "讨厌", "受不了"),
    "name": ("叫什么", "名字", "称呼"),
    "phone": ("电话", "手机号", "联系方式"),
    "partner": ("对象", "伴侣", "男朋友", "女朋友", "结婚"),
}

#: Words that make the caller ask about history instead of the present.
HISTORY_CUES = ("以前", "原来", "之前", "过去", "曾经", "换过", "上一个", "小时候", "儿时", "小时候住")

W_PRED = 0.45
W_SIM = 0.30
W_RECENCY = 0.15
W_IMPORTANCE = 0.10

#: When the caller explicitly asks about the past ("以前住哪"), closed intervals
#: get a boost so history outranks the present -- which is the whole point of
#: the question.
HISTORY_BOOST = 0.15

#: Below this similarity, a row that does not match the query's intent is noise.
SIM_FLOOR = 0.15


def bigrams(text: str) -> set[str]:
    t = "".join(ch for ch in text if ch.strip())
    if len(t) < 2:
        return {t} if t else set()
    return {t[i : i + 2] for i in range(len(t) - 1)}


def _similarity(qgrams: set[str], row: sqlite3.Row) -> float:
    if not qgrams:
        return 0.0
    target = bigrams(f"{row['content']} {row['object']} {row['predicate']}")
    if not target:
        return 0.0
    return len(qgrams & target) / len(qgrams)


def _implied_predicates(query: str) -> set[str]:
    return {pred for pred, words in INTENT.items() if any(w in query for w in words)}


def _recency(row: sqlite3.Row, now: datetime) -> float:
    ref = parse_ts(row["last_confirmed"] or row["t_valid_from"] or row["created_at"])
    if ref is None:
        return 0.5
    age_days = max(0.0, (now - ref).total_seconds() / 86400.0)
    return 0.5 ** (age_days / 365.0)


def _why(row: sqlite3.Row, parts: dict[str, float], decay: float) -> str:
    bits: list[str] = []
    if parts["pred"]:
        bits.append(f"谓词命中 {row['predicate']}")
    if parts["sim"] > 0:
        bits.append(f"字面 {parts['sim']:.2f}")
    bits.append(f"时效 {human_ts(row['t_valid_from'])}")
    if row["superseded_by"]:
        bits.append(f"已被 id={row['superseded_by']} 取代")
    elif row["t_valid_to"]:
        bits.append(f"止于 {human_ts(row['t_valid_to'])}")
    else:
        bits.append("仍有效")
    if decay < 1.0:
        bits.append(f"衰减权重 {decay:.2f}")
    if row["last_confirmed"]:
        bits.append(f"最近确认 {human_ts(row['last_confirmed'])}")
    return " · ".join(bits)


def recall(
    store: SqliteStore,
    query: str,
    *,
    limit: int = 8,
    include_history: bool | None = None,
    min_score: float = 0.05,
) -> list[Hit]:
    """Rank memories against a natural-language query."""
    now = datetime.now(timezone.utc)
    if include_history is None:
        include_history = any(cue in query for cue in HISTORY_CUES)

    implied = _implied_predicates(query)
    qgrams = bigrams(query)
    rows = store.raw_rows(live_only=not include_history)

    scored: list[tuple[float, sqlite3.Row, dict[str, float], float]] = []
    for row in rows:
        pred = 1.0 if row["predicate"] in implied else 0.0
        if pred == 0.0:
            # Cheap gate before the expensive similarity: does the row share
            # any character pair with the query at all?
            haystack = row["content"] + " " + row["object"]
            if not any(g in haystack for g in qgrams):
                continue

        sim = _similarity(qgrams, row)
        if pred == 0.0 and sim < SIM_FLOOR:
            continue

        rec = _recency(row, now)
        decay = decay_from_fields(
            row["kind"], row["predicate"], row["last_confirmed"] or row["created_at"] or row["t_valid_from"], now
        )
        raw = W_PRED * pred + W_SIM * sim + W_RECENCY * rec + W_IMPORTANCE * row["importance"]
        score = raw * row["confidence"] * decay
        if row["superseded_by"] or row["t_valid_to"]:
            # History stays visible when asked for, but never outranks the
            # present unless the question was about the past.
            score = score + HISTORY_BOOST if include_history else score * 0.35
        if score < min_score:
            continue
        scored.append((score, row, {"pred": pred, "sim": sim, "rec": rec}, decay))

    scored.sort(key=lambda t: (t[0], t[1]["id"]), reverse=True)

    # One value per functional predicate: the top-scoring live row wins, and
    # the rest are dropped rather than offered as competing truths.
    picked: list[Hit] = []
    seen_live: set[tuple[str, str]] = set()
    for score, row, parts, decay in scored:
        is_live = row["t_valid_to"] is None and row["superseded_by"] is None
        key = (row["subject_key"], row["predicate"])
        if is_live and row["predicate"] in FUNCTIONAL_PREDICATES:
            if key in seen_live:
                continue
            seen_live.add(key)

        picked.append(
            Hit(
                id=row["id"],
                content=row["content"],
                kind=row["kind"],
                predicate=row["predicate"],
                subject=row["subject"],
                score=score,
                why=_why(row, parts, decay),
                t_valid_from=row["t_valid_from"],
                t_valid_to=row["t_valid_to"],
                confidence=row["confidence"],
                stale=bool(row["superseded_by"] or row["t_valid_to"]),
            )
        )
        if len(picked) >= limit:
            break

    store.touch([h.id for h in picked])
    store.add_audit(
        "recall",
        query,
        [h.id for h in picked],
        "; ".join(f"{h.id}:{h.why}" for h in picked) or "no hit",
    )
    return picked

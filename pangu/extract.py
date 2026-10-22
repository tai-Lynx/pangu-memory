"""Rule-based fact extraction.

Design note: v1 deliberately does **not** call an LLM here.

Extraction quality is not what makes this project interesting -- conflict
resolution is. A rule-based extractor keeps the library dependency-free, keeps
tests deterministic, and keeps the interesting failure modes concentrated in
``maintain.py`` where they belong.

The extractor is a single function with a swappable signature, so plugging in
an LLM extractor later is a one-line change in ``Memory``.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from .models import Fact, now_ts

#: How far back a vague anchor like "小时候 / 以前" is placed. Deliberately
#: coarse: the point is "not now", not historical accuracy.
_VAGUE_PAST_YEARS = 15

_PUNCT_STRIP = re.compile(r"[\s,，。、；;!！?？~～…]+$")

#: Surface subject -> canonical subject. Everything else keeps its raw form.
_SUBJECT_MAP = {
    "我": "我",
    "俺": "我",
    "本人": "我",
    "自己": "我",
}

_WHEN_PATTERNS = [
    (re.compile(r"^(\d{4})年(?:(\d{1,2})月)?"), "explicit"),
    (re.compile(r"^(小时候|小时候|儿时|童年)"), "past"),
    (re.compile(r"^(以前|从前|曾经|之前|当时|那时)"), "past"),
    (re.compile(r"^(去年)"), "past"),
    (re.compile(r"^(前年)"), "past"),
]

# (kind, predicate, pattern). Order matters: most specific first.
_RULES: list[tuple[str, str, re.Pattern[str]]] = [
    # -- retraction ---------------------------------------------------------
    (
        "retraction",
        "live_at",
        re.compile(r"^(?:我)?(?:已经)?不(?:再)?(?:住|住在|居住)(?:在|于)?(?P<obj>[^，。！？,.!?]{1,20}?)了?$"),
    ),
    (
        "retraction",
        "work_at",
        re.compile(r"^(?:我)?(?:已经)?不(?:在)(?P<obj>[^，。！？,.!?]{1,20}?)(?:工作|上班)了?$"),
    ),
    # -- moving: an event that implies a new live_at ------------------------
    (
        "event",
        "moved_to",
        re.compile(
            r"^(?:我)?(?P<when>[^，。]{0,8}?)从(?P<frm>[^，。]{1,12}?)(?:搬|迁移?)(?:到|去)了?(?P<obj>[^，。！？,.!?]{1,12})$"
        ),
    ),
    # -- functional facts ---------------------------------------------------
    (
        "fact",
        "live_at",
        re.compile(
            r"^(?:我)?(?P<when>小时候|以前|曾经|从前|去年|前年|之前|\d{4}年[^，。]{0,6})?(?:现在)?(?:住|住在|居住)(?:在|于)?(?P<obj>[^，。！？,.!?]{1,20})$"
        ),
    ),
    (
        "fact",
        "work_at",
        re.compile(
            r"^(?:我)?(?P<when>以前|曾经|去年|前年)?(?:现在)?在(?P<obj>[^，。！？,.!?]{1,20}?)(?:工作|上班|实习|读书|上学)$"
        ),
    ),
    (
        "fact",
        "study_at",
        re.compile(r"^(?:我)?(?:现在)?(?:在)?(?P<obj>[^，。！？,.!?]{2,20}?)(?:上学|读大学|读大[一二三四]|念大学)$"),
    ),
    ("fact", "origin", re.compile(r"^(?:我)?(?:是)?(?P<obj>[^，。！？,.!?]{2,10}?)人$")),
    ("fact", "age", re.compile(r"^(?:我)?(?:今年)?(?P<obj>\d{1,3})\s*岁$")),
    ("fact", "job_title", re.compile(r"^(?:我)?(?:是|做)(?:一[个名])?(?P<obj>[^，。！？,.!?]{2,12}?)(?:工程师|设计师|老师|学生|程序员)?$")),
    # -- multi-valued preferences ------------------------------------------
    (
        "preference",
        "likes",
        re.compile(r"^(?:我)?(?:很|特别|超)?(?:喜欢|爱|偏爱|钟爱)(?P<obj>[^，。！？,.!?]{1,20})$"),
    ),
    (
        "preference",
        "dislikes",
        re.compile(r"^(?:我)?(?:很|特别|超)?(?:不喜欢|讨厌|受不了)(?P<obj>[^，。！？,.!?]{1,20})$"),
    ),
    # -- generic fallback: keep the sentence, infer nothing -----------------
    ("fact", "mentioned", re.compile(r"^(?P<obj>.{2,60})$")),
]


def _strip(text: str) -> str:
    return _PUNCT_STRIP.sub("", (text or "").strip())


def _parse_when(text: str | None, now: datetime) -> tuple[str | None, str | None, str]:
    """Return ``(t_valid_from, t_valid_to, anchor)`` for an anchor phrase.

    ``anchor == "now"`` means the sentence talks about the present, so it is
    allowed to supersede existing live facts. ``past`` / ``explicit`` facts are
    stored as *closed* intervals -- history never competes with the present.
    """
    if not text:
        return now_ts(), None, "now"

    for pattern, kind in _WHEN_PATTERNS:
        m = pattern.match(text)
        if not m:
            continue

        if kind == "explicit":
            year = int(m.group(1))
            month = int(m.group(2) or 1)
            start = datetime(year, month, 1, tzinfo=timezone.utc)
            if year < now.year:
                # A closed interval ending at the end of that period.
                end = datetime(year + 1, month, 1, tzinfo=timezone.utc) - timedelta(seconds=1)
                return start.strftime("%Y-%m-%dT%H:%M:%S"), end.strftime("%Y-%m-%dT%H:%M:%S"), "explicit"
            return start.strftime("%Y-%m-%dT%H:%M:%S"), None, "explicit"

        # past / vague anchors
        if text.startswith("去年"):
            start = now - timedelta(days=365)
        elif text.startswith("前年"):
            start = now - timedelta(days=730)
        else:
            start = now - timedelta(days=365 * _VAGUE_PAST_YEARS)
        # Closed at "recently": anything vague is treated as finished business.
        end = now - timedelta(days=1)
        if end <= start:
            end = start + timedelta(days=1)
        return start.strftime("%Y-%m-%dT%H:%M:%S"), end.strftime("%Y-%m-%dT%H:%M:%S"), "past"

    return now_ts(), None, "now"


def _importance_for(kind: str, predicate: str) -> float:
    base = {"fact": 0.55, "event": 0.6, "preference": 0.5, "relation": 0.5, "retraction": 0.85}[kind]
    if predicate in {"live_at", "work_at", "name", "phone", "partner"}:
        base += 0.2
    return round(min(1.0, base), 2)


def extract(
    text: str,
    *,
    source: str = "",
    subject: str = "我",
    now: datetime | None = None,
) -> list[Fact]:
    """Turn one sentence into zero or more :class:`Fact` objects.

    Empty list means "nothing worth remembering", not an error.
    """
    now = now or datetime.now(timezone.utc)
    sentence = _strip(text)
    if not sentence:
        return []

    for kind, predicate, pattern in _RULES:
        m = pattern.match(sentence)
        if not m:
            continue

        groups = m.groupdict()
        obj = (groups.get("obj") or "").strip()
        if not obj:
            continue

        when = groups.get("when")
        t_from, t_to, anchor = _parse_when(when, now)

        facts = [
            Fact(
                content=sentence,
                kind=kind,
                predicate=predicate,
                object=obj,
                subject=subject,
                t_valid_from=t_from,
                t_valid_to=t_to,
                anchor=anchor,
                confidence=0.85 if kind != "fact" or predicate != "mentioned" else 0.55,
                importance=_importance_for(kind, predicate),
                source=source,
            )
        ]

        # "我九月从深圳搬到北京" is an event, but it also settles where the
        # person lives *now*. Deriving that is the cheapest way to show that
        # the memory layer understands implications, not just sentences.
        if predicate == "moved_to":
            facts.append(
                Fact(
                    content=f"住在{obj}",
                    kind="fact",
                    predicate="live_at",
                    object=obj,
                    subject=subject,
                    t_valid_from=t_from,
                    t_valid_to=None,
                    anchor="now",
                    confidence=0.8,
                    importance=0.8,
                    source=source,
                    meta={"derived_from": "moved_to", "previous": groups.get("frm", "")},
                )
            )
        return facts

    return []

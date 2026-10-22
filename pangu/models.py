"""Core data types for pangu-memory.

Everything that crosses a module boundary is one of these three dataclasses.
Keeping them dumb (no behaviour) makes the storage layer swappable.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

# ---------------------------------------------------------------------------
# time helpers
# ---------------------------------------------------------------------------

#: Everything is stored as a naive-UTC ISO string, second precision. Naive-UTC
#: keeps string ordering == chronological ordering, which lets SQLite sort
#: timestamps without parsing them.
_TS_FMT = "%Y-%m-%dT%H:%M:%S"


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime(_TS_FMT)


def to_ts(value: str | datetime | None) -> str | None:
    """Accept ISO-ish input and normalise it. Returns None for falsy input."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime(_TS_FMT)
    s = str(value).strip()
    for fmt in (_TS_FMT, "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc).strftime(_TS_FMT)
        except ValueError:
            continue
    raise ValueError(f"unrecognised timestamp: {value!r}")


def parse_ts(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    return datetime.strptime(value, _TS_FMT).replace(tzinfo=timezone.utc)


def human_ts(value: str | None) -> str:
    """'2026-09-15T00:00:00' -> '2026-09-15'. For logs and `why` strings."""
    if not value:
        return "—"
    return value[:10]


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[\s,，。、；;：:!！?？~～…\"'“”‘’()（）\[\]【】]+")

#: First-person surface forms that all collapse to the single key ``self``.
_SELF_FORMS = frozenset({"我", "俺", "本人", "自己", "我自己", "i", "me", "my", "self"})

#: Predicates where one subject can only hold one value at a time.
#: This set is the single most important judgement call in the conflict
#: resolver -- it decides whether two facts *contradict* or merely *coexist*.
FUNCTIONAL_PREDICATES = frozenset(
    {
        "live_at",
        "work_at",
        "study_at",
        "age",
        "phone",
        "partner",
        "origin",
        "major",
        "name",
        "job_title",
    }
)

#: Predicates that accept many values. "I like coffee" never invalidates
#: "I like tea".
MULTI_PREDICATES = frozenset({"likes", "dislikes", "owns", "knows", "skill"})


def norm_text(text: str) -> str:
    """Collapse punctuation/whitespace and lowercase. Used for equality checks."""
    return _PUNCT.sub("", text or "").strip().lower()


def subject_key(subject: str) -> str:
    """Canonical subject key.

    First person collapses to ``self`` so that "我 / 俺 / 本人" don't create
    three parallel universes of memories. Third parties keep their raw name.
    """
    s = norm_text(subject)
    if s in _SELF_FORMS:
        return "self"
    return s


def object_key(obj: str) -> str:
    return norm_text(obj)


def fingerprint(kind: str, subject: str, predicate: str, obj: str) -> str:
    raw = f"{kind}|{subject_key(subject)}|{predicate}|{object_key(obj)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


#: Chinese surface forms, so that facts written through the structured API read
#: like the ones extracted from sentences. ``{o}`` marks where the object goes.
PREDICATE_LABELS = {
    "live_at": "住在",
    "work_at": "在{o}工作",
    "study_at": "在{o}上学",
    "age": "{o}岁",
    "origin": "{o}人",
    "likes": "喜欢",
    "dislikes": "不喜欢",
    "name": "叫",
    "phone": "的电话是",
    "partner": "的伴侣是",
    "job_title": "是",
    "moved_to": "搬到了",
}


def render_fact(subject: str, predicate: str, obj: str) -> str:
    """Human-readable line for a structured fact."""
    label = PREDICATE_LABELS.get(predicate)
    if not label:
        return f"{subject} {predicate} {obj}".strip()
    if "{o}" in label:
        return f"{subject}{label.replace('{o}', obj)}"
    return f"{subject}{label}{obj}"


# ---------------------------------------------------------------------------
# dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Fact:
    """A single thing worth remembering.

    ``anchor`` is what keeps the resolver honest:

    * ``now``    -- stated as current truth ("我住昌平")
    * ``past``   -- explicitly anchored to a past period ("我小时候住福田")
    * ``explicit`` -- carries its own date ("2019年我住深圳")

    Only ``now`` facts are allowed to supersede live facts. A ``past`` fact
    with a different value is not a contradiction, it is history.
    """

    content: str
    predicate: str
    object: str = ""
    subject: str = "self"
    kind: str = "fact"  # fact | event | preference | relation | retraction
    t_valid_from: str | None = None
    t_valid_to: str | None = None
    confidence: float = 0.8
    importance: float = 0.5
    source: str = ""
    anchor: str = "now"  # now | past | explicit
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t_valid_from = to_ts(self.t_valid_from) or now_ts()
        self.t_valid_to = to_ts(self.t_valid_to)
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.importance = max(0.0, min(1.0, float(self.importance)))

    @property
    def subject_key(self) -> str:
        return subject_key(self.subject)

    @property
    def object_key(self) -> str:
        return object_key(self.object)

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.kind, self.subject, self.predicate, self.object)

    @property
    def is_functional(self) -> bool:
        return self.predicate in FUNCTIONAL_PREDICATES

    def meta_json(self) -> str:
        return json.dumps(self.meta, ensure_ascii=False)


@dataclass
class MemoryRow(Fact):
    """A Fact that has been persisted: adds identity and lifecycle fields."""

    id: int = 0
    superseded_by: int | None = None
    quarantined: int = 0
    last_access: str | None = None
    last_confirmed: str | None = None
    created_at: str | None = None
    score: float = 0.0

    @classmethod
    def from_row(cls, row: Any) -> "MemoryRow":
        d = dict(row)
        meta = d.pop("meta", None)
        # Columns that mirror @property names cannot be passed to __init__.
        for shadowed in ("subject_key", "object_key", "fingerprint", "is_functional"):
            d.pop(shadowed, None)
        return cls(meta=json.loads(meta) if meta else {}, **d)

    @property
    def is_live(self) -> bool:
        return self.t_valid_to is None and self.superseded_by is None and not self.quarantined


@dataclass
class Resolution:
    """What the resolver decided, before anything is written.

    ``action`` is one of:

    * ``insert``      -- no relation to existing memories, just store it
    * ``duplicate``   -- same thing again; reinforce instead of inserting
    * ``supersede``   -- newer truth replaces older one; old row is closed
    * ``coexist``     -- different valid period or multi-valued predicate
    * ``retract``     -- "I don't live there anymore"; closes matching rows
    * ``conflict``    -- genuine ambiguity; quarantine for human review
    """

    action: str
    reason: str
    supersedes: list[int] = field(default_factory=list)
    conflict_with: list[int] = field(default_factory=list)
    duplicate_of: int | None = None

    def __str__(self) -> str:
        extra = ""
        if self.supersedes:
            extra = f" -> supersedes {self.supersedes}"
        elif self.conflict_with:
            extra = f" -> queued against {self.conflict_with}"
        elif self.duplicate_of:
            extra = f" -> duplicate of {self.duplicate_of}"
        return f"{self.action}: {self.reason}{extra}"


@dataclass
class Hit:
    """A recall result plus the explanation of why it was picked."""

    id: int
    content: str
    kind: str
    predicate: str
    subject: str
    score: float
    why: str
    t_valid_from: str | None = None
    t_valid_to: str | None = None
    confidence: float = 0.8
    stale: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "kind": self.kind,
            "predicate": self.predicate,
            "subject": self.subject,
            "score": round(self.score, 4),
            "why": self.why,
            "t_valid_from": self.t_valid_from,
            "t_valid_to": self.t_valid_to,
            "confidence": self.confidence,
            "stale": self.stale,
        }

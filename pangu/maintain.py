"""Conflict detection and resolution -- the moat.

This module answers the only question a real memory layer has to answer and a
vector store never does:

    *She moved. Does the old address still count?*

The answer is a policy, and the policy is written down here.

Three ideas carry the whole thing:

1. **Not every difference is a contradiction.** "I like coffee" and "I like
   tea" coexist. Only *functional* predicates (``live_at``, ``age``, ...) can
   hold one value per subject at a time.
2. **Time turns contradiction into evolution.** "I lived in Futian as a kid"
   is not in conflict with "I live in Changping" -- the first one is stored as
   a closed interval and never competes for *current* value.
3. **Nothing is deleted.** A superseded row keeps its interval and a
   ``superseded_by`` pointer. Forgetting is explicit, not accidental.

The decision table lives in :func:`resolve`.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .models import MemoryRow, Fact, Resolution, parse_ts
from .store.sqlite import SqliteStore

#: Kinds whose weight fades with time. Durable facts (where you live, your
#: name) do not decay -- being unmentioned for a year does not make them false.
_DECAYING_KINDS = frozenset({"event", "preference"})

#: Individual predicates that fade regardless of kind. A passing mention is
#: weak evidence by nature; it should sink on its own.
_DECAYING_PREDICATES = frozenset({"mentioned"})

#: How much more confident a new fact must be before it may overwrite a live
#: one. Below this margin the pair is quarantined instead of guessed at.
CONFIDENCE_MARGIN = 0.15

DEFAULT_HALF_LIFE_DAYS = 90.0


def resolve(store: SqliteStore, fact: Fact) -> Resolution:
    """Pure decision function: what should happen to ``fact``?

    Reads only. Writing happens in :func:`apply`.
    """
    # 0. An explicitly past-anchored fact is history by construction.
    #    extract() already closed its interval, so it can never fight the present.
    if fact.t_valid_to is not None and fact.anchor != "now":
        return Resolution("insert", f"{fact.anchor}-anchored fact, stored as closed interval")

    live = store.live_for(fact.subject_key, fact.predicate)

    # 1. Retraction: "I don't live there anymore."
    if fact.kind == "retraction":
        targets = [r for r in live if not fact.object_key or r.object_key == fact.object_key]
        if not targets:
            return Resolution("insert", "retraction with nothing to retract")
        return Resolution(
            "retract",
            f"explicit retraction closes {len(targets)} live fact(s)",
            supersedes=[r.id for r in targets],
        )

    if not live:
        return Resolution("insert", "no existing fact on this (subject, predicate)")

    # 2. Same value: reinforce instead of duplicating.
    for row in live:
        if row.object_key == fact.object_key and row.kind == fact.kind:
            return Resolution("duplicate", "same value already remembered", duplicate_of=row.id)

    # 3. Multi-valued predicates accumulate.
    if not fact.is_functional:
        return Resolution("insert", f"'{fact.predicate}' is multi-valued, values coexist")

    # 4. Functional predicate, different value -> someone has to win.
    newest = max(live, key=lambda r: (r.t_valid_from or "", r.id))

    if fact.confidence + CONFIDENCE_MARGIN < newest.confidence:
        return Resolution(
            "conflict",
            f"new fact less confident ({fact.confidence:.2f}) than existing "
            f"({newest.confidence:.2f}); quarantined rather than guessed",
            conflict_with=[r.id for r in live],
        )

    if fact.t_valid_from and newest.t_valid_from and fact.t_valid_from < newest.t_valid_from:
        return Resolution(
            "conflict",
            "new fact claims to be current but starts earlier than the live one; "
            "quarantined as a probable extraction error",
            conflict_with=[r.id for r in live],
        )

    return Resolution(
        "supersede",
        f"newer value for functional predicate '{fact.predicate}'; "
        f"closing {len(live)} previous interval(s)",
        supersedes=[r.id for r in live],
    )


def apply(store: SqliteStore, fact: Fact) -> tuple[int, Resolution]:
    """Run the decision and write the result. Returns ``(row_id, resolution)``.

    ``row_id`` is ``0`` for retractions, which are recorded in the audit log
    but never stored as memories -- a sentence saying *what is not true* has no
    business being recalled as a fact.
    """
    res = resolve(store, fact)

    if res.action == "duplicate":
        assert res.duplicate_of is not None
        store.reinforce(res.duplicate_of)
        store.add_audit("write", fact.content, [res.duplicate_of], res.reason)
        return res.duplicate_of, res

    if res.action == "retract":
        for oid in res.supersedes:
            store.close_interval(oid, fact.t_valid_from or "", superseded_by=None)
        store.add_audit("retract", fact.content, res.supersedes, res.reason)
        return 0, res

    new_id = store.insert(fact)

    if res.action == "supersede":
        for oid in res.supersedes:
            store.close_interval(oid, fact.t_valid_from or "", superseded_by=new_id)
    elif res.action == "conflict":
        # Park it. Quarantined facts are stored and auditable but invisible to
        # recall, because a memory layer that guesses is worse than one that
        # admits it does not know.
        store.quarantine(new_id)
        store.add_conflict(new_id, res.conflict_with, res.reason)

    store.add_audit("write", fact.content, [new_id], res.reason)
    return new_id, res


# ---------------------------------------------------------------------------
# maintenance operations
# ---------------------------------------------------------------------------


def decay_from_fields(
    kind: str,
    predicate: str,
    reference_ts: str | None,
    now: datetime | None = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> float:
    """Field-level decay so the recall hot path never builds objects."""
    if kind not in _DECAYING_KINDS and predicate not in _DECAYING_PREDICATES:
        return 1.0
    ref = parse_ts(reference_ts)
    if ref is None:
        return 1.0
    now = now or datetime.now(timezone.utc)
    age_days = max(0.0, (now - ref).total_seconds() / 86400.0)
    return 0.5 ** (age_days / half_life_days)


def decay_factor(row: MemoryRow, now: datetime | None = None, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> float:
    """Weight multiplier in ``(0, 1]`` based on how long ago it was confirmed.

    Only events and preferences fade. A fact about where someone lives is
    either current or superseded -- never "half true because it is old".
    """
    return decay_from_fields(
        row.kind,
        row.predicate,
        row.last_confirmed or row.created_at or row.t_valid_from,
        now,
        half_life_days,
    )


def forget(
    store: SqliteStore,
    *,
    subject_key: str | None = None,
    predicate: str | None = None,
    before: str | None = None,
    ids: list[int] | None = None,
) -> int:
    """Explicitly close memories. Returns the number of rows affected.

    Deleting is not offered on purpose: a closed interval keeps the history
    auditable. If a caller truly needs the bytes gone, they delete the sqlite
    file -- which is a feature, not a gap.
    """
    from .models import now_ts

    targets: list[MemoryRow] = []
    if ids:
        targets = [r for r in (store.get(i) for i in ids) if r and r.is_live]
    else:
        rows = store.all_rows(live_only=True)
        for r in rows:
            if subject_key and r.subject_key != subject_key:
                continue
            if predicate and r.predicate != predicate:
                continue
            if before and (r.t_valid_from or "") >= before:
                continue
            targets.append(r)

    end = before or now_ts()
    for r in targets:
        store.close_interval(r.id, end)
    store.add_audit("forget", f"subject={subject_key} predicate={predicate} before={before}", [r.id for r in targets], "explicit forget")
    return len(targets)


def consolidate(store: SqliteStore) -> int:
    """Collapse exact-duplicate live rows. Returns rows closed."""
    seen: dict[str, MemoryRow] = {}
    closed = 0
    for row in store.all_rows(live_only=True):
        key = f"{row.subject_key}|{row.predicate}|{row.object_key}|{row.kind}"
        if key in seen:
            keeper = seen[key]
            if row.id != keeper.id:
                store.close_interval(row.id, row.t_valid_from or "", superseded_by=keeper.id)
                closed += 1
        else:
            seen[key] = row
    return closed

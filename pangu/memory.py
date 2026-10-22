"""The public façade. Everything a user touches lives here.

    from pangu import Memory
    m = Memory()
    m.write("我九月从深圳搬到北京")
    m.context_for("她现在住哪")

Five lines, no server, no API key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import context as _context
from . import maintain
from .extract import extract as default_extractor
from .models import Fact, Hit, MemoryRow, Resolution, render_fact
from .recall import recall as _recall
from .store.sqlite import SqliteStore

DEFAULT_PATH = "~/.pangu/db.sqlite"

Extractor = Callable[..., list[Fact]]


@dataclass
class WriteResult:
    """What one ``write()`` call did, sentence by sentence."""

    ids: list[int] = field(default_factory=list)
    resolutions: list[Resolution] = field(default_factory=list)
    stored: int = 0
    skipped: int = 0

    @property
    def conflicts(self) -> list[Resolution]:
        return [r for r in self.resolutions if r.action == "conflict"]

    def report(self) -> str:
        if not self.resolutions:
            return "nothing stored"
        return "\n".join(f"  [{i}] {r}" for i, r in enumerate(self.resolutions))

    def __str__(self) -> str:
        return f"stored={self.stored} skipped={self.skipped} conflicts={len(self.conflicts)}"


class Memory:
    """A local, explainable, forgettable long-term memory."""

    def __init__(self, path: str | Path = DEFAULT_PATH, extractor: Extractor | None = None) -> None:
        self.store = SqliteStore(path)
        self._extract = extractor or default_extractor

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- write -------------------------------------------------------------

    def write(self, text: str, *, source: str = "", subject: str = "我", silent: bool = False) -> WriteResult:
        """Extract facts from one sentence and store them under conflict policy."""
        result = WriteResult()
        facts = self._extract(text, source=source, subject=subject)
        if not facts:
            result.skipped = 1
            if not silent:
                print(f"  (nothing worth remembering in: {text!r})")
            return result

        for fact in facts:
            row_id, resolution = maintain.apply(self.store, fact)
            result.resolutions.append(resolution)
            if row_id:
                result.ids.append(row_id)
                result.stored += 1
            else:
                result.skipped += 1
        return result

    def remember(
        self,
        predicate: str,
        object: str,
        *,
        subject: str = "我",
        kind: str = "fact",
        t_valid_from: str | None = None,
        confidence: float = 0.9,
        importance: float = 0.7,
        source: str = "api",
    ) -> tuple[int, Resolution]:
        """Store a structured fact without going through extraction.

        Useful when the caller already knows what it wants to remember.
        """
        fact = Fact(
            content=render_fact(subject, predicate, object),
            predicate=predicate,
            object=object,
            subject=subject,
            kind=kind,
            t_valid_from=t_valid_from,
            confidence=confidence,
            importance=importance,
            source=source,
        )
        return maintain.apply(self.store, fact)

    # -- read --------------------------------------------------------------

    def recall(self, query: str, *, limit: int = 8, include_history: bool | None = None) -> list[Hit]:
        return _recall(self.store, query, limit=limit, include_history=include_history)

    def context_for(self, query: str, *, max_tokens: int = 600, include_why: bool = False, limit: int = 8) -> str:
        hits = self.recall(query, limit=limit)
        return _context.render(hits, max_tokens=max_tokens, include_why=include_why)

    def history(self, *, subject: str = "我", predicate: str = "live_at") -> list[MemoryRow]:
        from .models import subject_key

        return self.store.history(subject_key(subject), predicate)

    # -- maintenance -------------------------------------------------------

    def conflicts(self, resolved: bool = False) -> list[dict]:
        return self.store.conflicts(resolved=resolved)

    def resolve_conflict(self, conflict_id: int, keep: str = "new") -> None:
        """Settle a quarantined conflict.

        ``keep="new"`` promotes the quarantined fact and closes the old one.
        ``keep="old"`` throws the new claim away. Both leave an audit trail;
        neither deletes anything.
        """
        import json

        from .models import now_ts

        row = self.store.get_conflict(conflict_id)
        if row is None:
            raise KeyError(f"no conflict with id {conflict_id}")

        new_id = row["new_id"]
        old_ids = json.loads(row["old_ids"] or "[]")

        if keep == "new" and new_id:
            self.store.activate(new_id)
            promoted = self.store.get(new_id)
            for oid in old_ids:
                self.store.close_interval(oid, (promoted.t_valid_from if promoted else now_ts()) or now_ts(), superseded_by=new_id)
        elif new_id:
            self.store.close_interval(new_id, now_ts())

        self.store.resolve_conflict(conflict_id)

    def forget(self, **kwargs) -> int:
        return maintain.forget(self.store, **kwargs)

    def consolidate(self) -> int:
        return maintain.consolidate(self.store)

    # -- inspection --------------------------------------------------------

    def audit_tail(self, n: int = 20) -> list[dict]:
        return self.store.audit_tail(n)

    def stats(self) -> dict:
        return self.store.stats()

    # -- escape hatch ------------------------------------------------------

    def use_extractor(self, fn: Extractor) -> None:
        """Swap in an LLM-backed extractor, e.g. ``memory.use_extractor(my_llm_extract)``.

        The callable must accept ``(text, source=..., subject=...)`` and return
        ``list[Fact]``. Contract is intentionally narrow so the interesting
        code (conflict policy) never has to change.
        """
        self._extract = fn

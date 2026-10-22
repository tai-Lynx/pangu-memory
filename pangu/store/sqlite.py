"""SQLite storage backend.

One file. No server. ``cp ~/.pangu/db.sqlite /Volumes/usb/`` *is* a full backup
of everything the system remembers -- that sentence is the whole privacy story,
and it is why this project does not use Postgres.

The store knows nothing about conflict policy; it only knows how to persist and
fetch rows. All judgement lives in ``pangu.maintain``.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..models import Fact, MemoryRow, now_ts

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS memory (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    kind           TEXT    NOT NULL,
    subject        TEXT    NOT NULL,
    subject_key    TEXT    NOT NULL,
    predicate      TEXT    NOT NULL,
    object         TEXT    NOT NULL DEFAULT '',
    object_key     TEXT    NOT NULL DEFAULT '',
    content        TEXT    NOT NULL,
    t_valid_from   TEXT    NOT NULL,
    t_valid_to     TEXT,
    anchor         TEXT    NOT NULL DEFAULT 'now',
    confidence     REAL    NOT NULL DEFAULT 0.8,
    importance     REAL    NOT NULL DEFAULT 0.5,
    source         TEXT    NOT NULL DEFAULT '',
    fingerprint    TEXT    NOT NULL,
    superseded_by  INTEGER REFERENCES memory(id),
    quarantined    INTEGER NOT NULL DEFAULT 0,
    last_access    TEXT,
    last_confirmed TEXT,
    created_at     TEXT    NOT NULL,
    meta           TEXT
);

CREATE INDEX IF NOT EXISTS idx_mem_sp      ON memory(subject_key, predicate);
CREATE INDEX IF NOT EXISTS idx_mem_live    ON memory(predicate, t_valid_to);
CREATE INDEX IF NOT EXISTS idx_mem_fp      ON memory(fingerprint);
CREATE INDEX IF NOT EXISTS idx_mem_created ON memory(created_at);

CREATE TABLE IF NOT EXISTS conflict (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    new_id     INTEGER,
    old_ids    TEXT NOT NULL,
    reason     TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS audit (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    op         TEXT NOT NULL,
    query      TEXT,
    picked_ids TEXT,
    why        TEXT,
    ts         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta_info (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

MIGRATIONS: dict[str, list[str]] = {
    "1": [],  # initial schema, nothing to do
}

SCHEMA_VERSION = "1"


class SqliteStore:
    def __init__(self, path: str | Path = "~/.pangu/db.sqlite") -> None:
        raw = str(path)
        if raw == ":memory:":
            self.path = raw
        else:
            p = Path(raw).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            self.path = str(p)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._ensure_columns()
        self._conn.execute(
            "INSERT OR IGNORE INTO meta_info(key, value) VALUES ('schema_version', ?)",
            (SCHEMA_VERSION,),
        )
        self._conn.commit()

    def _ensure_columns(self) -> None:
        """Tiny forward-migration: add columns introduced after the first release.

        A memory database is something people keep. Breaking their file on
        upgrade would be the one unforgivable bug.
        """
        have = {row[1] for row in self._conn.execute("PRAGMA table_info(memory)")}
        additions = {
            "quarantined": "INTEGER NOT NULL DEFAULT 0",
            "anchor": "TEXT NOT NULL DEFAULT 'now'",
            "last_confirmed": "TEXT",
            "fingerprint": "TEXT NOT NULL DEFAULT ''",
        }
        for col, ddl in additions.items():
            if col not in have:
                self._conn.execute(f"ALTER TABLE memory ADD COLUMN {col} {ddl}")

    # -- plumbing ----------------------------------------------------------

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SqliteStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- writes ------------------------------------------------------------

    def insert(self, fact: Fact) -> int:
        with self.tx() as c:
            cur = c.execute(
                """
                INSERT INTO memory (kind, subject, subject_key, predicate, object, object_key,
                                    content, t_valid_from, t_valid_to, anchor, confidence,
                                    importance, source, fingerprint, last_confirmed,
                                    created_at, meta)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    fact.kind,
                    fact.subject,
                    fact.subject_key,
                    fact.predicate,
                    fact.object,
                    fact.object_key,
                    fact.content,
                    fact.t_valid_from,
                    fact.t_valid_to,
                    fact.anchor,
                    fact.confidence,
                    fact.importance,
                    fact.source,
                    fact.fingerprint,
                    now_ts(),  # last_confirmed
                    now_ts(),  # created_at
                    fact.meta_json(),
                ),
            )
            return int(cur.lastrowid or 0)

    def close_interval(self, row_id: int, end_ts: str, superseded_by: int | None = None) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE memory SET t_valid_to = ?, superseded_by = COALESCE(?, superseded_by) WHERE id = ?",
                (end_ts, superseded_by, row_id),
            )

    def reinforce(self, row_id: int, importance_bump: float = 0.02) -> None:
        with self.tx() as c:
            c.execute(
                """UPDATE memory
                      SET last_confirmed = ?,
                          importance = MIN(1.0, importance + ?),
                          confidence = MIN(1.0, confidence + 0.01)
                    WHERE id = ?""",
                (now_ts(), importance_bump, row_id),
            )

    def touch(self, row_ids: Iterable[int]) -> None:
        ids = list(row_ids)
        if not ids:
            return
        with self.tx() as c:
            c.executemany(
                "UPDATE memory SET last_access = ? WHERE id = ?",
                [(now_ts(), i) for i in ids],
            )

    def add_conflict(self, new_id: int | None, old_ids: list[int], reason: str) -> int:
        with self.tx() as c:
            cur = c.execute(
                "INSERT INTO conflict(new_id, old_ids, reason, created_at) VALUES (?,?,?,?)",
                (new_id, json.dumps(old_ids), reason, now_ts()),
            )
            return int(cur.lastrowid or 0)

    def add_audit(self, op: str, query: str | None, picked: list[int], why: str) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO audit(op, query, picked_ids, why, ts) VALUES (?,?,?,?,?)",
                (op, query, json.dumps(picked), why, now_ts()),
            )

    # -- reads -------------------------------------------------------------

    def get(self, row_id: int) -> MemoryRow | None:
        row = self._conn.execute("SELECT * FROM memory WHERE id = ?", (row_id,)).fetchone()
        return MemoryRow.from_row(row) if row else None

    def live_for(self, subject_key: str, predicate: str) -> list[MemoryRow]:
        rows = self._conn.execute(
            """SELECT * FROM memory
                WHERE subject_key = ? AND predicate = ?
                  AND t_valid_to IS NULL AND superseded_by IS NULL AND quarantined = 0
                ORDER BY t_valid_from DESC, id DESC""",
            (subject_key, predicate),
        ).fetchall()
        return [MemoryRow.from_row(r) for r in rows]

    def by_fingerprint(self, fp: str, live_only: bool = True) -> MemoryRow | None:
        sql = "SELECT * FROM memory WHERE fingerprint = ?"
        if live_only:
            sql += " AND t_valid_to IS NULL AND superseded_by IS NULL AND quarantined = 0"
        sql += " ORDER BY id DESC LIMIT 1"
        row = self._conn.execute(sql, (fp,)).fetchone()
        return MemoryRow.from_row(row) if row else None

    def all_rows(
        self, live_only: bool = False, limit: int | None = None, include_quarantined: bool = False
    ) -> list[MemoryRow]:
        sql = "SELECT * FROM memory"
        if live_only:
            sql += " WHERE t_valid_to IS NULL AND superseded_by IS NULL AND quarantined = 0"
        elif not include_quarantined:
            sql += " WHERE quarantined = 0"
        sql += " ORDER BY id"
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = self._conn.execute(sql).fetchall()
        return [MemoryRow.from_row(r) for r in rows]

    def raw_rows(
        self, live_only: bool = False, include_quarantined: bool = False
    ) -> list[sqlite3.Row]:
        """Rows without dataclass materialisation -- the hot path for recall.

        Building 10k ``MemoryRow`` objects just to throw most of them away was
        the single largest cost in the first benchmark. Rows are read-only
        here; anything that survives scoring gets materialised later.
        """
        sql = "SELECT * FROM memory"
        if live_only:
            sql += " WHERE t_valid_to IS NULL AND superseded_by IS NULL AND quarantined = 0"
        elif not include_quarantined:
            sql += " WHERE quarantined = 0"
        return self._conn.execute(sql).fetchall()

    def quarantine(self, row_id: int) -> None:
        """Park a suspicious fact: stored, auditable, but never recalled."""
        with self.tx() as c:
            c.execute("UPDATE memory SET quarantined = 1 WHERE id = ?", (row_id,))

    def activate(self, row_id: int) -> None:
        with self.tx() as c:
            c.execute("UPDATE memory SET quarantined = 0 WHERE id = ?", (row_id,))

    def history(self, subject_key: str, predicate: str) -> list[MemoryRow]:
        rows = self._conn.execute(
            """SELECT * FROM memory WHERE subject_key = ? AND predicate = ?
                ORDER BY t_valid_from ASC, id ASC""",
            (subject_key, predicate),
        ).fetchall()
        return [MemoryRow.from_row(r) for r in rows]

    def conflicts(self, resolved: bool = False) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM conflict WHERE resolved = ? ORDER BY id DESC", (1 if resolved else 0,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_conflict(self, conflict_id: int) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM conflict WHERE id = ?", (conflict_id,)).fetchone()
        return dict(row) if row else None

    def resolve_conflict(self, conflict_id: int) -> None:
        with self.tx() as c:
            c.execute("UPDATE conflict SET resolved = 1 WHERE id = ?", (conflict_id,))

    def audit_tail(self, n: int = 20) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0])

    def stats(self) -> dict[str, Any]:
        c = self._conn
        return {
            "rows": self.count(),
            "live": int(
                c.execute(
                    "SELECT COUNT(*) FROM memory WHERE t_valid_to IS NULL AND superseded_by IS NULL AND quarantined = 0"
                ).fetchone()[0]
            ),
            "superseded": int(c.execute("SELECT COUNT(*) FROM memory WHERE superseded_by IS NOT NULL").fetchone()[0]),
            "open_conflicts": int(c.execute("SELECT COUNT(*) FROM conflict WHERE resolved = 0").fetchone()[0]),
            "path": self.path,
            "size_kb": round(os.path.getsize(self.path) / 1024, 1) if self.path != ":memory:" else 0.0,
        }

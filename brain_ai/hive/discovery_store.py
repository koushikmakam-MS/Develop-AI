"""
Discovery Store — SQLite-backed metadata store for hive discovery data.

Tracks per-hive:
  - Auto-extracted topics (from docs & code)
  - Last indexed timestamps (docs + code separately)
  - Chunk counts
  - Freshness / staleness

This replaces manual topic maintenance in config.yaml with
auto-generated, always-fresh topic data.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = ".chromadb/discovery.db"
STALE_THRESHOLD_DAYS = 7  # Warn if a hive hasn't been re-indexed in this many days


class DiscoveryStore:
    """SQLite store for hive discovery metadata."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS hive_metadata (
                hive_name       TEXT PRIMARY KEY,
                display_name    TEXT,
                last_code_index TEXT,   -- ISO timestamp
                last_doc_index  TEXT,   -- ISO timestamp
                last_topic_refresh TEXT, -- ISO timestamp
                code_chunks     INTEGER DEFAULT 0,
                doc_chunks      INTEGER DEFAULT 0,
                index_duration_s REAL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS hive_topics (
                hive_name   TEXT NOT NULL,
                topic       TEXT NOT NULL,
                source      TEXT DEFAULT 'auto',  -- 'auto' or 'manual'
                confidence  REAL DEFAULT 1.0,
                created_at  TEXT NOT NULL,
                PRIMARY KEY (hive_name, topic)
            );

            CREATE TABLE IF NOT EXISTS index_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                hive_name   TEXT NOT NULL,
                index_type  TEXT NOT NULL,  -- 'code' or 'docs'
                started_at  TEXT NOT NULL,
                finished_at TEXT,
                chunks      INTEGER DEFAULT 0,
                files       INTEGER DEFAULT 0,
                duration_s  REAL DEFAULT 0,
                status      TEXT DEFAULT 'running'  -- 'running', 'success', 'failed'
            );
        """)
        self._conn.commit()

    # ── Metadata ────────────────────────────────────────────────

    def update_index_metadata(
        self,
        hive_name: str,
        index_type: str,  # "code" or "docs"
        chunks: int,
        display_name: str = "",
        duration_s: float = 0,
    ):
        """Record that a hive was just indexed."""
        now = datetime.now(timezone.utc).isoformat()

        # Upsert metadata
        self._conn.execute("""
            INSERT INTO hive_metadata (hive_name, display_name, code_chunks, doc_chunks)
            VALUES (?, ?, 0, 0)
            ON CONFLICT(hive_name) DO NOTHING
        """, (hive_name, display_name))

        if index_type == "code":
            self._conn.execute("""
                UPDATE hive_metadata
                SET last_code_index = ?, code_chunks = ?,
                    display_name = COALESCE(NULLIF(?, ''), display_name),
                    index_duration_s = ?
                WHERE hive_name = ?
            """, (now, chunks, display_name, duration_s, hive_name))
        else:
            self._conn.execute("""
                UPDATE hive_metadata
                SET last_doc_index = ?, doc_chunks = ?,
                    display_name = COALESCE(NULLIF(?, ''), display_name)
                WHERE hive_name = ?
            """, (now, chunks, display_name, hive_name))

        # Add to history
        self._conn.execute("""
            INSERT INTO index_history (hive_name, index_type, started_at, finished_at,
                                       chunks, duration_s, status)
            VALUES (?, ?, ?, ?, ?, ?, 'success')
        """, (hive_name, index_type, now, now, chunks, duration_s))

        self._conn.commit()
        log.info("Discovery store updated: %s %s → %d chunks", hive_name, index_type, chunks)

    def get_metadata(self, hive_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a hive."""
        row = self._conn.execute(
            "SELECT * FROM hive_metadata WHERE hive_name = ?", (hive_name,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """Get metadata for all hives."""
        rows = self._conn.execute(
            "SELECT * FROM hive_metadata ORDER BY hive_name"
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Topics ──────────────────────────────────────────────────

    def set_topics(
        self,
        hive_name: str,
        topics: List[str],
        source: str = "auto",
    ):
        """Replace auto-generated topics for a hive. Manual topics are preserved."""
        now = datetime.now(timezone.utc).isoformat()

        # Remove old topics of same source
        self._conn.execute(
            "DELETE FROM hive_topics WHERE hive_name = ? AND source = ?",
            (hive_name, source),
        )

        # Insert new
        for topic in topics:
            self._conn.execute("""
                INSERT OR REPLACE INTO hive_topics (hive_name, topic, source, created_at)
                VALUES (?, ?, ?, ?)
            """, (hive_name, topic.strip().lower(), source, now))

        # Update refresh timestamp
        self._conn.execute("""
            UPDATE hive_metadata SET last_topic_refresh = ? WHERE hive_name = ?
        """, (now, hive_name))

        self._conn.commit()
        log.info("Updated %d %s topics for hive '%s'", len(topics), source, hive_name)

    def get_topics(self, hive_name: str, source: Optional[str] = None) -> List[str]:
        """Get topics for a hive. If source is None, returns all."""
        if source:
            rows = self._conn.execute(
                "SELECT topic FROM hive_topics WHERE hive_name = ? AND source = ? ORDER BY topic",
                (hive_name, source),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT topic FROM hive_topics WHERE hive_name = ? ORDER BY topic",
                (hive_name,),
            ).fetchall()
        return [r["topic"] for r in rows]

    def get_all_topics(self) -> Dict[str, List[str]]:
        """Get topics for all hives."""
        rows = self._conn.execute(
            "SELECT hive_name, topic FROM hive_topics ORDER BY hive_name, topic"
        ).fetchall()
        result: Dict[str, List[str]] = {}
        for r in rows:
            result.setdefault(r["hive_name"], []).append(r["topic"])
        return result

    # ── Staleness ───────────────────────────────────────────────

    def get_stale_hives(self, threshold_days: int = STALE_THRESHOLD_DAYS) -> List[Dict[str, Any]]:
        """Return hives that haven't been re-indexed recently."""
        stale = []
        for meta in self.get_all_metadata():
            last_code = meta.get("last_code_index")
            if not last_code:
                stale.append({**meta, "stale_days": None, "reason": "never indexed"})
                continue
            last_dt = datetime.fromisoformat(last_code)
            age = datetime.now(timezone.utc) - last_dt
            if age.days >= threshold_days:
                stale.append({**meta, "stale_days": age.days, "reason": f"{age.days} days old"})
        return stale

    def is_stale(self, hive_name: str, threshold_days: int = STALE_THRESHOLD_DAYS) -> bool:
        """Check if a specific hive is stale."""
        meta = self.get_metadata(hive_name)
        if not meta or not meta.get("last_code_index"):
            return True
        last_dt = datetime.fromisoformat(meta["last_code_index"])
        age = datetime.now(timezone.utc) - last_dt
        return age.days >= threshold_days

    # ── Index History ───────────────────────────────────────────

    def get_index_history(self, hive_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent index history for a hive."""
        rows = self._conn.execute(
            "SELECT * FROM index_history WHERE hive_name = ? ORDER BY id DESC LIMIT ?",
            (hive_name, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Cleanup ─────────────────────────────────────────────────

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

"""
Audit Logger — SQLite persistence.

Every submission and appeal is written as a structured row in the
audit_log table.  The schema follows planning.md §6 and supports
both classification and appeal event types.

For M3 the LLM classifier fields are NULL — only the stylometric
signal (signal_2) is live.  M4 adds the LLM classifier; M5 adds
the appeal workflow.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any


DB_PATH = "audit_log.db"


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create the audit_log table if it doesn't already exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS audit_log (
            entry_id        TEXT PRIMARY KEY,
            timestamp       TEXT NOT NULL,
            event_type      TEXT NOT NULL,             -- 'classification' | 'appeal'
            content_id      TEXT NOT NULL,
            content_length  INTEGER,
            creator_id      TEXT,

            -- Classification fields (null for appeal entries)
            label           TEXT,                      -- 'ai' | 'human' | 'uncertain'
            confidence      REAL,                      -- 0.0 – 1.0
            combined_score  REAL,                      -- 0.0 – 1.0

            -- Signal scores
            signal_1_name   TEXT DEFAULT 'llm_classifier',
            signal_1_score  REAL,
            signal_1_detail TEXT,                      -- JSON
            signal_2_name   TEXT DEFAULT 'stylometric',
            signal_2_score  REAL,
            signal_2_detail TEXT,                      -- JSON

            transparency_label_variant TEXT,           -- 'A' | 'B' | 'C'
            submitter_ip_hash TEXT,
            processing_time_ms INTEGER,

            -- Status tracking
            status          TEXT DEFAULT 'classified', -- 'classified' | 'under_review'

            -- Appeal fields (null for classification entries)
            linked_entry_id TEXT,
            creator_reason  TEXT,
            original_label  TEXT,
            original_confidence REAL
        );

        CREATE INDEX IF NOT EXISTS idx_audit_content_id
            ON audit_log(content_id);
        CREATE INDEX IF NOT EXISTS idx_audit_timestamp
            ON audit_log(timestamp);
        CREATE INDEX IF NOT EXISTS idx_audit_status
            ON audit_log(status);
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def insert_entry(entry: dict[str, Any]) -> str:
    """Insert a row into audit_log.  Returns the entry_id."""
    conn = sqlite3.connect(DB_PATH)
    entry_id = entry.get("entry_id", str(uuid.uuid4()))

    conn.execute("""
        INSERT INTO audit_log (
            entry_id, timestamp, event_type, content_id, content_length,
            creator_id, label, confidence, combined_score,
            signal_1_name, signal_1_score, signal_1_detail,
            signal_2_name, signal_2_score, signal_2_detail,
            transparency_label_variant, submitter_ip_hash, processing_time_ms,
            status, linked_entry_id, creator_reason,
            original_label, original_confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        entry_id,
        entry.get("timestamp", _utcnow()),
        entry.get("event_type", "classification"),
        entry.get("content_id", ""),
        entry.get("content_length"),
        entry.get("creator_id"),
        entry.get("label"),
        entry.get("confidence"),
        entry.get("combined_score"),
        entry.get("signal_1_name"),
        entry.get("signal_1_score"),
        _json_dump(entry.get("signal_1_detail")),
        entry.get("signal_2_name", "stylometric"),
        entry.get("signal_2_score"),
        _json_dump(entry.get("signal_2_detail")),
        entry.get("transparency_label_variant"),
        entry.get("submitter_ip_hash"),
        entry.get("processing_time_ms"),
        entry.get("status", "classified"),
        entry.get("linked_entry_id"),
        entry.get("creator_reason"),
        entry.get("original_label"),
        entry.get("original_confidence"),
    ))

    conn.commit()
    conn.close()
    return entry_id


def lookup_by_content_id(content_id: str) -> dict[str, Any] | None:
    """Return the most recent classification row for *content_id*, or None."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM audit_log WHERE content_id = ? AND event_type = 'classification' ORDER BY timestamp DESC LIMIT 1",
        (content_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_recent_entries(
    limit: int = 20,
    content_id: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """Return recent audit log entries, optionally filtered by content_id and/or status."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    where: list[str] = []
    params: list[Any] = []

    if content_id:
        where.append("content_id = ?")
        params.append(content_id)
    if status:
        where.append("status = ?")
        params.append(status)

    if where:
        sql = f"SELECT * FROM audit_log WHERE {' AND '.join(where)} ORDER BY timestamp DESC LIMIT ?"
    else:
        sql = "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, tuple(params)).fetchall()
    conn.close()

    return [_row_to_dict(r) for r in rows]


def update_status(content_id: str, new_status: str) -> bool:
    """Update the status of all entries for a given content_id."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE audit_log SET status = ? WHERE content_id = ?",
        (new_status, content_id),
    )
    conn.commit()
    affected = conn.total_changes
    conn.close()
    return affected > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(obj: Any) -> str | None:
    if obj is None:
        return None
    return json.dumps(obj)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a sqlite3.Row to a plain dict, parsing JSON detail fields."""
    d = dict(row)
    for field in ("signal_1_detail", "signal_2_detail"):
        if d.get(field) and isinstance(d[field], str):
            try:
                d[field] = json.loads(d[field])
            except json.JSONDecodeError:
                pass
    return d

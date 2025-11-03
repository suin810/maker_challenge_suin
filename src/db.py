from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional


def _connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    conn = _connect(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meetings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                meeting_date TEXT NOT NULL,
                audio_path TEXT,
                transcript_text TEXT,
                segments_json TEXT,
                participants_json TEXT,
                meta_json TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
            """
        )
    conn.close()


def create_meeting(
    db_path: str,
    name: str,
    meeting_date: str,
    audio_path: Optional[str],
    transcript_text: str,
    segments: List[Dict[str, Any]],
    participants: Dict[str, str],
    meta: Optional[Dict[str, Any]] = None,
) -> int:
    conn = _connect(db_path)
    with conn:
        cur = conn.execute(
            """
            INSERT INTO meetings (name, meeting_date, audio_path, transcript_text, segments_json, participants_json, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                meeting_date,
                audio_path,
                transcript_text,
                json.dumps(segments, ensure_ascii=False),
                json.dumps(participants, ensure_ascii=False),
                json.dumps(meta or {}, ensure_ascii=False),
            ),
        )
        meeting_id = int(cur.lastrowid)
    conn.close()
    return meeting_id


def list_meetings(db_path: str) -> List[Dict[str, Any]]:
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT id, name, meeting_date, created_at FROM meetings ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_meeting(db_path: str, meeting_id: int) -> Optional[Dict[str, Any]]:
    conn = _connect(db_path)
    row = conn.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
    conn.close()
    if not row:
        return None
    data = dict(row)
    # decode json fields
    for key in ("segments_json", "participants_json", "meta_json"):
        try:
            if key == "segments_json":
                data[key] = json.loads(data.get(key) or "[]")
            else:
                data[key] = json.loads(data.get(key) or "{}")
        except json.JSONDecodeError:
            data[key] = [] if key == "segments_json" else {}
    return data

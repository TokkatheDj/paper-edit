"""Project storage and the job queue.

Everything lives in SQLite, deliberately. A desktop can be shut down or lose
power mid-job, and a two-hour transcription held only in memory would vanish
with it. Job state on disk means an interrupted job is visible as interrupted on
the next start instead of silently disappearing.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = ROOT / "projects"
DB_PATH = PROJECTS / "paperedit.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'new',
    source      TEXT,
    proxy       TEXT,
    duration    REAL DEFAULT 0,
    fps         REAL DEFAULT 0,
    width       INTEGER DEFAULT 0,
    height      INTEGER DEFAULT 0,
    has_video   INTEGER DEFAULT 0,
    created     REAL NOT NULL,
    silence_on   INTEGER DEFAULT 0,
    silence_keep REAL DEFAULT 0.3,
    silence_min  REAL DEFAULT 0.6,
    sound_preset TEXT DEFAULT 'auto',
    sound_snr    REAL DEFAULT 0,
    caption_style TEXT DEFAULT 'off'
);
CREATE TABLE IF NOT EXISTS words (
    project_id  TEXT NOT NULL,
    idx         INTEGER NOT NULL,
    text        TEXT NOT NULL,
    start       REAL NOT NULL,
    end         REAL NOT NULL,
    speaker     TEXT,
    deleted     INTEGER NOT NULL DEFAULT 0,
    probability REAL DEFAULT 1.0,
    PRIMARY KEY (project_id, idx)
);
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL,
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL,
    progress    REAL DEFAULT 0,
    message     TEXT DEFAULT '',
    result      TEXT DEFAULT '',
    created     REAL NOT NULL,
    updated     REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS auth (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    salt        BLOB NOT NULL,
    hash        BLOB NOT NULL,
    created     REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token       TEXT PRIMARY KEY,
    created     REAL NOT NULL,
    expires     REAL NOT NULL,
    label       TEXT DEFAULT '',
    address     TEXT DEFAULT '',
    last_seen   REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS jobs_project ON jobs(project_id);
"""

_local = threading.local()


def connect() -> sqlite3.Connection:
    """One connection per thread; jobs run off-thread from requests."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        PROJECTS.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")      # survives an abrupt power cut
        conn.execute("PRAGMA busy_timeout=30000")
        conn.executescript(SCHEMA)
        _local.conn = conn
    return conn


# Columns added after the first release. SQLite has no "ADD COLUMN IF NOT
# EXISTS", so we add them and shrug off the duplicate error -- simpler and safer
# than tracking a migration version for a single-user local app.
# Columns added after the first release. CREATE TABLE IF NOT EXISTS will not
# add a column to a table that already exists, so every one of these has to be
# ALTERed in on start-up -- an existing database must never need a hand edit.
_LATER_COLUMNS = [
    ("projects", "silence_on", "INTEGER DEFAULT 0"),
    ("projects", "silence_keep", "REAL DEFAULT 0.3"),
    ("projects", "silence_min", "REAL DEFAULT 0.6"),
    ("projects", "sound_preset", "TEXT DEFAULT 'auto'"),
    ("projects", "sound_snr", "REAL DEFAULT 0"),
    ("projects", "caption_style", "TEXT DEFAULT 'off'"),
    ("sessions", "label", "TEXT DEFAULT ''"),
    ("sessions", "address", "TEXT DEFAULT ''"),
    ("sessions", "last_seen", "REAL DEFAULT 0"),
]


def init() -> None:
    conn = connect()
    for table, name, decl in _LATER_COLUMNS:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
        except sqlite3.OperationalError:
            pass                      # already there
    conn.commit()
    # A job marked running at startup cannot be running -- nothing survived the
    # restart. Mark it plainly rather than leaving a spinner forever.
    conn.execute(
        "UPDATE jobs SET status='interrupted', message='interrupted by restart or shutdown',"
        " updated=? WHERE status IN ('queued','running')", (time.time(),))
    conn.commit()


# --------------------------------------------------------------------------- projects

def project_dir(pid: str) -> Path:
    d = PROJECTS / pid
    d.mkdir(parents=True, exist_ok=True)
    return d


def create_project(name: str) -> str:
    pid = uuid.uuid4().hex[:12]
    conn = connect()
    conn.execute("INSERT INTO projects (id, name, created) VALUES (?,?,?)",
                 (pid, name, time.time()))
    conn.commit()
    project_dir(pid)
    return pid


def get_project(pid: str) -> dict[str, Any] | None:
    row = connect().execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    return dict(row) if row else None


def list_projects() -> list[dict[str, Any]]:
    rows = connect().execute(
        "SELECT * FROM projects ORDER BY created DESC").fetchall()
    return [dict(r) for r in rows]


def update_project(pid: str, **fields) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    conn = connect()
    conn.execute(f"UPDATE projects SET {cols} WHERE id=?", (*fields.values(), pid))
    conn.commit()


def delete_project(pid: str) -> None:
    conn = connect()
    conn.execute("DELETE FROM words WHERE project_id=?", (pid,))
    conn.execute("DELETE FROM jobs WHERE project_id=?", (pid,))
    conn.execute("DELETE FROM projects WHERE id=?", (pid,))
    conn.commit()


# --------------------------------------------------------------------------- words

def set_words(pid: str, words: Iterable[dict]) -> int:
    conn = connect()
    conn.execute("DELETE FROM words WHERE project_id=?", (pid,))
    rows = [(pid, i, w["text"], w["start"], w["end"], w.get("speaker"),
             int(w.get("deleted", 0)), w.get("probability", 1.0))
            for i, w in enumerate(words)]
    conn.executemany(
        "INSERT INTO words (project_id, idx, text, start, end, speaker, deleted,"
        " probability) VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    return len(rows)


def get_words(pid: str) -> list[dict[str, Any]]:
    rows = connect().execute(
        "SELECT idx, text, start, end, speaker, deleted, probability FROM words"
        " WHERE project_id=? ORDER BY idx", (pid,)).fetchall()
    return [dict(r) for r in rows]


def set_deleted(pid: str, indices: Iterable[int], deleted: bool) -> int:
    idx = list(indices)
    if not idx:
        return 0
    conn = connect()
    qs = ",".join("?" * len(idx))
    cur = conn.execute(
        f"UPDATE words SET deleted=? WHERE project_id=? AND idx IN ({qs})",
        (int(deleted), pid, *idx))
    conn.commit()
    return cur.rowcount


# --------------------------------------------------------------------------- jobs

def create_job(pid: str, kind: str) -> str:
    jid = uuid.uuid4().hex[:12]
    now = time.time()
    conn = connect()
    conn.execute(
        "INSERT INTO jobs (id, project_id, kind, status, created, updated)"
        " VALUES (?,?,?,'queued',?,?)", (jid, pid, kind, now, now))
    conn.commit()
    return jid


def update_job(jid: str, **fields) -> None:
    fields["updated"] = time.time()
    cols = ", ".join(f"{k}=?" for k in fields)
    conn = connect()
    conn.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), jid))
    conn.commit()


def get_job(jid: str) -> dict[str, Any] | None:
    row = connect().execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
    return dict(row) if row else None


def latest_job(pid: str, kind: str | None = None) -> dict[str, Any] | None:
    q = "SELECT * FROM jobs WHERE project_id=?"
    args: list[Any] = [pid]
    if kind:
        q += " AND kind=?"
        args.append(kind)
    q += " ORDER BY created DESC LIMIT 1"
    row = connect().execute(q, args).fetchone()
    return dict(row) if row else None


def run_in_background(fn, *args, **kwargs) -> None:
    threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True).start()

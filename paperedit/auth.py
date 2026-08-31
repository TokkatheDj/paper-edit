"""One password for the house, and sessions that remember it.

The threat model is small and worth stating: this app has no accounts and no
roles. It is two people on a home network, and the job is to stop a device
that wanders onto the wifi -- a guest, a kid's tablet, a neighbour who guessed
the wifi key -- from reading and deleting unreleased videos. It is NOT hardened
for the open internet, and putting it there is still the wrong move.

So: one shared password, stored as an scrypt hash, and opaque session tokens
kept server-side in SQLite. Server-side sessions rather than signed cookies
because they can be revoked -- signing out actually signs you out, and it
survives a restart, which matters when the desktop reboots nightly.
"""
from __future__ import annotations

import hmac
import secrets
import time
from typing import Any

from . import store

SESSION_DAYS = 30
COOKIE = "pe_session"

# scrypt at these parameters takes ~100ms here: slow enough to make guessing
# over the network pointless, fast enough that signing in feels instant.
_N, _R, _P, _DKLEN = 2 ** 14, 8, 1, 32

# Login attempts, per client address. Kept in memory on purpose: a restart
# clearing it is fine, and it must never grow into a reason the editor is down.
_fails: dict[str, tuple[int, float]] = {}
MAX_FAILS = 5
LOCKOUT = 60.0


def _hash(password: str, salt: bytes) -> bytes:
    import hashlib
    return hashlib.scrypt(password.encode("utf-8"), salt=salt,
                          n=_N, r=_R, p=_P, dklen=_DKLEN)


def has_password() -> bool:
    return store.connect().execute(
        "SELECT 1 FROM auth WHERE id=1").fetchone() is not None


def set_password(password: str) -> None:
    """Set or replace the password. Every existing session is dropped, so
    changing it actually kicks out whoever was signed in."""
    if len(password) < 4:
        raise ValueError("password too short")
    salt = secrets.token_bytes(16)
    conn = store.connect()
    conn.execute("INSERT OR REPLACE INTO auth (id, salt, hash, created)"
                 " VALUES (1,?,?,?)", (salt, _hash(password, salt), time.time()))
    conn.execute("DELETE FROM sessions")
    conn.commit()
    _fails.clear()


def check_password(password: str) -> bool:
    row = store.connect().execute(
        "SELECT salt, hash FROM auth WHERE id=1").fetchone()
    if row is None:
        return False
    return hmac.compare_digest(_hash(password, row["salt"]), row["hash"])


def locked_out(who: str) -> float:
    """Seconds still to wait, or 0. Slows guessing without ever locking the
    house out for long -- a wrong password is far more likely to be a typo."""
    n, until = _fails.get(who, (0, 0.0))
    return max(0.0, until - time.time()) if n >= MAX_FAILS else 0.0


def note_failure(who: str) -> None:
    n, _ = _fails.get(who, (0, 0.0))
    _fails[who] = (n + 1, time.time() + LOCKOUT)


def clear_failures(who: str) -> None:
    _fails.pop(who, None)


def new_session() -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    conn = store.connect()
    conn.execute("INSERT INTO sessions (token, created, expires) VALUES (?,?,?)",
                 (token, now, now + SESSION_DAYS * 86400))
    conn.execute("DELETE FROM sessions WHERE expires < ?", (now,))
    conn.commit()
    return token


def valid_session(token: str | None) -> bool:
    if not token:
        return False
    row = store.connect().execute(
        "SELECT expires FROM sessions WHERE token=?", (token,)).fetchone()
    return bool(row) and row["expires"] > time.time()


def end_session(token: str | None) -> None:
    if not token:
        return
    conn = store.connect()
    conn.execute("DELETE FROM sessions WHERE token=?", (token,))
    conn.commit()

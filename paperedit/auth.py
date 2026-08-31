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


# How often a session's "last seen" is written. Every request would mean a
# database write per request for no extra insight.
TOUCH_EVERY = 300.0


def device_label(user_agent: str | None) -> str:
    """A short, human-readable name for whatever signed in.

    Deliberately coarse: enough to tell a phone from a laptop from a script,
    and no more. Scripts are checked first and named as such, because the most
    confusing thing in this list is your own automated check looking like a
    person. Kept to plain ASCII -- this gets printed to a Windows console.
    """
    ua = user_agent or ""
    for needle, name in (("PowerShell", "PowerShell script"),
                         ("curl", "curl script"),
                         ("python", "Python script"),
                         ("httpx", "Python script"),
                         ("Wget", "Wget script")):
        if needle.lower() in ua.lower():
            return name

    if "iPhone" in ua:
        system = "iPhone"
    elif "iPad" in ua:
        system = "iPad"
    elif "Android" in ua:
        system = "Android"
    elif "Windows" in ua:
        system = "Windows"
    elif "Macintosh" in ua or "Mac OS X" in ua:
        system = "Mac"
    elif "Linux" in ua:
        system = "Linux"
    else:
        return "Unknown device"

    # Order matters: Edge claims Chrome and Safari, Chrome claims Safari.
    if "Edg" in ua:
        browser = "Edge"
    elif "OPR/" in ua or "Opera" in ua:
        browser = "Opera"
    elif "Firefox/" in ua or "FxiOS" in ua:
        browser = "Firefox"
    elif "Chrome/" in ua or "CriOS" in ua:
        browser = "Chrome"
    elif "Safari/" in ua:
        browser = "Safari"
    else:
        return system
    return system + " / " + browser


def new_session(label: str = "", address: str = "") -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    conn = store.connect()
    conn.execute("INSERT INTO sessions (token, created, expires, label, address,"
                 " last_seen) VALUES (?,?,?,?,?,?)",
                 (token, now, now + SESSION_DAYS * 86400, label[:60], address[:45], now))
    conn.execute("DELETE FROM sessions WHERE expires < ?", (now,))
    conn.commit()
    return token


def see_session(token: str | None) -> bool:
    """Check a session and note that it is still in use.

    The write is throttled: knowing a device was active in the last five
    minutes is as useful as knowing it to the second, and costs one write an
    hour instead of one per request.
    """
    if not token:
        return False
    conn = store.connect()
    row = conn.execute("SELECT expires, last_seen FROM sessions WHERE token=?",
                       (token,)).fetchone()
    now = time.time()
    if not row or row["expires"] <= now:
        return False
    if now - (row["last_seen"] or 0) > TOUCH_EVERY:
        conn.execute("UPDATE sessions SET last_seen=? WHERE token=?", (now, token))
        conn.commit()
    return True


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

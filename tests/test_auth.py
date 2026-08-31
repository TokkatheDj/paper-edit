"""Sign-in: the thing standing between a guest device and her unreleased videos.

Every one of these is a claim someone could otherwise take on trust.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASSWORD = "kitchen-table-2026"


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from paperedit import auth, store
    store.init()
    conn = store.connect()
    conn.execute("DELETE FROM auth")
    conn.execute("DELETE FROM sessions")
    conn.commit()
    auth._fails.clear()
    import server
    with TestClient(server.app) as c:
        yield c


def test_the_editor_is_closed_before_a_password_exists(client):
    """No password set is not the same as no lock: the API still says no."""
    assert client.get("/api/projects").status_code == 401
    assert client.get("/api/health").status_code == 401


def test_a_browser_is_sent_to_the_sign_in_page(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


def test_the_sign_in_page_itself_is_reachable(client):
    assert client.get("/login").status_code == 200
    assert client.get("/api/session").json()["needs_setup"] is True


def test_first_run_sets_the_password_and_signs_in(client):
    r = client.post("/api/session", json={"password": PASSWORD})
    assert r.status_code == 200
    from paperedit import auth
    assert auth.COOKIE in r.cookies or auth.COOKIE in client.cookies
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/session").json() == {"signed_in": True,
                                                 "needs_setup": False}


def test_a_short_password_is_refused(client):
    assert client.post("/api/session", json={"password": "no"}).status_code == 400
    from paperedit import auth
    assert not auth.has_password()


def test_the_wrong_password_gets_nowhere(client):
    client.post("/api/session", json={"password": PASSWORD})
    client.delete("/api/session")
    assert client.post("/api/session", json={"password": "guess"}).status_code == 401
    assert client.get("/api/projects").status_code == 401


def test_signing_out_actually_revokes_the_session(client):
    """Server-side sessions exist for this: the cookie alone must not be enough
    once it has been signed out, even if a browser hangs on to it."""
    from paperedit import auth
    client.post("/api/session", json={"password": PASSWORD})
    token = client.cookies.get(auth.COOKIE)
    assert client.get("/api/health").status_code == 200

    client.delete("/api/session")
    assert client.get("/api/health").status_code == 401

    client.cookies.set(auth.COOKIE, token)          # replay the old cookie
    assert client.get("/api/health").status_code == 401


def test_repeated_guessing_gets_throttled(client):
    client.post("/api/session", json={"password": PASSWORD})
    client.delete("/api/session")
    for _ in range(5):
        assert client.post("/api/session", json={"password": "x"}).status_code == 401
    r = client.post("/api/session", json={"password": "x"})
    assert r.status_code == 429
    # The real password is refused too while throttled -- that is the point.
    assert client.post("/api/session", json={"password": PASSWORD}).status_code == 429


def test_changing_the_password_signs_everyone_out(client):
    from paperedit import auth
    client.post("/api/session", json={"password": PASSWORD})
    assert client.get("/api/health").status_code == 200
    auth.set_password("something-else-entirely")
    assert client.get("/api/health").status_code == 401


def test_a_finished_export_is_not_downloadable_while_signed_out(client):
    """The exports are the whole point of protecting this -- her unreleased
    video sits behind /file/ and must not be fetchable without signing in."""
    r = client.get("/api/projects/anything/file/export.mp4")
    assert r.status_code == 401


# --------------------------------------------------------------- device labels

@pytest.mark.parametrize("ua,expected", [
    ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15"
     " (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1", "iPhone / Safari"),
    ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
     " (KHTML, like Gecko) Chrome/120.0 Safari/537.36", "Mac / Chrome"),
    # Edge claims to be Chrome AND Safari; Chrome claims to be Safari. Order matters.
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like"
     " Gecko) Chrome/120.0 Safari/537.36 Edg/120.0", "Windows / Edge"),
    ("Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 Chrome/120"
     " Mobile Safari/537.36", "Android / Chrome"),
    # The whole point: an automated check must not look like a person.
    ("Mozilla/5.0 (Windows NT 10.0) WindowsPowerShell/5.1", "PowerShell script"),
    ("curl/8.4.0", "curl script"),
    ("python-httpx/0.27.0", "Python script"),
    ("", "Unknown device"),
    (None, "Unknown device"),
])
def test_device_labels_read_like_devices(ua, expected):
    from paperedit.auth import device_label
    assert device_label(ua) == expected


def test_labels_stay_ascii():
    """These get printed to a Windows console by Editor Activity.cmd, where a
    stray non-ASCII character is an exception rather than a cosmetic issue."""
    from paperedit.auth import device_label
    for ua in ("Mozilla/5.0 (iPhone) Safari/604.1", "curl/8.4.0", "nonsense"):
        device_label(ua).encode("ascii")


def test_a_sign_in_records_what_signed_in(client):
    from paperedit import store
    client.post("/api/session", json={"password": PASSWORD},
                headers={"User-Agent": "Mozilla/5.0 (iPhone) Version/17.0 Safari/604.1"})
    row = store.connect().execute(
        "SELECT label, address, created, last_seen FROM sessions").fetchone()
    assert row["label"] == "iPhone / Safari"
    assert row["address"]                        # whatever the client was
    assert row["last_seen"] >= row["created"]


def test_last_seen_moves_when_the_session_is_used(client):
    """Distinguishing a device that was used this morning from one that signed
    in a fortnight ago is the entire point of the column."""
    import time as _t
    from paperedit import store
    client.post("/api/session", json={"password": PASSWORD})
    conn = store.connect()
    # Backdate past the throttle window so the next request must write.
    conn.execute("UPDATE sessions SET last_seen = ?", (_t.time() - 600,))
    conn.commit()
    before = conn.execute("SELECT last_seen FROM sessions").fetchone()["last_seen"]

    assert client.get("/api/health").status_code == 200
    after = conn.execute("SELECT last_seen FROM sessions").fetchone()["last_seen"]
    assert after > before


def test_last_seen_is_not_written_on_every_request(client):
    """One write per request would be a database write for every video chunk."""
    from paperedit import store
    client.post("/api/session", json={"password": PASSWORD})
    conn = store.connect()
    first = conn.execute("SELECT last_seen FROM sessions").fetchone()["last_seen"]
    for _ in range(3):
        client.get("/api/health")
    assert conn.execute(
        "SELECT last_seen FROM sessions").fetchone()["last_seen"] == first

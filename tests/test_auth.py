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

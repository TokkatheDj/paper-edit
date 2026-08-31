"""Keep the suite away from the real projects database.

That database holds real recordings and, since sign-in was added, the household
password. A test that called set_password() against it would sign both of them
out; tests that create and delete projects were already leaving stray folders
in the live projects directory. Redirecting storage to a temp directory fixes
both, and it has to happen before anything opens a connection.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEST_PASSWORD = "not-the-real-one"


@pytest.fixture(scope="session", autouse=True)
def isolated_storage(tmp_path_factory):
    from paperedit import store
    d = tmp_path_factory.mktemp("paperedit-tests")
    store.PROJECTS = d
    store.DB_PATH = d / "paperedit.db"
    yield


def sign_in(client):
    """Sign a test client in, whatever the auth tests left behind.

    The suite shares one temp database, and test_auth.py deliberately changes
    the password and trips the guess throttle. Neither should decide whether an
    unrelated test can reach the API, so this forces a known password and
    clears the throttle rather than assuming a first-run database.
    """
    from paperedit import auth
    if not auth.has_password() or not auth.check_password(TEST_PASSWORD):
        auth.set_password(TEST_PASSWORD)
    auth._fails.clear()
    r = client.post("/api/session", json={"password": TEST_PASSWORD})
    assert r.status_code == 200, r.text
    return client

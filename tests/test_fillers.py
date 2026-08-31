"""The filler-word button, tested without Whisper in the way.

The end-to-end test in test_api.py can only remove the fillers Whisper chose to
write down, and on the AMI clip that is none -- which is the 7-17% recall from
FINDINGS.md showing up in the test suite. So the endpoint's own behaviour is
pinned here against seeded words instead, where the expected answer is known.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED = [
    ("So",         0.0, 0.4),
    ("um",         0.5, 0.7),      # filler
    ("the",        0.8, 1.0),
    ("umbrella",   1.1, 1.7),      # NOT a filler, merely starts with "um"
    ("Uh,",        1.8, 2.0),      # filler: capitalised and punctuated
    ("was",        2.1, 2.4),
    ("hmm",        2.5, 2.9),      # filler
    ("outside.",   3.0, 3.6),
]
FILLER_IDX = [1, 4, 6]


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from paperedit import store
    store.init()
    import server
    with TestClient(server.app) as c:
        yield c


@pytest.fixture()
def project(client):
    from paperedit import store
    pid = client.post("/api/projects", json={"name": "fillers"}).json()["id"]
    store.set_words(pid, [{"text": t, "start": s, "end": e} for t, s, e in SEED])
    store.update_project(pid, duration=SEED[-1][2], status="ready")
    yield pid
    client.delete(f"/api/projects/{pid}")


def test_it_removes_exactly_the_standalone_fillers(client, project):
    r = client.post(f"/api/projects/{project}/fillers", json={"deleted": True}).json()
    assert r["indices"] == FILLER_IDX
    assert r["marked"] == 3

    words = client.get(f"/api/projects/{project}/words").json()
    assert [w["idx"] for w in words if w["deleted"]] == FILLER_IDX
    # "umbrella" surviving is the point: the match is on the whole word.
    assert not any(w["deleted"] for w in words if w["text"] == "umbrella")


def test_the_edit_gets_shorter_then_comes_back(client, project):
    before = client.get(f"/api/projects/{project}/plan").json()["duration"]

    cut = client.post(f"/api/projects/{project}/fillers", json={"deleted": True}).json()
    assert cut["duration"] < before

    back = client.post(f"/api/projects/{project}/fillers", json={"deleted": False}).json()
    assert back["duration"] == pytest.approx(before, abs=0.01)
    assert not any(w["deleted"] for w in
                   client.get(f"/api/projects/{project}/words").json())


def test_a_transcript_with_no_fillers_is_a_no_op(client):
    """The vertical recording is exactly this case, so it must not error."""
    from paperedit import store
    pid = client.post("/api/projects", json={"name": "clean"}).json()["id"]
    store.set_words(pid, [{"text": "A", "start": 0.0, "end": 0.5},
                          {"text": "clean", "start": 0.6, "end": 1.0},
                          {"text": "take.", "start": 1.1, "end": 1.6}])
    store.update_project(pid, duration=1.6, status="ready")
    try:
        r = client.post(f"/api/projects/{pid}/fillers", json={"deleted": True}).json()
        assert r["marked"] == 0 and r["indices"] == []
        assert r["duration"] == pytest.approx(1.6, abs=0.05)
    finally:
        client.delete(f"/api/projects/{pid}")

"""End-to-end: a real file through upload, transcribe, edit and export.

This is the Phase 1 verification from the plan. It uses the AMI test video built
by spikes/cut_quality.py and skips if that is not present, so the suite still
runs on a clean checkout.
"""
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MEDIA = ROOT / "spikes" / "media" / "test-300s.mp4"
pytestmark = pytest.mark.skipif(
    not MEDIA.exists(), reason="run spikes/cut_quality.py first to build the test media")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from paperedit import store
    store.init()
    import server
    with TestClient(server.app) as c:
        yield c


def _wait(client, jid, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "failed", "interrupted"):
            return j
        time.sleep(1)
    raise AssertionError("job timed out")


@pytest.fixture(scope="module")
def project(client):
    pid = client.post("/api/projects", json={"name": "e2e"}).json()["id"]
    yield pid
    client.delete(f"/api/projects/{pid}")


def test_upload_is_resumable(client, project):
    """A dropped connection mid-upload must resume, not restart."""
    data = MEDIA.read_bytes()
    half = len(data) // 2

    r = client.put(f"/api/projects/{project}/upload?name=src.mp4&offset=0",
                   content=data[:half])
    assert r.status_code == 200 and r.json()["received"] == half

    # Simulate the browser reconnecting and asking where it got to.
    assert client.get(f"/api/projects/{project}/upload?name=src.mp4"
                      ).json()["received"] == half

    # A chunk that starts past what the server holds would silently corrupt the
    # file, so it must be refused.
    assert client.put(f"/api/projects/{project}/upload?name=src.mp4&offset={len(data)+10}",
                      content=b"x").status_code == 409

    r = client.put(f"/api/projects/{project}/upload?name=src.mp4&offset={half}&final=1",
                   content=data[half:])
    assert r.json()["received"] == len(data)

    from paperedit import store
    assert (store.project_dir(project) / "src.mp4").read_bytes() == data


def test_ingest_produces_transcript_and_proxy(client, project):
    jid = client.post(f"/api/projects/{project}/ingest").json()["job"]
    job = _wait(client, jid)
    assert job["status"] == "done", job["message"]

    p = client.get(f"/api/projects/{project}").json()
    assert p["status"] == "ready"
    assert p["duration"] == pytest.approx(300, abs=2)
    assert p["fps"] == pytest.approx(30, abs=0.1)
    assert p["word_count"] > 200

    words = client.get(f"/api/projects/{project}/words").json()
    assert all(w["end"] > w["start"] for w in words)
    assert words == sorted(words, key=lambda w: w["start"])
    assert Path(p["proxy"]).exists()
    assert client.get(f"/api/projects/{project}/proxy").status_code == 200


def test_deleting_words_shortens_the_edit(client, project):
    before = client.get(f"/api/projects/{project}/plan").json()
    words = client.get(f"/api/projects/{project}/words").json()
    doomed = [w["idx"] for w in words if 100.0 <= w["start"] <= 140.0]
    assert doomed, "expected words in that range"

    r = client.post(f"/api/projects/{project}/words",
                    json={"indices": doomed, "deleted": True}).json()
    assert r["updated"] == len(doomed)
    assert r["duration"] < before["duration"]

    after = client.get(f"/api/projects/{project}/plan").json()
    # NOT "more cuts": deleting a contiguous block swallows the gaps that were
    # already inside it, so the cut COUNT can fall. What must hold is that no
    # surviving cut still covers the deleted span.
    assert not any(c["start"] < 120.0 < c["end"] for c in after["cuts"])
    for c in after["cuts"]:
        assert c["end"] > c["start"]
    for a, b in zip(after["cuts"], after["cuts"][1:]):
        assert a["end"] <= b["start"]                     # no overlap

    # Restoring must put the time back.
    client.post(f"/api/projects/{project}/words",
                json={"indices": doomed, "deleted": False})
    assert client.get(f"/api/projects/{project}/plan").json()["duration"] == \
        pytest.approx(before["duration"], abs=0.05)


def test_export_matches_the_plan(client, project):
    words = client.get(f"/api/projects/{project}/words").json()
    doomed = [w["idx"] for w in words if 60.0 <= w["start"] <= 90.0]
    client.post(f"/api/projects/{project}/words",
                json={"indices": doomed, "deleted": True})
    plan = client.get(f"/api/projects/{project}/plan").json()

    jid = client.post(f"/api/projects/{project}/export",
                      json={"preset": "source"}).json()["job"]
    job = _wait(client, jid)
    assert job["status"] == "done", job["message"]

    from paperedit import store
    out = store.project_dir(project) / job["result"]
    assert out.exists()

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,duration", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True).stdout.strip().splitlines()
    kinds = {line.split(",")[0]: float(line.split(",")[1]) for line in probe}
    assert "video" in kinds and "audio" in kinds
    # Audio and video must stay locked together; total length runs a hair long
    # because ffmpeg's trim keeps the partial frame at each cut (see FINDINGS.md).
    assert abs(kinds["video"] - kinds["audio"]) < 0.05
    assert kinds["video"] == pytest.approx(plan["duration"], rel=0.01)

    assert client.get(f"/api/projects/{project}/file/{job['result']}").status_code == 200

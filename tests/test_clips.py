"""Clip suggestions, tested without a model in the way.

The model only proposes times; repair_clips() decides what is actually played
and cut, so that is where the tests are. Every transcript here is invented:
numbered sentences of filler prose with known timings, so the right answer is
computable. The endpoint tests swap the Ollama call for a fake.
"""
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paperedit import clips  # noqa: E402
from paperedit.clips import parse_reply, parse_time, repair_clips  # noqa: E402

WORD, GAP, BREATH = 0.3, 0.1, 0.6        # word length, gap between words, after a sentence
VOCAB = ["the", "small", "boat", "drifted", "past", "seven", "quiet", "harbour",
         "lights", "while", "gulls", "argued", "overhead", "about", "bread"]


def talk(sentences=40, per=12, deleted=()):
    """A synthetic transcript: `sentences` sentences of `per` words each.

    Sentence k starts at k * (per*0.4 - 0.1 + 0.6) = 5.3 s for per=12, and every
    sentence ends in a full stop followed by a 0.6 s breath.
    """
    words, t, i = [], 0.0, 0
    for s in range(sentences):
        for k in range(per):
            text = VOCAB[(s + k) % len(VOCAB)]
            if k == 0:
                text = text.capitalize()
            if k == per - 1:
                text += "."
            words.append({"idx": i, "text": " " + text, "start": round(t, 3),
                          "end": round(t + WORD, 3), "deleted": int(i in deleted),
                          "speaker": None, "probability": 1.0})
            t += WORD + (BREATH if k == per - 1 else GAP)
            i += 1
    return words


def sentence_start(words, s, per=12):
    return words[s * per]["start"]


def check(words, clip):
    """The invariants every returned clip must hold, whatever went in."""
    first, last = words[clip["first_idx"]], words[clip["last_idx"]]
    assert not first["deleted"] and not last["deleted"]
    assert clip["start"] <= first["start"] and clip["start"] > first["start"] - 0.2
    assert clip["end"] >= last["end"] and clip["end"] <= last["end"] + 0.45
    assert 30 <= clip["duration"] <= 60


# --------------------------------------------------------------------------- parsing

@pytest.mark.parametrize("raw, secs", [
    (192, 192.0), (192.5, 192.5), ("192", 192.0), ("3:12", 192.0), ("03:12", 192.0),
    ("[03:12]", 192.0), ("0:03:12", 192.0), ("03:12.5", 192.5),
    ("03:12-03:45", 192.0), ("", None), ("soon", None), (None, None), (-4, None),
    (True, None),
])
def test_times_are_read_in_every_shape_models_write(raw, secs):
    assert parse_time(raw) == secs


def test_garbage_replies_become_no_clips_not_a_crash():
    assert parse_reply("") == []
    assert parse_reply("I could not find anything good.") == []
    assert parse_reply('{"clips": [{"start": "00:10"') == []        # cut off mid-reply
    assert parse_reply('{"clips": "none"}') == []
    assert parse_reply('```json\n{"clips": [{"start": "1:00"}]}\n```') == [{"start": "1:00"}]
    assert parse_reply('[{"start": 5}, "nonsense", 7]') == [{"start": 5}]
    assert repair_clips(talk(), "not json at all") == {"clips": [], "dropped": []}


# --------------------------------------------------------------------------- repair

def test_sloppy_times_snap_to_sentence_starts_and_ends():
    words = talk()
    s = sentence_start(words, 6) + 1.7               # mid-word, mid-sentence
    e = sentence_start(words, 14) - 2.2
    out = repair_clips(words, [{"start": s, "end": e, "title": "A", "reason": "r"}])
    [c] = out["clips"]
    check(words, c)
    assert c["start_boundary"] == "sentence" and c["end_boundary"] == "sentence"
    assert c["first_idx"] % 12 == 0 and c["last_idx"] % 12 == 11
    # It keeps the model's opening line rather than skipping past it.
    assert c["first_idx"] == 6 * 12
    assert "start_snapped" in c["repairs"]


def test_a_clean_answer_is_left_alone():
    words = talk()
    s, e = sentence_start(words, 4), words[11 * 12 - 1]["end"]        # sentences 4-10
    [c] = repair_clips(words, [{"start": s, "end": e, "title": "A"}])["clips"]
    check(words, c)
    assert (c["first_idx"], c["last_idx"]) == (48, 131)
    assert c["repairs"] == []


def test_too_short_is_extended_and_too_long_is_trimmed():
    words = talk()
    short = {"start": sentence_start(words, 2), "end": sentence_start(words, 3)}
    long = {"start": sentence_start(words, 20), "end": sentence_start(words, 39)}
    out = repair_clips(words, [short, long])
    a, b = out["clips"]
    check(words, a), check(words, b)
    assert "extended" in a["repairs"] and "trimmed" in b["repairs"]
    assert a["end_boundary"] == b["end_boundary"] == "sentence"


def test_overlapping_suggestions_keep_the_first():
    words = talk()
    one = {"start": sentence_start(words, 5), "end": sentence_start(words, 12), "title": "one"}
    same = {"start": sentence_start(words, 6), "end": sentence_start(words, 13), "title": "two"}
    out = repair_clips(words, [one, same])
    assert [c["title"] for c in out["clips"]] == ["one"]
    assert out["dropped"] == [{"title": "two", "why": "dropped_overlap"}]


def test_a_partial_overlap_is_moved_off_rather_than_lost():
    words = talk()
    one = {"start": sentence_start(words, 5), "end": sentence_start(words, 12)}
    # Starts inside "one" but mostly after it: fits once moved past its end.
    two = {"start": sentence_start(words, 11), "end": sentence_start(words, 19)}
    a, b = repair_clips(words, [one, two])["clips"]
    assert b["start"] >= a["end"]
    assert "moved_off_overlap" in b["repairs"]


def test_clips_in_deleted_text_are_dropped():
    gone = range(10 * 12, 20 * 12)                    # sentences 10-19 deleted
    words = talk(deleted=gone)
    bad = {"start": sentence_start(words, 11), "end": sentence_start(words, 18), "title": "x"}
    out = repair_clips(words, [bad])
    assert out["clips"] == []
    assert out["dropped"][0]["why"] == "dropped_deleted"


def test_a_clip_never_starts_or_ends_on_a_deleted_word():
    # Delete the sentence the model starts in and the one it ends in.
    gone = list(range(5 * 12, 6 * 12)) + list(range(12 * 12, 13 * 12))
    words = talk(deleted=gone)
    s, e = sentence_start(words, 5) + 2, sentence_start(words, 13) - 1
    [c] = repair_clips(words, [{"start": s, "end": e}])["clips"]
    check(words, c)


def test_duration_counts_only_kept_time():
    """Deleted words inside a clip are skipped on playback, so the clip is
    measured the way it will actually play."""
    gone = range(8 * 12, 9 * 12)                      # one sentence in the middle
    words = talk(deleted=gone)
    s, e = sentence_start(words, 4), words[13 * 12 - 1]["end"]
    [c] = repair_clips(words, [{"start": s, "end": e}])["clips"]
    check(words, c)
    assert c["end"] - c["start"] > c["duration"] + 4  # a 5.3 s sentence is not counted
    assert c["first_idx"] < 8 * 12 < c["last_idx"]


def test_times_past_the_end_are_dropped_or_pulled_back():
    words = talk()
    end = words[-1]["end"]
    out = repair_clips(words, [
        {"start": end + 30, "end": end + 70, "title": "gone"},
        {"start": end - 40, "end": end + 25, "title": "late"},
    ])
    assert out["dropped"] == [{"title": "gone", "why": "dropped_past_end"}]
    [c] = out["clips"]
    check(words, c)
    assert c["last_idx"] == len(words) - 1 and "end_past_end" in c["repairs"]


def test_swapped_and_missing_fields_are_survivable():
    words = talk()
    out = repair_clips(words, [
        {"start": "1:30", "end": "1:00"},                    # backwards
        {"start": "2:10"},                                   # no end, no title
        {"end": "1:00", "title": "no start"},                # unusable
    ])
    a, b = out["clips"]
    check(words, a), check(words, b)
    assert "swapped" in a["repairs"] and "end_missing" in b["repairs"]
    assert b["title"] == "Untitled moment"
    assert out["dropped"] == [{"title": "no start", "why": "dropped_bad_time"}]


def test_a_talk_shorter_than_a_clip_gives_nothing():
    words = talk(sentences=4)                                # ~21 s
    out = repair_clips(words, [{"start": 0, "end": 20}])
    assert out["clips"] == [] and out["dropped"][0]["why"] == "dropped_too_short"


def test_only_n_clips_come_back():
    words = talk()
    many = [{"start": sentence_start(words, s), "end": sentence_start(words, s + 7)}
            for s in (0, 8, 16, 24, 32)]
    assert len(repair_clips(words, many, n=3)["clips"]) == 3


def test_a_run_on_with_no_full_stops_falls_back_to_commas_then_words():
    """Whisper often writes two minutes without a full stop. The clip must still
    fit, and say honestly what it landed on."""
    words = talk(sentences=30)
    for w in words:
        w["text"] = w["text"].replace(".", "")
    for w in words[::9]:
        w["text"] += ","
    [c] = repair_clips(words, [{"start": 20, "end": 65}])["clips"]
    assert 30 <= c["duration"] <= 60
    assert c["start_boundary"] in ("clause", "pause") and c["end_boundary"] in ("clause", "pause")


# --------------------------------------------------------------------------- prompt

def test_prompt_lines_carry_times_from_kept_words_only():
    words = talk(sentences=6, deleted=range(12, 24))
    segs = clips.segments(words)
    assert len(segs) == 5
    assert all(t != words[12]["start"] for t, _ in segs)
    prompt = clips.build_prompt(segs, n=3)
    assert "[00:00]" in prompt and "30 and 60 seconds" in prompt


def test_a_long_run_on_still_gets_regular_timestamps():
    words = talk(sentences=10, per=60)
    for w in words:
        w["text"] = w["text"].replace(".", "")
    segs = clips.segments(words)
    assert max(len(s.split()) for _, s in segs) <= clips.SEGMENT_WORDS


def test_an_hour_long_talk_is_asked_about_in_overlapping_parts():
    words = talk(sentences=700)                              # 8,400 words, ~62 min
    segs = clips.segments(words)
    parts = clips.chunk_segments(segs)
    assert len(parts) > 1
    assert all(clips.estimate_tokens(clips._lines(p)) <= clips.PROMPT_TOKEN_BUDGET
               for p in parts)
    # Every line is covered, in order, and each seam overlaps.
    seen = sorted({t for p in parts for t, _ in p})
    assert seen == [t for t, _ in segs]
    for a, b in zip(parts, parts[1:]):
        assert b[0][0] < a[-1][0]


def test_find_clips_merges_parts_strongest_first():
    words = talk(sentences=700)
    calls = []

    def fake(prompt, model=None, url=None):
        calls.append(prompt)
        first = prompt.split("Transcript:\n[", 1)[1].split("]")[0]
        return json.dumps({"clips": [{"start": first, "end": "", "title": f"p{len(calls)}",
                                      "reason": "r", "score": len(calls)}]}), {}

    out = clips.find_clips(words, n=2, ask=fake)
    assert len(calls) == out["parts"] > 1
    assert [c["title"] for c in out["clips"]] == [f"p{len(calls)}", f"p{len(calls) - 1}"]
    assert "parts" in out["note"]


def test_ollama_not_running_is_a_plain_error(monkeypatch):
    with pytest.raises(clips.ClipsError, match="not running"):
        clips.ask_model("hi", url="http://127.0.0.1:9", timeout=2)


# --------------------------------------------------------------------------- endpoint

@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from paperedit import store
    store.init()
    import server
    from conftest import sign_in
    with TestClient(server.app) as c:
        yield sign_in(c)


@pytest.fixture()
def project(client):
    from paperedit import store
    pid = client.post("/api/projects", json={"name": "clips"}).json()["id"]
    words = talk()
    store.set_words(pid, words)
    store.update_project(pid, duration=words[-1]["end"] + 1, status="ready")
    yield pid
    client.delete(f"/api/projects/{pid}")


def _wait(client, jid, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "failed", "interrupted"):
            return j
        time.sleep(0.05)
    raise AssertionError("job timed out")


def test_find_clips_endpoint_runs_as_a_job(client, project, monkeypatch):
    words = talk()
    reply = {"clips": [
        {"start": "00:31", "end": "01:02", "title": "First", "reason": "r1", "score": 8},
        {"start": "01:50", "end": "02:05", "title": "Second", "reason": "r2", "score": 6},
        {"start": "99:00", "end": "99:40", "title": "Nowhere", "reason": "r3", "score": 5},
    ]}
    monkeypatch.setattr(clips, "ask_model",
                        lambda prompt, model=None, url=None: (json.dumps(reply), {"eval_count": 1}))

    assert client.get(f"/api/projects/{project}/clips").json()["job"] is None
    jid = client.post(f"/api/projects/{project}/clips", json={}).json()["job"]
    job = _wait(client, jid)
    assert job["status"] == "done", job["message"]
    assert job["message"].startswith("2 clips found")

    r = client.get(f"/api/projects/{project}/clips").json()
    assert [c["title"] for c in r["clips"]] == ["First", "Second"]
    for c in r["clips"]:
        check(words, c)
    assert r["dropped"] == [{"title": "Nowhere", "why": "dropped_past_end"}]


def test_ollama_down_fails_the_job_with_a_readable_message(client, project, monkeypatch):
    def down(prompt, model=None, url=None):
        raise clips.ClipsError("Ollama is not running at http://127.0.0.1:11434")
    monkeypatch.setattr(clips, "ask_model", down)
    jid = client.post(f"/api/projects/{project}/clips").json()["job"]
    job = _wait(client, jid)
    assert job["status"] == "failed" and "Ollama is not running" in job["message"]
    assert client.get(f"/api/projects/{project}/clips").json()["clips"] == []


def test_a_garbage_reply_is_a_finished_job_with_no_clips(client, project, monkeypatch):
    monkeypatch.setattr(clips, "ask_model",
                        lambda prompt, model=None, url=None: ("Sure! Here are some clips:", {}))
    job = _wait(client, client.post(f"/api/projects/{project}/clips").json()["job"])
    assert job["status"] == "done"
    assert "no usable suggestions" in job["message"]


def test_clips_need_a_finished_transcript(client):
    pid = client.post("/api/projects", json={"name": "empty"}).json()["id"]
    try:
        r = client.post(f"/api/projects/{pid}/clips")
        assert r.status_code == 400
    finally:
        client.delete(f"/api/projects/{pid}")


def test_clips_are_behind_sign_in():
    from fastapi.testclient import TestClient
    import server
    with TestClient(server.app) as anon:
        assert anon.post("/api/projects/whatever/clips").status_code == 401
        assert anon.get("/api/projects/whatever/clips").status_code == 401

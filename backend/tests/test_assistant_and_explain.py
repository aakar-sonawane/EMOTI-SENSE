"""Tests for the emotion-aware assistant (against a fake Ollama server) and the audio+video explanation."""
import io, json, os, subprocess, tempfile, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import cv2
from fastapi.testclient import TestClient
from skimage import data as skdata

import assistant
import engine
import server

client = TestClient(server.app, raise_server_exceptions=False)


# ---------------------------------------------------------------- fake Ollama
class FakeOllama:
    def __init__(self, port=0):
        self.requests, self.generate_requests, self.models, self.mode = [], [], ["llama3:8b", "qwen3:8b"], "ok"
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                body = json.dumps({"models": [{"name": m} for m in outer.models]}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if self.path == "/api/generate":
                    outer.generate_requests.append(req)
                    self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                    self.wfile.write(json.dumps({"model": req["model"], "done": True}).encode()); return
                outer.requests.append(req)
                if outer.mode == "missing_model":
                    self.send_response(404); self.send_header("Content-Type", "application/json"); self.end_headers()
                    self.wfile.write(json.dumps({"error": f"model '{req['model']}' not found"}).encode()); return
                last = req["messages"][-1]["content"]
                reply = "<think>secret reasoning</think>\n\nI hear you. You said: " + last[:30]
                self.send_response(200); self.send_header("Content-Type", "application/x-ndjson"); self.end_headers()
                for i in range(0, len(reply), 6):
                    self.wfile.write((json.dumps({"message": {"role": "assistant", "content": reply[i:i + 6]}, "done": False}) + "\n").encode()); self.wfile.flush()
                self.wfile.write((json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}) + "\n").encode())

        self.server = HTTPServer(("127.0.0.1", port), H)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


def fake_text(texts, batch_size=16):
    out = []
    for t in texts:
        w = t.lower().split()
        pos, neg = sum(x in ("love", "great", "wonderful") for x in w), sum(x in ("hate", "terrible", "sad", "die") for x in w)
        z = np.array([neg * 1.0, 0.5, pos * 1.0]) + 0.05
        out.append(z / z.sum())
    return np.stack(out)


@pytest.fixture
def ollama(monkeypatch):
    fake = FakeOllama()
    monkeypatch.setenv("OLLAMA_HOST", f"127.0.0.1:{fake.port}")
    monkeypatch.setattr(engine, "predict_text_batch", fake_text)
    yield fake
    fake.close()


def chat(messages=None, **kw):
    body = {"model": "qwen3:8b", "messages": messages or [{"role": "user", "content": "I love this wonderful project"}], **kw}
    r = client.post("/api/assistant/chat", json=body)
    events = [json.loads(line) for line in r.text.splitlines() if line.strip()] if r.status_code == 200 else []
    return r, events


# ---------------------------------------------------------------- status
def test_status_lists_models_and_prefers_qwen3(ollama):
    j = client.get("/api/assistant/status").json()
    assert j["available"] and j["models"] == ["llama3:8b", "qwen3:8b"] and j["preferred"] == "qwen3:8b" and j["error"] is None


def test_status_when_ollama_is_not_running(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:9")
    j = client.get("/api/assistant/status").json()
    assert j["available"] is False and "ollama serve" in j["error"] and j["models"] == []


def test_status_with_no_models_installed(ollama):
    ollama.models = []
    j = client.get("/api/assistant/status").json()
    assert j["available"] is False and "ollama pull" in j["error"]


def test_preferred_model_order():
    assert assistant.preferred_model(["llama3", "Qwen2.5:3b", "qwen3:8b"]) == "qwen3:8b"
    assert assistant.preferred_model(["llama3", "qwen2.5:3b"]) == "qwen2.5:3b"
    assert assistant.preferred_model(["llama3"]) == "llama3" and assistant.preferred_model([]) is None


# ---------------------------------------------------------------- chat
def test_chat_streams_sentiment_then_text_then_done_and_hides_thinking(ollama):
    r, ev = chat()
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    types = [e["type"] for e in ev]
    assert types[0] == "sentiment" and types[-1] == "done" and "crisis" not in types and "error" not in types
    assert ev[0]["label"] == "Positive" and ev[0]["valence"] > 0.5 and len(ev[0]["probs"]) == 3
    text = "".join(e["text"] for e in ev if e["type"] == "delta")
    assert text == "I hear you. You said: I love this wonderful project"[:len(text)] and text.startswith("I hear you.") and "secret" not in text and "<think>" not in text
    assert len([e for e in ev if e["type"] == "delta"]) > 3          # really streamed in pieces


def test_chat_sends_the_right_request_to_ollama(ollama):
    chat()
    req = ollama.requests[-1]
    assert req["model"] == "qwen3:8b" and req["stream"] is True and req["think"] is False
    assert req["messages"][0]["role"] == "system" and "not a therapist" in req["messages"][0]["content"]
    assert "estimated the user's latest message as Positive" in req["messages"][0]["content"]
    assert req["messages"][-1] == {"role": "user", "content": "I love this wonderful project"}
    assert req["options"]["temperature"] == 0.7


def test_think_flag_is_only_sent_to_qwen3(ollama):
    chat(model="llama3:8b")
    assert "think" not in ollama.requests[-1]


def test_style_changes_the_system_prompt(ollama):
    chat(style="Professional"); assert "professional" in ollama.requests[-1]["messages"][0]["content"].lower()
    chat(style="NotAStyle");   assert "warm and gentle" in ollama.requests[-1]["messages"][0]["content"]


def test_sentiment_can_be_switched_off(ollama):
    r, ev = chat(analyse=False)
    assert "sentiment" not in [e["type"] for e in ev]
    assert "classifier" not in ollama.requests[-1]["messages"][0]["content"]


def test_history_is_trimmed_and_order_kept(ollama):
    msgs = []
    for i in range(15):
        msgs += [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}]
    msgs.append({"role": "user", "content": "final question"})
    chat(messages=msgs)
    sent = ollama.requests[-1]["messages"]
    assert len(sent) == 13 and sent[0]["role"] == "system" and sent[-1]["content"] == "final question"


def test_crisis_message_gets_helpline_and_careful_prompt(ollama):
    r, ev = chat(messages=[{"role": "user", "content": "I want to die"}])
    crisis = [e for e in ev if e["type"] == "crisis"]
    assert crisis and "14416" in crisis[0]["note"] and "112" in crisis[0]["note"]
    assert ev.index(crisis[0]) < next(i for i, e in enumerate(ev) if e["type"] == "delta")      # shown before the reply
    assert "risk of self-harm" in ollama.requests[-1]["messages"][0]["content"]
    _, ev2 = chat(messages=[{"role": "user", "content": "I love pizza"}])
    assert "crisis" not in [e["type"] for e in ev2]


def test_face_emotion_context_is_used_only_when_valid(ollama):
    chat(face_emotion={"label": "Sad", "confidence": 0.7, "age_s": 12})
    assert "looked sad about 12 seconds ago" in ollama.requests[-1]["messages"][0]["content"]
    for bad in ({"label": "Confused", "confidence": 0.7, "age_s": 1}, {"label": "Sad", "confidence": 1.7, "age_s": 1}, {"label": "Sad", "confidence": 0.7, "age_s": 999}):
        chat(face_emotion=bad)
        assert "camera suggested" not in ollama.requests[-1]["messages"][0]["content"]


def test_chat_validation(ollama):
    assert client.post("/api/assistant/chat", json={"model": "m", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}).status_code == 422
    assert client.post("/api/assistant/chat", json={"model": "m", "messages": [{"role": "user", "content": "   "}]}).status_code == 400
    assert client.post("/api/assistant/chat", json={"model": "m", "messages": []}).status_code == 422
    assert client.post("/api/assistant/chat", json={"model": "m", "messages": [{"role": "user", "content": "a" * 4001}]}).status_code == 422
    assert client.post("/api/assistant/chat", json={"messages": [{"role": "user", "content": "x"}]}).status_code == 422
    r, ev = chat(temperature=9)
    assert ollama.requests[-1]["options"]["temperature"] == 1.2


def test_chat_survives_a_missing_sentiment_model(ollama, monkeypatch):
    def broken(text): raise engine.ModelError("Text checkpoint not found. Expected: /x")
    monkeypatch.setattr(engine, "analyse_text", broken)
    _, ev = chat()
    types = [e["type"] for e in ev]
    assert types[0] == "notice" and "Text checkpoint not found" in ev[0]["message"] and types[-1] == "done"


def test_chat_reports_ollama_down_as_an_event(monkeypatch):
    monkeypatch.setattr(engine, "predict_text_batch", fake_text)
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:9")
    r, ev = chat()
    assert r.status_code == 200 and ev[-1]["type"] == "error" and "ollama serve" in ev[-1]["message"]


def test_chat_reports_unknown_model_with_pull_hint(ollama):
    ollama.mode = "missing_model"
    _, ev = chat(model="qwen3:99b")
    assert ev[-1]["type"] == "error" and "ollama pull qwen3:99b" in ev[-1]["message"] and "404" in ev[-1]["message"]


def test_think_filter_edge_cases():
    f = lambda parts: "".join(assistant.filter_think_stream(parts))
    assert f(["Hel", "lo <thi", "nk>secret</thi", "nk> world"]) == "Hello world"
    assert f(["no tags"]) == "no tags" and f(["<think>all hidden"]) == "" and f(["a < b and <thin", "g"]) == "a < b and <thing"
    assert f(["<think>x</think>\n\nAnswer"]) == "Answer"


# ---------------------------------------------------------------- audio + video explanation
def make_clip(tmp, with_audio=True, seconds=4, freq=300):
    face = cv2.cvtColor(skdata.astronaut(), cv2.COLOR_RGB2BGR)
    h, w = face.shape[:2]
    silent = os.path.join(tmp, "silent.mp4")
    vw = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    blank = np.full_like(face, 120)
    for i in range(30 * seconds):
        vw.write(face if i < 30 * seconds * 0.75 else blank)     # the last quarter has no face
    vw.release()
    if not with_audio:
        return silent
    import imageio_ffmpeg
    out = os.path.join(tmp, "clip.mp4")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", silent, "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}",
                    "-shortest", "-c:v", "libx264", "-c:a", "aac", out], capture_output=True)
    return out


ALPHA, BETA = 0.07, 0.002
def linear_av(audio, video):
    """A model with known behaviour: score = ALPHA*sum(audio) + BETA*sum(video)."""
    return float(ALPHA * np.sum(audio) + BETA * np.sum(video)), [0.3, 0.7]


def test_windows_cover_the_clip():
    assert engine._windows(50, 10, 5) == [(s, s + 10) for s in range(0, 41, 5)]
    assert engine._windows(7, 10, 5) == [(0, 7)]
    assert engine._windows(12, 10, 5) == [(0, 10), (2, 12)]


def test_explain_matches_an_independent_calculation(monkeypatch):
    monkeypatch.setattr(engine, "av_forward", linear_av)
    with tempfile.TemporaryDirectory() as tmp:
        path = make_clip(tmp)
        r = client.post("/api/av/explain", files={"file": ("c.mp4", Path(path).read_bytes(), "video/mp4")})
        feats = engine.av_features(path, tmp)
    j = r.json()
    assert r.status_code == 200, j
    a, v = feats["audio"], feats["video"]
    raw = ALPHA * a.sum() + BETA * v.sum()
    ex = j["explain"]
    clip = lambda x: float(np.clip(x, -3, 3))
    assert j["raw"] == pytest.approx(raw, rel=1e-4) and ex["modalities"]["both"] == pytest.approx(clip(raw))
    assert ex["modalities"]["audio_only"] == pytest.approx(clip(ALPHA * a.sum()))          # video stream removed
    assert ex["modalities"]["video_only"] == pytest.approx(clip(BETA * v.sum()))           # audio stream removed
    for w in ex["audio_windows"]:                                                         # effect = score - score_without_window
        s0, s1 = round(w["start_s"] * 16000 / 512), round(w["end_s"] * 16000 / 512)
        assert w["effect"] == pytest.approx(ALPHA * a[s0:s1].sum(), rel=1e-3, abs=1e-3)
    t = feats["frame_times"]
    for w in ex["video_windows"]:
        s0 = t.index(next(x for x in t if abs(x - w["start_s"]) < 1e-9)); s1 = t.index(next(x for x in t if abs(x - w["end_s"]) < 1e-9)) + 1
        assert w["effect"] == pytest.approx(BETA * v[s0:s1].sum(), rel=1e-3, abs=1e-3)
    assert len(ex["audio_windows"]) >= 2 and len(ex["video_windows"]) >= 2
    assert all(w["end_s"] > w["start_s"] for w in ex["audio_windows"] + ex["video_windows"])


def test_explain_frames_and_voice(monkeypatch):
    monkeypatch.setattr(engine, "av_forward", linear_av)
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(make_clip(tmp)).read_bytes()
    ex = client.post("/api/av/explain", files={"file": ("c.mp4", data, "video/mp4")}).json()["explain"]
    frames = ex["frames"]
    assert 2 <= len(frames) <= 8 and [f["t"] for f in frames] == sorted(f["t"] for f in frames)
    assert frames[0]["face"] is True and frames[-1]["face"] is False and frames[-1]["thumb"] is None      # last quarter has no face
    import base64
    thumb = Image.open(io.BytesIO(base64.b64decode(frames[0]["thumb"].split(",", 1)[1])))
    assert thumb.size == (96, 96)
    st = ex["voice"]["stats"]
    assert 3.5 < st["Duration (s)"] < 4.5 and 270 < st["Median pitch (Hz)"] < 330           # the clip carries a 300 Hz tone
    assert len(ex["voice"]["t"]) == len(ex["voice"]["rms"]) == len(ex["voice"]["f0"]) <= 120


def test_explain_errors(monkeypatch):
    monkeypatch.setattr(engine, "av_forward", linear_av)
    with tempfile.TemporaryDirectory() as tmp:
        silent = Path(make_clip(tmp, with_audio=False)).read_bytes()
    r = client.post("/api/av/explain", files={"file": ("s.mp4", silent, "video/mp4")})
    assert r.status_code == 422 and "audio" in r.json()["detail"].lower()
    assert client.post("/api/av/explain", files={"file": ("s.mp4", b"", "video/mp4")}).status_code == 400

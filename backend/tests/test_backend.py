"""Backend tests. Real code paths: HTTP layer, Haar face detection, image/clip handling, librosa audio features,
ffmpeg, OpenCV video reading, validation and error handling.
Only the three neural-network forward passes are replaced by small stand-ins (torch/tensorflow are not needed to run these tests)."""
import base64, io, os, subprocess, sys, tempfile, threading, time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT_DIR = os.environ["EMOTISENSE_ROOT"]      # created by conftest.py

import cv2
from fastapi.testclient import TestClient
from skimage import data as skdata

import engine
import server

client = TestClient(server.app, raise_server_exceptions=False)


# ---------------- stand-ins for the neural networks ----------------
def fake_text(texts, batch_size=16):
    out = []
    for t in texts:
        w = t.lower().split()
        pos = sum(x in ("love", "great", "wonderful") for x in w)
        neg = sum(x in ("hate", "terrible", "bad") for x in w)
        z = np.array([neg * 1.0, 0.5, pos * 1.0]) + 0.05
        out.append(z / z.sum())
    return np.stack(out)


def fake_face(gray_batch):
    g = np.asarray(gray_batch, dtype=np.float32)
    assert g.ndim == 3 and g.shape[1:] == (48, 48), g.shape
    bright = 1 / (1 + np.exp(-(g.mean(axis=(1, 2)) - 110) / 25))
    out = np.zeros((g.shape[0], 7))
    out[:, 3], out[:, 4] = bright, 1 - bright           # Happy vs Neutral
    return out


AV_CALLS = []
def fake_av(audio, video):
    AV_CALLS.append((audio.shape, video.shape, float(video.sum())))
    return 1.4, [0.35, 0.65]


@pytest.fixture(autouse=True)
def patch_models(monkeypatch):
    monkeypatch.setattr(engine, "predict_text_batch", fake_text)
    monkeypatch.setattr(engine, "_predict_face_gray", fake_face)
    monkeypatch.setattr(engine, "av_forward", fake_av)
    engine._cache.pop("cascade", None)
    AV_CALLS.clear()


def face_image(size=None):
    img = Image.fromarray(skdata.astronaut())
    return img if size is None else img.resize(size)


def jpeg_bytes(img, quality=80):
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=quality); return buf.getvalue()


# ---------------- health ----------------
def test_health_reports_files_and_detector():
    r = client.get("/api/health").json()
    assert all(r["models"][k]["found"] for k in ("text", "face", "av"))
    assert r["face_detector"]["ok"] is True and "OpenCV" in r["face_detector"]["detail"]
    assert "packages" in r and isinstance(r["packages"]["cv2"], bool)


def test_health_flags_missing_checkpoint(monkeypatch):
    monkeypatch.setattr(engine, "TEXT_CHECKPOINT", Path(ROOT_DIR) / "nope.pt")
    assert client.get("/api/health").json()["models"]["text"]["found"] is False


# ---------------- text ----------------
def test_text_positive_and_negative():
    r = client.post("/api/text", json={"text": "I love this wonderful thing"}).json()
    assert r["label"] == "Positive" and abs(sum(r["probs"]) - 1) < 1e-6 and r["labels"] == engine.TEXT_LABELS and r["valence"] > 0.5
    assert client.post("/api/text", json={"text": "I hate this terrible thing"}).json()["label"] == "Negative"


def test_text_validation():
    assert client.post("/api/text", json={"text": "   "}).status_code == 400
    assert client.post("/api/text", json={"text": "a" * 5001}).status_code == 413
    assert client.post("/api/text", json={}).status_code == 422


def test_text_explain_marks_sentiment_words():
    r = client.post("/api/text/explain", json={"text": "I love this wonderful project"}).json()
    scores = dict(zip(r["words"], r["scores"]))
    assert r["label"] == "Positive" and scores["love"] > 0 and scores["wonderful"] > 0 and scores["project"] <= 1e-9
    long = client.post("/api/text/explain", json={"text": "word " * 60}).json()
    assert long["truncated"] is True and len(long["words"]) == 40


def test_text_model_missing_gives_503_with_path(monkeypatch):
    monkeypatch.undo()                                   # use the real predict_text_batch
    monkeypatch.setattr(engine, "TEXT_CHECKPOINT", Path(ROOT_DIR) / "missing.pt")
    engine._cache.pop("text", None)
    r = client.post("/api/text", json={"text": "hello"})
    assert r.status_code == 503 and "checkpoint not found" in r.json()["detail"] and "missing.pt" in r.json()["detail"]


# ---------------- face (photo) ----------------
def test_face_photo_crops_the_face_and_returns_thumbnail():
    r = client.post("/api/face", files={"file": ("a.jpg", jpeg_bytes(face_image()), "image/jpeg")}, data={"crop": "true"})
    j = r.json()
    assert r.status_code == 200 and j["label"] in engine.FACE_LABELS and len(j["probs"]) == 7 and j["note"] is None
    thumb = Image.open(io.BytesIO(base64.b64decode(j["face"].split(",", 1)[1])))
    assert max(thumb.size) <= 224 and thumb.size[0] < face_image().size[0]          # it really is a crop


def test_face_photo_without_crop_and_no_face_note():
    blank = Image.new("RGB", (200, 200), (20, 20, 20))
    j = client.post("/api/face", files={"file": ("b.png", jpeg_bytes(blank), "image/png")}, data={"crop": "true"}).json()
    assert "No face was detected" in j["note"]
    j2 = client.post("/api/face", files={"file": ("a.jpg", jpeg_bytes(face_image()), "image/jpeg")}, data={"crop": "false"}).json()
    assert j2["note"] is None


def test_face_photo_uses_the_original_pil_preprocessing():
    seen = {}
    def spy(gray_batch):
        seen["g"] = np.asarray(gray_batch); return fake_face(gray_batch)
    engine._predict_face_gray = spy
    img = face_image((120, 120))
    client.post("/api/face", files={"file": ("a.jpg", jpeg_bytes(img, 95), "image/jpeg")}, data={"crop": "false"})
    expected = np.asarray(Image.open(io.BytesIO(jpeg_bytes(img, 95))).convert("L").resize((48, 48)), dtype=np.float32)
    assert seen["g"].shape == (1, 48, 48) and np.allclose(seen["g"][0], expected)


def test_face_photo_bad_input():
    assert client.post("/api/face", files={"file": ("x.txt", b"not an image", "text/plain")}).status_code == 415
    assert client.post("/api/face", files={"file": ("x.jpg", b"", "image/jpeg")}).status_code == 400


def test_heatmap_overlay_is_an_image():
    r = client.post("/api/face/heatmap", files={"file": ("a.jpg", jpeg_bytes(face_image()), "image/jpeg")})
    j = r.json()
    img = Image.open(io.BytesIO(base64.b64decode(j["overlay"].split(",", 1)[1])))
    assert r.status_code == 200 and img.size == (336, 336) and j["label"] in engine.FACE_LABELS


# ---------------- face (live frames) ----------------
def test_live_frame_finds_face_with_normalised_box():
    frame = face_image().resize((320, 320))
    j = client.post("/api/face/live", content=jpeg_bytes(frame, 70), headers={"Content-Type": "image/jpeg"}).json()
    assert j["width"] == 320 and len(j["faces"]) >= 1
    x, y, w, h = j["faces"][0]["box"]
    assert 0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1 and x + w <= 1.001 and y + h <= 1.001
    assert set(j["ms"]) == {"detect", "predict", "total"} and j["faces"][0]["label"] in engine.FACE_LABELS


def test_live_frame_without_face_returns_empty_list():
    j = client.post("/api/face/live", content=jpeg_bytes(Image.new("RGB", (320, 240), (90, 90, 90))), headers={"Content-Type": "image/jpeg"}).json()
    assert j["faces"] == [] and j["ms"]["predict"] == 0


def test_live_frame_validation():
    assert client.post("/api/face/live", content=b"").status_code == 400
    assert client.post("/api/face/live", content=b"garbage", headers={"Content-Type": "image/jpeg"}).status_code == 422
    assert client.post("/api/face/live", content=b"0" * (3 * 1024 * 1024)).status_code == 413


def test_live_frame_multiple_faces_sorted_by_size():
    canvas = Image.new("RGB", (640, 360), (200, 200, 200))
    big = face_image().resize((300, 300)); small = face_image().resize((150, 150))
    canvas.paste(big, (20, 30)); canvas.paste(small, (400, 100))
    j = client.post("/api/face/live", content=jpeg_bytes(canvas, 90), headers={"Content-Type": "image/jpeg"}).json()
    sizes = [f["box"][2] * f["box"][3] for f in j["faces"]]
    assert len(j["faces"]) >= 2 and sizes == sorted(sizes, reverse=True)


def test_live_parallel_requests_do_not_deadlock():
    def slow_face(g):
        time.sleep(0.05); return fake_face(g)
    engine._predict_face_gray = slow_face
    frame = jpeg_bytes(face_image().resize((320, 320)), 70)
    results = []
    def worker():
        results.append(client.post("/api/face/live", content=frame, headers={"Content-Type": "image/jpeg"}).status_code)
    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]; [t.join(timeout=30) for t in threads]
    assert results == [200] * 8


def test_opencv5_gives_clear_503(monkeypatch):
    saved = cv2.CascadeClassifier
    monkeypatch.delattr(cv2, "CascadeClassifier")
    try:
        r = client.post("/api/face/live", content=jpeg_bytes(face_image().resize((320, 320))), headers={"Content-Type": "image/jpeg"})
        assert r.status_code == 503 and "removed it" in r.json()["detail"] and "opencv-python-headless<5" in r.json()["detail"]
        # photos still work (without cropping) and say why
        j = client.post("/api/face", files={"file": ("a.jpg", jpeg_bytes(face_image()), "image/jpeg")}, data={"crop": "true"}).json()
        assert "Face detection was skipped" in j["note"]
        assert client.get("/api/health").json()["face_detector"]["ok"] is False
    finally:
        cv2.CascadeClassifier = saved


# ---------------- audio + video clip ----------------
def make_clip(tmp, with_audio=True, seconds=3):
    face = cv2.cvtColor(skdata.astronaut(), cv2.COLOR_RGB2BGR)
    h, w = face.shape[:2]
    silent = os.path.join(tmp, "silent.mp4")
    vw = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for _ in range(30 * seconds):
        vw.write(face)
    vw.release()
    if not with_audio:
        return silent
    import imageio_ffmpeg
    out = os.path.join(tmp, "clip.mp4")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", silent, "-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}",
                    "-shortest", "-c:v", "libx264", "-c:a", "aac", out], capture_output=True)
    return out


def test_clip_pipeline_shapes_and_result():
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(make_clip(tmp)).read_bytes()
    r = client.post("/api/av", files={"file": ("clip.mp4", data, "video/mp4")})
    j = r.json()
    assert r.status_code == 200, j
    assert AV_CALLS and AV_CALLS[0][0] == (50, 28) and AV_CALLS[0][1] == (50, 2304)
    assert j["label"] == "Positive" and abs(j["score"] - 1.4) < 1e-9 and abs(j["gate_audio"] + j["gate_video"] - 1) < 1e-9
    assert j["face_rate"] > 0.8 and 2.5 < j["video_total_s"] < 3.5 and abs(j["audio_used_s"] - 1.6) < 0.05
    assert AV_CALLS[0][2] > 0                      # real face pixels reached the model


def test_clip_without_audio_is_a_clear_422():
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(make_clip(tmp, with_audio=False)).read_bytes()
    r = client.post("/api/av", files={"file": ("silent.mp4", data, "video/mp4")})
    assert r.status_code == 422 and "audio" in r.json()["detail"].lower()


def test_clip_score_is_clipped_and_empty_upload_rejected(monkeypatch):
    monkeypatch.setattr(engine, "av_forward", lambda a, v: (7.2, [0.5, 0.5]))
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(make_clip(tmp)).read_bytes()
    j = client.post("/api/av", files={"file": ("c.mp4", data, "video/mp4")}).json()
    assert j["score"] == 3.0 and j["raw"] == 7.2 and j["valence"] == 1.0
    assert client.post("/api/av", files={"file": ("c.mp4", b"", "video/mp4")}).status_code == 400


# ---------------- self-test and static ----------------
def test_selftest_reports_each_check(monkeypatch):
    monkeypatch.setattr(engine, "_load_av", lambda: object())
    res = client.get("/api/selftest").json()["results"]
    names = {r["name"]: r for r in res}
    assert names["Text model: clearly positive sentence"]["ok"] and names["Text model: clearly negative sentence"]["ok"]
    assert names["Face model: output shape"]["ok"] and names["Face detector (Haar cascade)"]["ok"] and names["Audio + video model: loads"]["ok"]


def test_selftest_surfaces_failures(monkeypatch):
    monkeypatch.setattr(engine, "predict_text_batch", lambda texts, batch_size=16: np.array([[0.1, 0.8, 0.1]] * len(texts)))
    res = {r["name"]: r for r in client.get("/api/selftest").json()["results"]}
    assert res["Text model: clearly positive sentence"]["ok"] is False and "Neutral" in res["Text model: clearly positive sentence"]["detail"]


def test_index_and_assets_served(tmp_path):
    static = tmp_path / "static"; (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<html>hello</html>"); (static / "assets" / "a.js").write_text("console.log(1)")
    server.STATIC = static
    server.app.router.routes = [r for r in server.app.router.routes if getattr(r, "path", "") != "/assets"]
    from fastapi.staticfiles import StaticFiles
    server.app.mount("/assets", StaticFiles(directory=static / "assets"), name="assets")
    assert client.get("/").text == "<html>hello</html>" and client.get("/assets/a.js").status_code == 200
    server.STATIC = tmp_path / "empty"
    assert client.get("/").status_code == 404

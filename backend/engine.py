"""
EmotiSense engine: model loading and inference. No web code in here.

The three models are used exactly as in your notebooks / Streamlit app:
  * Text  : BERT + RoBERTa cross-attention fusion (PyTorch)   -> Negative / Neutral / Positive
  * Face  : VGG16 + ResNet50 + InceptionV3 fusion (Keras)     -> 7 FER2013 emotions
  * AV    : audio + video adaptive fusion (PyTorch, CMU-MOSI) -> sentiment score -3 .. +3

Heavy libraries (torch, tensorflow, librosa) are imported lazily, so the server starts fast and
/api/health works even if one of them is missing.
"""
from __future__ import annotations

import base64
import importlib.util
import io
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Project root = the folder that contains TEXT/, FACE/ and "AUDIO AND VIDEO/" (the parent of backend/).
ROOT = Path(os.environ.get("EMOTISENSE_ROOT", Path(__file__).resolve().parent.parent))
TEXT_CHECKPOINT = ROOT / "TEXT" / "best_model_finetuned.pt"
FACE_CHECKPOINT = ROOT / "FACE" / "EVALUATED_MODEL" / "robust_v2_epoch4.keras"
AV_CHECKPOINT = ROOT / "AUDIO AND VIDEO" / "best_model.pt"

TEXT_LABELS = ["Negative", "Neutral", "Positive"]
FACE_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
FACE_POSITIVE = ("Happy", "Surprise")
FACE_NEGATIVE = ("Angry", "Disgust", "Fear", "Sad")

AV_SEQ_LEN = 50  # same as training: 50 audio frames, 50 sampled video frames
AV_AUDIO_SR, AV_HOP = 16000, 512
AV_VIDEO_STEP, AV_CROP = 3, 48


class ModelError(RuntimeError):
    """A model or dependency is not available. The message is safe to show to the user."""


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
_cache: dict = {}
_cache_lock = threading.Lock()
_text_lock, _face_lock, _av_lock = threading.Lock(), threading.Lock(), threading.Lock()
_cascade_lock = threading.Lock()  # a cv2 cascade object is not thread-safe


def _cached(key, factory):
    with _cache_lock:
        if key not in _cache:
            _cache[key] = factory()
        return _cache[key]


def _need_file(path: Path, what: str):
    if not path.is_file():
        raise ModelError(f"{what} checkpoint not found. Expected: {path}")


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def face_valence(probs) -> float:
    """7 face classes -> valence = P(Happy, Surprise) - P(Angry, Disgust, Fear, Sad). Neutral counts as 0."""
    p = np.asarray(probs, dtype=float)
    pos = sum(p[i] for i, lab in enumerate(FACE_LABELS) if lab in FACE_POSITIVE)
    neg = sum(p[i] for i, lab in enumerate(FACE_LABELS) if lab in FACE_NEGATIVE)
    return float(pos - neg)


def text_valence(probs) -> float:
    p = np.asarray(probs, dtype=float)
    return float(p[2] - p[0])


def av_valence(score: float) -> float:
    return float(np.clip(score / 3.0, -1.0, 1.0))


def av_polarity(score: float) -> str:
    if score <= -0.5:
        return "Negative"
    if score >= 0.5:
        return "Positive"
    return "Neutral"


def _png_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _jpeg_data_url(img: Image.Image, max_side: int = 224) -> str:
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# TEXT model (copied from the Streamlit app / training code)
# ---------------------------------------------------------------------------
def _load_text():
    _need_file(TEXT_CHECKPOINT, "Text")
    if not (_importable("torch") and _importable("transformers")):
        raise ModelError("The text model needs the 'torch' and 'transformers' packages.")
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_name, roberta_name = "bert-base-uncased", "roberta-base"

    class CrossAttentionFusionModel(nn.Module):
        def __init__(self, num_classes=3):
            super().__init__()
            self.bert = AutoModel.from_pretrained(bert_name)
            self.roberta = AutoModel.from_pretrained(roberta_name)
            hidden = self.bert.config.hidden_size
            self.cross_attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=8, batch_first=True)
            self.norm = nn.LayerNorm(hidden)
            self.dropout = nn.Dropout(0.3)
            self.classifier = nn.Linear(hidden, num_classes)

        def forward(self, bert_ids, bert_mask, roberta_ids, roberta_mask):
            bert_hidden = self.bert(input_ids=bert_ids, attention_mask=bert_mask).last_hidden_state
            roberta_out = self.roberta(input_ids=roberta_ids, attention_mask=roberta_mask).last_hidden_state
            roberta_cls = roberta_out[:, 0:1, :]
            attn_out, _ = self.cross_attn(query=roberta_cls, key=bert_hidden, value=bert_hidden,
                                          key_padding_mask=(bert_mask == 0))
            fused = self.norm(attn_out.squeeze(1) + roberta_out[:, 0, :])
            return self.classifier(self.dropout(fused))

    bert_tok = AutoTokenizer.from_pretrained(bert_name)
    roberta_tok = AutoTokenizer.from_pretrained(roberta_name)
    model = CrossAttentionFusionModel().to(device)
    checkpoint = torch.load(str(TEXT_CHECKPOINT), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, bert_tok, roberta_tok, device


def predict_text_batch(texts, batch_size: int = 16) -> np.ndarray:
    """(n_texts, 3) softmax probabilities, tokenised exactly like the training/inference code (max_length 256)."""
    model, bert_tok, roberta_tok, device = _cached("text", _load_text)  # raises a friendly ModelError first
    import torch
    out = []
    with _text_lock:
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i:i + batch_size])
            bert = bert_tok(chunk, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
            roberta = roberta_tok(chunk, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
            with torch.no_grad():
                logits = model(bert["input_ids"].to(device), bert["attention_mask"].to(device),
                               roberta["input_ids"].to(device), roberta["attention_mask"].to(device))
                out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(out)


def analyse_text(text: str) -> dict:
    t0 = time.perf_counter()
    probs = predict_text_batch([text])[0]
    ms = (time.perf_counter() - t0) * 1000
    i = int(np.argmax(probs))
    return {"label": TEXT_LABELS[i], "confidence": float(probs[i]), "labels": TEXT_LABELS,
            "probs": [float(p) for p in probs], "valence": text_valence(probs), "ms": ms}


def explain_text_words(text: str, max_words: int = 40) -> dict:
    """Occlusion importance: drop in the predicted class probability when a word is removed."""
    all_words = text.split()
    words = all_words[:max_words]
    variants = [" ".join(words)] + [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    probs = predict_text_batch(variants)
    base = probs[0]
    cls = int(np.argmax(base))
    return {"words": words, "scores": [float(x) for x in (base[cls] - probs[1:, cls])], "label": TEXT_LABELS[cls],
            "confidence": float(base[cls]), "truncated": len(all_words) > max_words}


# ---------------------------------------------------------------------------
# FACE model (copied from the Streamlit app)
# ---------------------------------------------------------------------------
def _load_face():
    _need_file(FACE_CHECKPOINT, "Face")
    if not _importable("tensorflow"):
        raise ModelError("The face model needs the 'tensorflow' package.")
    import tensorflow as tf
    return tf.keras.models.load_model(str(FACE_CHECKPOINT))


def face_inputs_from_gray(gray_batch):
    """(N,48,48) gray 0-255 -> the three model inputs: 48x48 -> 3 channels -> 224 bicubic -> VGG/ResNet/Inception preprocessing."""
    import tensorflow as tf
    from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
    from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
    rgb = np.repeat(np.asarray(gray_batch, dtype=np.float32)[..., None], 3, axis=-1)
    resized = tf.image.resize(rgb, (224, 224), method="bicubic").numpy()
    return [pre(resized.copy()) for pre in (vgg_preprocess, resnet_preprocess, inception_preprocess)]


def _predict_face_gray(gray_batch) -> np.ndarray:
    """(N,48,48) float gray -> (N,7) probabilities. The only function that touches TensorFlow."""
    model = _cached("face", _load_face)
    with _face_lock:
        return np.asarray(model.predict(face_inputs_from_gray(gray_batch), verbose=0, batch_size=16))


def haar_face_cascade():
    """Haar-cascade face detector. OpenCV 5 removed CascadeClassifier, so explain the fix instead of crashing."""
    def make():
        import cv2
        if not hasattr(cv2, "CascadeClassifier"):
            raise ModelError(f"Your OpenCV ({cv2.__version__}) has no CascadeClassifier because OpenCV 5 removed it. "
                             'Fix: python -m pip uninstall -y opencv-python opencv-python-headless  then  '
                             'python -m pip install "opencv-python-headless<5"  and restart the server.')
        base = getattr(getattr(cv2, "data", None), "haarcascades", "")
        cascade = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        if cascade.empty():
            raise ModelError("The Haar cascade file could not be loaded from your OpenCV installation.")
        return cascade
    return _cached("cascade", make)


def detect_faces(gray: np.ndarray, max_faces: int = 3, min_size: int = 30):
    cascade = haar_face_cascade()
    with _cascade_lock:
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size))
    faces = sorted([tuple(int(v) for v in f) for f in faces], key=lambda b: b[2] * b[3], reverse=True)
    return faces[:max_faces]


def _face_result(probs, extra=None) -> dict:
    i = int(np.argmax(probs))
    out = {"label": FACE_LABELS[i], "confidence": float(probs[i]), "labels": FACE_LABELS,
           "probs": [float(p) for p in probs], "valence": face_valence(probs)}
    out.update(extra or {})
    return out


def analyse_frame(jpeg_bytes: bytes, max_faces: int = 3) -> dict:
    """Live camera: one JPEG frame in, boxes (normalised 0-1) and emotions out."""
    import cv2
    t0 = time.perf_counter()
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the frame.")
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes = detect_faces(gray, max_faces)
    t_detect = (time.perf_counter() - t0) * 1000
    faces = []
    if boxes:
        grays = np.stack([cv2.resize(gray[y:y + bh, x:x + bw], (48, 48), interpolation=cv2.INTER_AREA)
                          for x, y, bw, bh in boxes]).astype(np.float32)
        t1 = time.perf_counter()
        probs = _predict_face_gray(grays)
        t_predict = (time.perf_counter() - t1) * 1000
        for (x, y, bw, bh), p in zip(boxes, probs):
            faces.append(_face_result(p, {"box": [x / w, y / h, bw / w, bh / h]}))
    else:
        t_predict = 0.0
    return {"width": w, "height": h, "faces": faces,
            "ms": {"detect": t_detect, "predict": t_predict, "total": (time.perf_counter() - t0) * 1000}}


def largest_face_crop(image: Image.Image, margin: float = 0.15):
    """Haar-cascade crop of the largest face; returns (image, found)."""
    import cv2
    arr = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    boxes = detect_faces(gray, 1, min_size=40)
    if not boxes:
        return image, False
    x, y, w, h = boxes[0]
    m = int(margin * w)
    return image.crop((max(x - m, 0), max(y - m, 0), min(x + w + m, image.width), min(y + h + m, image.height))), True


def analyse_image(image: Image.Image, crop: bool = True) -> dict:
    """Photo: same preprocessing as the original predict_face() (PIL grayscale, resize to 48x48)."""
    t0 = time.perf_counter()
    image = image.convert("RGB")
    note, used = None, image
    if crop:
        try:
            used, found = largest_face_crop(image)
            if not found:
                note = "No face was detected, so the whole photo was used."
        except ModelError as exc:
            note = f"Face detection was skipped. {exc}"
    gray = np.asarray(used.convert("L").resize((48, 48)), dtype=np.float32)
    probs = _predict_face_gray(gray[None])[0]
    return _face_result(probs, {"note": note, "face": _jpeg_data_url(used), "ms": (time.perf_counter() - t0) * 1000})


def occlusion_heatmap(image: Image.Image, patch: int = 12, stride: int = 12) -> dict:
    """Model-agnostic importance map: how much the predicted class drops when a region of the 48x48 face is hidden."""
    image = image.convert("RGB")
    gray = np.asarray(image.convert("L").resize((48, 48)), dtype=np.float32)
    base = _predict_face_gray(gray[None])[0]
    cls = int(np.argmax(base))
    fill = float(gray.mean())
    positions = list(range(0, 48 - patch + 1, stride))
    variants, coords = [], []
    for y in positions:
        for x in positions:
            g = gray.copy()
            g[y:y + patch, x:x + patch] = fill
            variants.append(g)
            coords.append((y, x))
    probs = _predict_face_gray(np.stack(variants))
    drop = base[cls] - probs[:, cls]
    heat, count = np.zeros((48, 48)), np.zeros((48, 48))
    for (y, x), d in zip(coords, drop):
        heat[y:y + patch, x:x + patch] += d
        count[y:y + patch, x:x + patch] += 1
    heat = heat / np.maximum(count, 1)
    return {"label": FACE_LABELS[cls], "confidence": float(base[cls]), "overlay": _png_data_url(heatmap_overlay(image, heat))}


def heatmap_overlay(image: Image.Image, heat, size: int = 336) -> Image.Image:
    """Red overlay (alpha ~ importance) on the face image. Negative values are ignored."""
    base = image.convert("RGB").resize((size, size))
    h = np.clip(np.asarray(heat, dtype=np.float32), 0, None)
    if h.max() > 0:
        h = h / h.max()
    a = np.asarray(Image.fromarray((h * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC), dtype=np.float32) / 255.0
    overlay = np.zeros((size, size, 4), dtype=np.uint8)
    overlay[..., 0] = 255
    overlay[..., 1] = (200 * (1 - a)).astype(np.uint8)
    overlay[..., 3] = (a * 0.7 * 255).astype(np.uint8)
    return Image.alpha_composite(base.convert("RGBA"), Image.fromarray(overlay, "RGBA")).convert("RGB")


# ---------------------------------------------------------------------------
# AUDIO + VIDEO model (copied from the training notebook / Streamlit app)
# ---------------------------------------------------------------------------
def _load_av():
    _need_file(AV_CHECKPOINT, "Audio + video")
    if not _importable("torch"):
        raise ModelError("The audio + video model needs the 'torch' package.")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ModalityEncoder(nn.Module):
        def __init__(self, input_dim, d_model, hidden_dim=128, num_layers=1, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                                bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
            self.proj = nn.Linear(hidden_dim * 2, d_model)
            self.norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.dropout(self.norm(self.proj(out)))

    class CrossModalAttention(nn.Module):
        def __init__(self, d_model, num_heads=4, dropout=0.1):
            super().__init__()
            self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model))
            self.norm2 = nn.LayerNorm(d_model)

        def forward(self, query, context):
            attn_out, _ = self.mha(query, context, context)
            x = self.norm(query + attn_out)
            return self.norm2(x + self.ffn(x))

    class AdaptiveFusionGate(nn.Module):
        def __init__(self, d_model, num_modalities=2, hidden_dim=64):
            super().__init__()
            self.gate_net = nn.Sequential(nn.Linear(d_model * num_modalities, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
                                          nn.Linear(hidden_dim, num_modalities))

        def forward(self, reprs):
            weights = F.softmax(self.gate_net(torch.cat(reprs, dim=-1)), dim=-1)
            return torch.sum(weights.unsqueeze(-1) * torch.stack(reprs, dim=1), dim=1), weights

    class AVFusionModel(nn.Module):
        def __init__(self, audio_dim=28, video_dim=2304, d_model=128, num_heads=4, dropout=0.2):
            super().__init__()
            self.audio_encoder = ModalityEncoder(audio_dim, d_model, dropout=dropout)
            self.video_encoder = ModalityEncoder(video_dim, d_model, dropout=dropout)
            self.audio_attends_video = CrossModalAttention(d_model, num_heads, dropout)
            self.video_attends_audio = CrossModalAttention(d_model, num_heads, dropout)
            self.fusion_gate = AdaptiveFusionGate(d_model, num_modalities=2)
            self.regressor = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1))

        def forward(self, audio, video):
            a, v = self.audio_encoder(audio), self.video_encoder(video)
            a_f, v_f = self.audio_attends_video(a, v), self.video_attends_audio(v, a)
            fused, gates = self.fusion_gate([a_f.mean(dim=1), v_f.mean(dim=1)])
            return self.regressor(fused).squeeze(-1), gates

    model = AVFusionModel()
    checkpoint = torch.load(str(AV_CHECKPOINT), map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def av_forward(audio: np.ndarray, video: np.ndarray):
    """(50,28) and (50,2304) float arrays -> (raw score, [audio gate, video gate]). The only function that touches torch for AV."""
    model = _cached("av", _load_av)  # raises a friendly ModelError first
    import torch
    with _av_lock, torch.no_grad():
        pred, gates = model(torch.tensor(np.nan_to_num(audio)[None]), torch.tensor(np.nan_to_num(video)[None]))
    return float(pred.item()), [float(g) for g in gates[0].numpy()]


def _pad(seq, target_len: int) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float32)
    if seq.shape[0] >= target_len:
        return seq[:target_len]
    return np.concatenate([seq, np.zeros((target_len - seq.shape[0], seq.shape[1]), dtype=np.float32)], axis=0)


def extract_wav(src_path: str, wav_path: str, max_seconds: int = 30) -> str:
    if not _importable("imageio_ffmpeg"):
        raise ModelError("Audio extraction needs the 'imageio-ffmpeg' package.")
    import imageio_ffmpeg
    proc = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", src_path, "-vn", "-ac", "1", "-ar", str(AV_AUDIO_SR),
                           "-t", str(max_seconds), "-f", "wav", wav_path], capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.isfile(wav_path) or os.path.getsize(wav_path) < 2000:
        raise ValueError("No audio could be extracted. Does the clip have an audio track?")
    return wav_path


def av_features(video_path: str, workdir: str) -> dict:
    """Same pipeline as the training notebook: MFCC(13)+delta+ZCR+RMS audio, every 3rd frame -> largest face -> 48x48 gray."""
    import cv2
    if not _importable("librosa"):
        raise ModelError("The audio + video model needs the 'librosa' package.")
    import librosa

    wav_path = extract_wav(video_path, os.path.join(workdir, "audio.wav"))
    y, sr = librosa.load(wav_path, sr=AV_AUDIO_SR)
    audio_total_s = len(y) / sr
    if len(y) < AV_HOP:
        y = np.pad(y, (0, AV_HOP - len(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=AV_HOP)
    delta = librosa.feature.delta(mfcc)
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=AV_HOP)
    rms = librosa.feature.rms(y=y, hop_length=AV_HOP)
    audio_feats = np.concatenate([mfcc, delta, zcr, rms], axis=0).T.astype(np.float32)
    audio = _pad(audio_feats, AV_SEQ_LEN)
    audio_used_s = min(audio_feats.shape[0], AV_SEQ_LEN) * AV_HOP / sr

    cascade = haar_face_cascade()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    frames, crops, times, hits, read_idx = [], [], [], 0, 0
    while len(frames) < AV_SEQ_LEN:
        ok, frame = cap.read()
        if not ok:
            break
        if read_idx % AV_VIDEO_STEP == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            with _cascade_lock:
                faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if len(faces) > 0:
                x, y0, w, h = max(faces, key=lambda b: b[2] * b[3])
                crop = cv2.resize(gray[y0:y0 + h, x:x + w], (AV_CROP, AV_CROP))
                frames.append(crop.flatten().astype(np.float32) / 255.0)
                crops.append(crop)
                hits += 1
            else:
                frames.append(np.zeros(AV_CROP * AV_CROP, dtype=np.float32))
                crops.append(None)
            times.append(read_idx / fps)
        read_idx += 1
    cap.release()
    if not frames:
        raise ValueError("No video frames could be read from this clip.")
    return {"audio": audio, "video": _pad(np.stack(frames, axis=0), AV_SEQ_LEN), "face_rate": hits / len(frames),
            "n_frames": len(frames), "audio_total_s": audio_total_s, "audio_used_s": audio_used_s,
            "video_total_s": (frame_count / fps) if frame_count else read_idx / fps, "video_used_s": read_idx / fps,
            "audio_real_frames": min(audio_feats.shape[0], AV_SEQ_LEN), "crops": crops, "frame_times": times,
            "wave": y, "sr": sr}


def _clip_result(feats: dict, raw: float, gates, feature_ms: float, model_ms: float, t0: float) -> dict:
    score = float(np.clip(raw, -3.0, 3.0))
    return {"score": score, "raw": raw, "label": av_polarity(score), "valence": av_valence(score),
            "gate_audio": gates[0], "gate_video": gates[1], "face_rate": feats["face_rate"], "n_frames": feats["n_frames"],
            "audio_total_s": feats["audio_total_s"], "audio_used_s": feats["audio_used_s"],
            "video_total_s": feats["video_total_s"], "video_used_s": feats["video_used_s"],
            "ms": {"features": feature_ms, "model": model_ms, "total": (time.perf_counter() - t0) * 1000}}


def analyse_clip(video_path: str, workdir: str) -> dict:
    t0 = time.perf_counter()
    feats = av_features(video_path, workdir)
    feature_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    raw, gates = av_forward(feats["audio"], feats["video"])
    return _clip_result(feats, raw, gates, feature_ms, (time.perf_counter() - t1) * 1000, t0)


def compute_audio_descriptors(y, sr: int) -> dict:
    """Descriptive acoustic statistics + energy/pitch curves (NOT an emotion classifier)."""
    import librosa
    y = np.asarray(y, dtype=np.float32)
    if len(y) < 4096:
        y = np.pad(y, (0, 4096 - len(y)))
    rms = librosa.feature.rms(y=y, hop_length=AV_HOP)[0]
    f0 = librosa.yin(y, fmin=65, fmax=400, sr=sr, frame_length=2048, hop_length=AV_HOP)
    n = min(len(rms), len(f0))
    rms, f0 = rms[:n], f0[:n]
    f0v = np.where(rms >= np.percentile(rms, 40), f0, np.nan)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=AV_HOP)
    dur = len(y) / sr
    has_pitch = bool(np.isfinite(f0v).any())
    stats = {
        "Duration (s)": round(dur, 2),
        "Mean energy (RMS)": round(float(rms.mean()), 4),
        "Energy variability (CV)": round(float(rms.std() / (rms.mean() + 1e-9)), 3),
        "Median pitch (Hz)": round(float(np.nanmedian(f0v)), 1) if has_pitch else None,
        "Pitch variability (Hz, std)": round(float(np.nanstd(f0v)), 1) if has_pitch else None,
        "Pause ratio": round(float((rms < 0.1 * rms.max()).mean()), 3),
        "Speech onsets per second": round(len(onsets) / max(dur, 1e-6), 2),
    }
    step = max(1, -(-n // 120))          # at most 120 points per curve
    t = (np.arange(n) * AV_HOP / sr)[::step]
    return {"stats": stats, "t": [round(float(v), 3) for v in t], "rms": [round(float(v), 5) for v in rms[::step]],
            "f0": [None if not np.isfinite(v) else round(float(v), 1) for v in f0v[::step]]}


def _windows(real: int, win: int, step: int):
    if real <= win:
        return [(0, real)]
    starts = list(range(0, real - win + 1, step))
    if starts[-1] + win < real:
        starts.append(real - win)
    return [(s0, s0 + win) for s0 in starts]


def explain_clip(video_path: str, workdir: str, win: int = 10, step: int = 5) -> dict:
    """Why this score? How audio and video work together, which moments mattered, which frames and voice features were used.

    * modalities: the score with both streams, with only audio (video zeroed) and with only video (audio zeroed). The model was
      trained on both together, so the single-stream numbers are a sensitivity check, not stand-alone predictions.
    * windows: for each stretch of frames, `effect` = score - score_without_that_stretch. Positive: it pulled the score up.
    """
    t0 = time.perf_counter()
    feats = av_features(video_path, workdir)
    feature_ms = (time.perf_counter() - t0) * 1000
    a, v = feats["audio"], feats["video"]
    t1 = time.perf_counter()
    raw, gates = av_forward(a, v)
    model_ms = (time.perf_counter() - t1) * 1000
    result = _clip_result(feats, raw, gates, feature_ms, model_ms, t0)

    clip = lambda x: float(np.clip(x, -3.0, 3.0))
    audio_only, _ = av_forward(a, np.zeros_like(v))     # video stream removed
    video_only, _ = av_forward(np.zeros_like(a), v)     # audio stream removed

    audio_windows, video_windows = [], []
    a_real, v_real = feats["audio_real_frames"], feats["n_frames"]
    for s0, s1 in _windows(a_real, win, step):
        a2 = a.copy(); a2[s0:s1] = 0
        audio_windows.append({"start_s": s0 * AV_HOP / AV_AUDIO_SR, "end_s": s1 * AV_HOP / AV_AUDIO_SR, "effect": float(raw - av_forward(a2, v)[0])})
    times = feats["frame_times"]
    for s0, s1 in _windows(v_real, win, step):
        v2 = v.copy(); v2[s0:s1] = 0
        video_windows.append({"start_s": float(times[s0]), "end_s": float(times[s1 - 1]), "effect": float(raw - av_forward(a, v2)[0])})

    picks = sorted({int(round(i)) for i in np.linspace(0, v_real - 1, min(8, v_real))})
    frames = []
    for i in picks:
        crop = feats["crops"][i]
        frames.append({"t": float(times[i]), "face": crop is not None,
                       "thumb": _jpeg_data_url(Image.fromarray(crop).resize((96, 96), Image.NEAREST), 96) if crop is not None else None})

    result["explain"] = {
        "modalities": {"both": clip(raw), "audio_only": clip(audio_only), "video_only": clip(video_only)},
        "audio_windows": audio_windows, "video_windows": video_windows, "frames": frames,
        "voice": compute_audio_descriptors(feats["wave"], feats["sr"]),
        "window_frames": win,
    }
    result["ms"]["total"] = (time.perf_counter() - t0) * 1000
    return result


# ---------------------------------------------------------------------------
# health and self-test
# ---------------------------------------------------------------------------
def health() -> dict:
    cascade_ok, cascade_detail = False, ""
    try:
        import cv2
        cascade_ok = hasattr(cv2, "CascadeClassifier")
        cascade_detail = f"OpenCV {cv2.__version__}" + ("" if cascade_ok else " has no CascadeClassifier (use opencv-python-headless<5)")
    except Exception as exc:
        cascade_detail = f"OpenCV not importable: {exc}"
    return {
        "models": {
            "text": {"file": str(TEXT_CHECKPOINT), "found": TEXT_CHECKPOINT.is_file(), "loaded": "text" in _cache},
            "face": {"file": str(FACE_CHECKPOINT), "found": FACE_CHECKPOINT.is_file(), "loaded": "face" in _cache},
            "av": {"file": str(AV_CHECKPOINT), "found": AV_CHECKPOINT.is_file(), "loaded": "av" in _cache},
        },
        "face_detector": {"ok": cascade_ok, "detail": cascade_detail},
        "packages": {n: _importable(n) for n in ("torch", "transformers", "tensorflow", "cv2", "librosa", "imageio_ffmpeg", "multipart")},
    }


def selftest() -> list[dict]:
    results = []

    def run(name, fn):
        t0 = time.perf_counter()
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        results.append({"name": name, "ok": bool(ok), "detail": detail, "ms": (time.perf_counter() - t0) * 1000})

    def t_pos():
        r = analyse_text("I absolutely love this, it is wonderful and works perfectly!")
        return r["label"] == "Positive", f"predicted {r['label']} ({r['confidence'] * 100:.0f}%)"

    def t_neg():
        r = analyse_text("This is terrible. I hate it and it is completely useless.")
        return r["label"] == "Negative", f"predicted {r['label']} ({r['confidence'] * 100:.0f}%)"

    def t_face():
        gray = np.tile(np.linspace(0, 255, 96, dtype=np.uint8), (96, 1))
        r = analyse_image(Image.fromarray(gray).convert("RGB"), crop=False)
        s = sum(r["probs"])
        return len(r["probs"]) == 7 and abs(s - 1.0) < 1e-2, f"7 probabilities summing to {s:.3f}, top: {r['label']}"

    def t_cascade():
        haar_face_cascade()
        return True, "face detector loaded"

    def t_av():
        _cached("av", _load_av)
        return True, "audio + video model loaded"

    run("Face detector (Haar cascade)", t_cascade)
    run("Text model: clearly positive sentence", t_pos)
    run("Text model: clearly negative sentence", t_neg)
    run("Face model: output shape", t_face)
    run("Audio + video model: loads", t_av)
    return results

"""
EmotiSense web server (FastAPI).

Run from the project folder (the one that contains TEXT/, FACE/, "AUDIO AND VIDEO/", backend/ and static/):

    python -m uvicorn server:app --app-dir backend --host 0.0.0.0 --port 8000

Then open http://localhost:8000 . For the phone camera you need HTTPS, see README.md.
"""
from __future__ import annotations

import io
import json
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

import assistant
import engine

log = logging.getLogger("emotisense")
STATIC = Path(__file__).resolve().parent.parent / "static"   # built UI: <project>/static
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_CLIP_BYTES = 80 * 1024 * 1024
MAX_FRAME_BYTES = 2 * 1024 * 1024

@asynccontextmanager
async def lifespan(_: FastAPI):
    assistant.bootstrap_async()      # start Ollama (if needed) and load the model in the background; never blocks the server
    yield
    assistant.shutdown()             # stops Ollama again only if this server started it


app = FastAPI(title="EmotiSense", version="2.0", lifespan=lifespan)


class TextIn(BaseModel):
    text: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class FaceEmotion(BaseModel):
    label: str
    confidence: float
    age_s: float = 0


class ChatIn(BaseModel):
    model: str = Field(min_length=1, max_length=120)
    style: str = "Supportive"
    messages: list[ChatMessage] = Field(min_length=1, max_length=60)
    analyse: bool = True
    face_emotion: Optional[FaceEmotion] = None
    temperature: float = 0.7


@app.exception_handler(engine.ModelError)
async def _model_error(_: Request, exc: engine.ModelError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def _value_error(_: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def _unexpected(_: Request, exc: Exception):
    log.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


def _open_image(data: bytes) -> Image.Image:
    if not data:
        raise HTTPException(400, "No image received.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image is too large (limit 15 MB).")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "This file is not a readable image.")


@app.get("/api/health")
def api_health():
    return engine.health()


@app.get("/api/selftest")
def api_selftest():
    return {"results": engine.selftest()}


@app.post("/api/text")
def api_text(body: TextIn):
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Enter some text first.")
    if len(text) > 5000:
        raise HTTPException(413, "Text is too long (limit 5000 characters).")
    return engine.analyse_text(text)


@app.post("/api/text/explain")
def api_text_explain(body: TextIn):
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Enter some text first.")
    if len(text) > 5000:
        raise HTTPException(413, "Text is too long (limit 5000 characters).")
    return engine.explain_text_words(text)


@app.post("/api/face")
async def api_face(file: UploadFile = File(...), crop: bool = Form(True)):
    return engine.analyse_image(_open_image(await file.read()), crop=crop)


@app.post("/api/face/heatmap")
async def api_face_heatmap(file: UploadFile = File(...), crop: bool = Form(True)):
    image = _open_image(await file.read())
    if crop:
        try:
            image, _ = engine.largest_face_crop(image)
        except engine.ModelError:
            pass
    return engine.occlusion_heatmap(image)


@app.post("/api/face/live")
async def api_face_live(request: Request):
    data = await request.body()
    if not data:
        raise HTTPException(400, "No frame received.")
    if len(data) > MAX_FRAME_BYTES:
        raise HTTPException(413, "Frame is too large.")
    return engine.analyse_frame(data)


@app.post("/api/av")
async def api_av(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "No video received.")
    if len(data) > MAX_CLIP_BYTES:
        raise HTTPException(413, "Clip is too large (limit 80 MB).")
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / f"clip{suffix}")
        Path(path).write_bytes(data)
        return engine.analyse_clip(path, tmp)


@app.post("/api/av/explain")
async def api_av_explain(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "No video received.")
    if len(data) > MAX_CLIP_BYTES:
        raise HTTPException(413, "Clip is too large (limit 80 MB).")
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / f"clip{suffix}")
        Path(path).write_bytes(data)
        return engine.explain_clip(path, tmp)


@app.get("/api/assistant/status")
def api_assistant_status():
    return assistant.status()


@app.post("/api/assistant/start")
def api_assistant_start():
    started = assistant.bootstrap_async(force=True)
    return {"started": started, **{k: v for k, v in assistant.status().items() if k in ("phase", "phase_message", "starting", "warming", "available")}}


@app.post("/api/assistant/chat")
def api_assistant_chat(body: ChatIn):
    messages = [m.model_dump() for m in body.messages]
    if messages[-1]["role"] != "user":
        raise HTTPException(422, "The last message must be from the user.")
    if not messages[-1]["content"].strip():
        raise HTTPException(400, "Type a message first.")
    style = body.style if body.style in assistant.STYLE_PROMPTS else "Supportive"
    temperature = max(0.0, min(1.2, body.temperature))
    face = body.face_emotion.model_dump() if body.face_emotion else None

    def stream():
        for event in assistant.chat_events(body.model, style, messages, body.analyse, face, temperature):
            yield json.dumps(event) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


if (STATIC / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")


@app.get("/")
def index():
    page = STATIC / "index.html"
    if not page.is_file():
        return JSONResponse(status_code=404, content={"detail": "The web UI is not built yet (static/index.html is missing)."})
    return FileResponse(page)


@app.get("/favicon.svg")
def favicon():
    icon = STATIC / "favicon.svg"
    return FileResponse(icon) if icon.is_file() else JSONResponse(status_code=404, content={})

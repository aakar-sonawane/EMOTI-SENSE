"""
Emotion-aware assistant.

Talks to a local Ollama model (a Qwen3 model is preferred when several are installed) and adds context from your own models:
the text-sentiment model estimates how the user's message sounds, and (optionally) the latest face emotion seen by the camera.
Everything stays on the laptop. Only the standard library is used to talk to Ollama.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import engine

LOG_PREFIX = "emotisense.assistant"


def base_url() -> str:
    base = os.environ.get("OLLAMA_HOST", "http://localhost:11434").strip()
    return (base if base.startswith("http") else "http://" + base).rstrip("/")


def list_models(timeout: float = 3.0) -> list[str]:
    with urllib.request.urlopen(base_url() + "/api/tags", timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [m.get("name") or m.get("model") for m in data.get("models", []) if (m.get("name") or m.get("model"))]


def preferred_model(models: list[str]) -> str | None:
    for key in ("qwen3", "qwen"):
        hit = next((m for m in models if key in m.lower()), None)
        if hit:
            return hit
    return models[0] if models else None


def status() -> dict:
    url = base_url()
    phase = _state["phase"]
    extra = {"phase": phase, "phase_message": _state["message"], "starting": phase in ("checking", "starting"), "warming": phase == "warming"}
    try:
        models = list_models()
        return {"available": bool(models), "url": url, "models": models, "preferred": preferred_model(models),
                "error": None if models else "Ollama is running but no model is installed. Run: ollama pull qwen3:8b", **extra}
    except Exception as exc:
        error = _state["message"] if phase in ("missing", "failed", "off") and _state["message"] else \
            f"Cannot reach Ollama at {url} ({exc}). Start it with: ollama serve"
        return {"available": False, "url": url, "models": [], "preferred": None, "error": error, **extra}


# ---------------------------------------------------------------------------
# Start Ollama automatically (so you do not have to run `ollama serve` before the project)
# ---------------------------------------------------------------------------
_state = {"phase": "idle", "message": ""}      # idle | checking | starting | warming | ready | missing | failed | off
_state_lock = threading.Lock()
_proc = None                                   # the `ollama serve` process, only if WE started it


def _set(phase: str, message: str = "") -> None:
    with _state_lock:
        _state["phase"], _state["message"] = phase, message


def autostart_enabled() -> bool:
    return os.environ.get("EMOTISENSE_AUTOSTART_OLLAMA", "1").strip() != "0"


def find_ollama_executable() -> str | None:
    exe = shutil.which("ollama")
    if exe:
        return exe
    candidates = []
    for env, parts in (("LOCALAPPDATA", ("Programs", "Ollama", "ollama.exe")), ("ProgramFiles", ("Ollama", "ollama.exe"))):
        if os.environ.get(env):
            candidates.append(Path(os.environ[env]).joinpath(*parts))
    candidates += [Path(p) for p in ("/usr/local/bin/ollama", "/usr/bin/ollama", "/opt/homebrew/bin/ollama",
                                     "/Applications/Ollama.app/Contents/Resources/ollama")]
    return next((str(p) for p in candidates if p.is_file()), None)


def is_up(timeout: float = 1.5) -> bool:
    try:
        list_models(timeout)
        return True
    except Exception:
        return False


def _spawn(exe: str):
    kwargs = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen([exe, "serve"], **kwargs)


def warm_up(timeout: float = 240.0) -> None:
    """Load the preferred model into memory so the first chat reply is not slow."""
    models = list_models()
    model = preferred_model(models)
    if not model:
        return
    _set("warming", f"Loading {model} into memory…")
    req = urllib.request.Request(base_url() + "/api/generate", data=json.dumps({"model": model, "keep_alive": "30m"}).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()


def bootstrap(force: bool = False) -> None:
    """Make sure Ollama is running (starting it if needed) and warm the model up. Safe to call more than once."""
    global _proc
    if not is_up():
        if not autostart_enabled() and not force:
            _set("off", "Ollama is not running and automatic start is switched off. Run: ollama serve")
            return
        exe = find_ollama_executable()
        if not exe:
            _set("missing", "Ollama is not installed on this computer (or not on PATH). Install it from https://ollama.com and run: ollama pull qwen3:8b")
            return
        _set("starting", "Starting Ollama…")
        try:
            _proc = _spawn(exe)
        except OSError as exc:
            _set("failed", f"Could not start Ollama ({exc}). Start it yourself with: ollama serve")
            return
        deadline = time.time() + float(os.environ.get("EMOTISENSE_OLLAMA_WAIT", "40"))
        while time.time() < deadline and not is_up():
            time.sleep(0.5)
        if not is_up():
            _set("failed", "Ollama was started but did not answer in time. Try again, or run: ollama serve")
            return
    try:
        warm_up()
        _set("ready", "Ollama is ready.")
    except Exception as exc:
        _set("ready", f"Ollama is running (model warm-up skipped: {exc}).")


def bootstrap_async(force: bool = False) -> bool:
    """Run bootstrap() in a background thread unless one is already running. Returns True if a run was started."""
    with _state_lock:
        if _state["phase"] in ("checking", "starting", "warming"):
            return False
        _state["phase"], _state["message"] = "checking", "Checking for Ollama…"
    threading.Thread(target=bootstrap, kwargs={"force": force}, name="ollama-bootstrap", daemon=True).start()
    return True


def shutdown() -> None:
    """Stop Ollama again, but only if this server started it."""
    global _proc
    if _proc is not None:
        try:
            _proc.terminate()
        except Exception:
            pass
        _proc = None


def chat_stream(model: str, messages: list[dict], temperature: float = 0.7, num_predict: int = 400, timeout: float = 180.0):
    payload = {"model": model, "messages": messages, "stream": True, "keep_alive": "10m",
               "options": {"temperature": temperature, "num_predict": num_predict}}
    if "qwen3" in model.lower():
        payload["think"] = False      # Qwen3 would otherwise spend most of the time "thinking" before it answers
    req = urllib.request.Request(base_url() + "/api/chat", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            piece = (chunk.get("message") or {}).get("content", "")
            if piece:
                yield piece
            if chunk.get("done"):
                break


def filter_think_stream(chunks):
    """Hide <think>...</think> reasoning blocks that some Qwen versions print inline."""
    open_tag, close_tag = "<think>", "</think>"
    buf, in_think, strip_next = "", False, False
    for piece in chunks:
        buf += piece
        out = ""
        while True:
            if in_think:
                end = buf.find(close_tag)
                if end == -1:
                    buf = buf[-(len(close_tag) - 1):]
                    break
                buf, in_think, strip_next = buf[end + len(close_tag):], False, True
            else:
                start = buf.find(open_tag)
                if start == -1:
                    keep = next((k for k in range(min(len(open_tag) - 1, len(buf)), 0, -1) if open_tag.startswith(buf[-k:])), 0)
                    out += buf[:len(buf) - keep]
                    buf = buf[len(buf) - keep:]
                    break
                out += buf[:start]
                buf, in_think = buf[start + len(open_tag):], True
        if strip_next and out:
            out = out.lstrip()
            strip_next = not out
        if out:
            yield out
    if buf and not in_think:
        yield buf.lstrip() if strip_next else buf


_CRISIS_RE = re.compile(
    r"(suicid|kill myself|end my life|want to die|wanna die|self[- ]?harm|hurt myself|cut myself|don't want to live|"
    r"khud ko khatam|jaan de|marna chah|mar jaana chah|jeena nahi chah|zindagi khatam)", re.IGNORECASE)
CRISIS_NOTE = ("If you might hurt yourself or you are in danger, please reach out right now. In India you can call Tele-MANAS "
               "on 14416 (or 1-800-891-4416; free, 24x7, many languages) or the emergency number 112. "
               "If you can, tell someone you trust who is near you.")


def needs_crisis_note(text: str) -> bool:
    return bool(_CRISIS_RE.search(text or ""))


STYLE_PROMPTS = {
    "Supportive": "Be warm and gentle. Acknowledge the situation in one sentence, then offer one small, practical suggestion or a caring question.",
    "Neutral": "Be calm, balanced and factual. Reflect what the user said and help them think it through without pushing an opinion.",
    "Professional": "Be concise, polite and professional, like a helpful customer-care or workplace assistant.",
}


def build_system_prompt(style: str, sentiment: dict | None = None, face: dict | None = None, crisis: bool = False) -> str:
    parts = [
        "You are the conversational assistant inside EmotiSense, a student emotion-recognition project.",
        STYLE_PROMPTS.get(style, STYLE_PROMPTS["Neutral"]),
        "Reply in the language the user writes in (English, Hindi or Hinglish) and keep it to 2-5 short sentences.",
        "You are not a therapist or doctor: never diagnose, never claim to know how the user truly feels, and do not give medical advice.",
    ]
    if sentiment:
        parts.append(f"An automatic classifier estimated the user's latest message as {sentiment['label']} "
                     f"(confidence {sentiment['confidence'] * 100:.0f}%). It is imperfect and English-only; treat it only as a hint and trust the user's own words over it.")
    if face:
        parts.append(f"The camera suggested the user looked {face['label'].lower()} about {int(face.get('age_s', 0))} seconds ago "
                     f"(confidence {face['confidence'] * 100:.0f}%). Facial expressions do not reliably show feelings; mention it only if it helps, and gently.")
    if crisis:
        parts.append("The user may be at risk of self-harm. Respond with calm care, take it seriously, give no methods, "
                     "encourage them to contact a trusted person or a helpline right now, and keep the reply short.")
    return " ".join(parts)


def _clean_face(face: dict | None) -> dict | None:
    if not face or face.get("label") not in engine.FACE_LABELS:
        return None
    try:
        conf, age = float(face.get("confidence", 0)), float(face.get("age_s", 0))
    except (TypeError, ValueError):
        return None
    if not (0 <= conf <= 1) or not (0 <= age <= 300):
        return None
    return {"label": face["label"], "confidence": conf, "age_s": age}


def chat_events(model: str, style: str, messages: list[dict], analyse: bool = True, face_emotion: dict | None = None,
                temperature: float = 0.7):
    """Yields dict events: sentiment, notice, crisis, delta..., done | error."""
    last = messages[-1]["content"]
    sentiment = None
    if analyse:
        try:
            r = engine.analyse_text(last)
            sentiment = {"label": r["label"], "confidence": r["confidence"], "valence": r["valence"], "probs": r["probs"], "labels": r["labels"]}
            yield {"type": "sentiment", **sentiment}
        except Exception as exc:     # the chat must keep working even if the sentiment model is unavailable
            yield {"type": "notice", "message": f"Sentiment estimate skipped: {exc}"}
    crisis = needs_crisis_note(last)
    if crisis:
        yield {"type": "crisis", "note": CRISIS_NOTE}
    face = _clean_face(face_emotion)
    convo = [{"role": "system", "content": build_system_prompt(style, sentiment, face, crisis)}] + \
            [{"role": m["role"], "content": m["content"]} for m in messages[-12:]]
    try:
        for piece in filter_think_stream(chat_stream(model, convo, temperature=temperature)):
            yield {"type": "delta", "text": piece}
        yield {"type": "done"}
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("error", "")
        except Exception:
            detail = ""
        hint = f" Install it with: ollama pull {model}" if exc.code == 404 else ""
        yield {"type": "error", "message": f"Ollama answered with an error ({exc.code}): {detail or exc.reason}.{hint}"}
    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
        yield {"type": "error", "message": f"Cannot reach Ollama at {base_url()} ({getattr(exc, 'reason', exc)}). Start it with: ollama serve"}
    except Exception as exc:
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}

# EmotiSense (FastAPI + React)

Real-time emotion and sentiment UI for your three models. Works on laptop and phone.

```
EMOTION RECOGNITION/            <- your project folder
├── TEXT/ FACE/ "AUDIO AND VIDEO"/   (your models, unchanged)
├── backend/    server.py (web routes), engine.py (model loading + preprocessing), requirements.txt, tests/
├── frontend/   React + Tailwind source (only needed if you want to change the design)
├── static/     the built UI (the server shows this; no Node needed)
├── start.bat   double-click launcher (opens the browser, starts the server and Ollama)
└── old_ui_backup/   put your old Streamlit files here (app.py, pages_modules.py, models.py, utils.py ...)
```

## What is in the UI
| Tab | What it does |
|---|---|
| Live | Camera in the browser, face boxes + emotion, only the newest frame is sent (no lag build-up) |
| Photo | Emotion from a photo, plus a heatmap of where the face model looked |
| Text | Sentiment, and which words mattered (green supports, red opposes) |
| Clip | Audio + video sentiment score; **Why this score?** shows how audio and video work together, which moments mattered, the frames used and voice features |
| Assistant | Emotion-aware chat with your local Ollama model (Qwen3 preferred). Uses the text-sentiment model and, optionally, your latest face emotion |
| History | Every prediction with a timestamp, graph, CSV download |
| System | Which model files exist, and a self-test that runs your real models |

## 1. One-time setup (Windows PowerShell)
```powershell
cd "D:\EMOTION RECOGNITION"
py -3.12 -m venv venv            # or py -3.11
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

## 2. Run (or just double-click start.bat)
```powershell
python -m uvicorn server:app --app-dir backend --host 0.0.0.0 --port 8000
```
Open http://localhost:8000 . Go to the **System** tab first: it shows which model files were found and has a
**Run self-test** button that runs your real models on built-in inputs.

## 3. The Assistant (Ollama starts by itself)
You do **not** have to run `ollama serve` first. When the server starts it checks whether Ollama is running; if not, it starts
Ollama in the background (it looks on PATH and in `%LOCALAPPDATA%\Programs\Ollama`) and loads your Qwen3 model into memory
(kept for 30 minutes), so the first answer is not slow. The Assistant tab shows "Starting Ollama..." while this happens.
When you stop the server, Ollama is stopped too, but only if the server started it.

Only once: `ollama pull qwen3:8b`.

Settings (environment variables, optional):
* `EMOTISENSE_AUTOSTART_OLLAMA=0` -> never start Ollama automatically
* `EMOTISENSE_OLLAMA_WAIT=60` -> seconds to wait for Ollama to answer (default 40)
* `OLLAMA_HOST=127.0.0.1:11434` -> if your Ollama listens somewhere else

**Easiest way to start everything:** double-click `start.bat` in the project folder (copy it next to `backend/`). It opens the
browser and starts the server (and therefore Ollama).

## 4. Use it on a phone
The phone's live camera only works on **HTTPS**. Easiest way (free):
```powershell
cloudflared tunnel --url http://localhost:8000
```
Open the https://....trycloudflare.com link on the phone. Without HTTPS the **Photo** and **Clip** tabs still work
("Take photo" / "Record clip" open the phone camera app), only the Live tab is blocked.
Keep laptop and phone on the same Wi-Fi for the lowest latency, and allow port 8000 in Windows Firewall.

## 5. Tests
```powershell
python -m pip install pytest httpx scikit-image
python -m pytest backend/tests -q            # backend
cd frontend; npm install; npm test           # UI (needs Node 18+)
```

## 6. Change the design / rebuild the UI
```powershell
cd frontend
npm install
npm run dev        # live preview on http://localhost:5173 (needs the backend running on port 8000)
npm run build      # writes ../static
```

## Troubleshooting
* Assistant says "Ollama is not installed" -> install it from https://ollama.com. "Cannot reach Ollama" -> press "Check again" (it asks the server to start it), or run `ollama serve`. "model not found" -> `ollama pull <name>`.
* Qwen3 answers slowly the first time -> the model is loading into memory (`keep_alive` keeps it loaded for 10 minutes afterwards).
* `no attribute 'CascadeClassifier'` -> OpenCV 5 is installed. Run
  `python -m pip uninstall -y opencv-python opencv-python-headless` then `python -m pip install "opencv-python-headless<5"`.
* A model says "checkpoint not found" -> the file must be at exactly these paths:
  `TEXT/best_model_finetuned.pt`, `FACE/EVALUATED_MODEL/robust_v2_epoch4.keras`, `AUDIO AND VIDEO/best_model.pt`.
* Live tab is slow -> the 3-backbone face model is heavy on CPU (a few updates per second). Frames are sent one at a time, so it never lags behind, it just updates less often. A GPU or a lighter model is the real fix.

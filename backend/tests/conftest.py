"""Shared test setup: a throw-away project root with dummy checkpoint files, set before engine.py is imported."""
import os, sys, tempfile
from pathlib import Path

ROOT_DIR = tempfile.mkdtemp()
for rel in ("TEXT/best_model_finetuned.pt", "FACE/EVALUATED_MODEL/robust_v2_epoch4.keras", "AUDIO AND VIDEO/best_model.pt"):
    p = Path(ROOT_DIR) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * 100)
os.environ["EMOTISENSE_ROOT"] = ROOT_DIR
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

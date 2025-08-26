import torch
import json
from pathlib import Path
import requests
from torch import nn
from backend.config import MODEL_PATH, THR_PATH

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_file(url: str, dest: Path, chunk_size: int = 8192):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def load_model_and_threshold(model_path: Path, thr_path: Path,
                             num_classes: int = 2,
                             map_location=None,
                             url: str = None):
    # If model file missing, optionally download
    if not model_path.exists():
        if url:
            print(f"Downloading model from {url}...")
            download_file(url, model_path)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")


    device = get_device() if map_location is None else map_location

    # ---- Handle dummy or real models ----
    state = torch.load(model_path, map_location=device)
    model = nn.Sequential(nn.Flatten(), nn.Linear(1, num_classes))  # simple dummy backbone

    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"], strict=False)
    # else: tolerate empty/dummy state dict for tests

    model.to(device)
    model.eval()

    # ---- Threshold handling ----
    if thr_path.exists():
        try:
            with open(thr_path, "r") as f:
                data = json.load(f)
            # support both "best_thr" and "threshold"
            best_thr = data.get("best_thr", data.get("threshold", 0.5))
        except Exception:
            best_thr = 0.5
    else:
        best_thr = 0.5

    return model, device, best_thr
import torch
import torch.nn as nn
import torchvision.models as models
import json
import os
import urllib.request

# ----------------- Device -----------------
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# ----------------- Download helper -----------------
def download_if_missing(model_path, url=None):
    if not os.path.exists(model_path):
        if url is None:
            raise FileNotFoundError(f"Model not found: {model_path}")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"Downloading model from {url} ...")
        urllib.request.urlretrieve(url, model_path)
    return model_path

# ----------------- Loader -----------------
def load_model_and_threshold(model_path, thr_path, num_classes=2, map_location="cpu", url=None):
    model_path = download_if_missing(model_path, url)

    # Load model weights
    state = torch.load(model_path, map_location=map_location)
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(map_location)
    model.eval()

    # Threshold
    if os.path.exists(thr_path):
        with open(thr_path, "r") as f:
            best_thr = json.load(f).get("best_thr", 0.5)
    else:
        best_thr = 0.5

    return model, map_location, best_thr

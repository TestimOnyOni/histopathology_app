import os
import torch
import urllib.request

def download_model_if_missing(model_path: str, url: str):
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"Downloading model from {url}...")
        urllib.request.urlretrieve(url, model_path)
    return model_path


def load_model_and_threshold(model_path, thr_path, num_classes, map_location, url=None):
    if url:
        model_path = download_model_if_missing(model_path, url)

    # Load model weights
    state = torch.load(model_path, map_location=map_location)

    # Example: build model (adapt to your architecture)
    from torchvision import models
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(map_location)
    model.eval()

    # Load threshold
    if os.path.exists(thr_path):
        with open(thr_path, "r") as f:
            best_thr = float(f.read().strip())
    else:
        best_thr = 0.5  # fallback

    return model, map_location, best_thr
import torch
import torch.nn as nn
import torchvision.models as models
import json
from pathlib import Path


def get_device():
    """Return GPU if available else CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_threshold(model_path, thr_path=None, num_classes=2, map_location="cpu"):
    """
    Load model weights and threshold, adapting automatically to checkpoint output shape.
    """
    state = torch.load(model_path, map_location=map_location)

    # Start with base ResNet
    model = models.resnet50(weights=None)

    # Determine number of output neurons from checkpoint if possible
    if "fc.weight" in state:
        out_features = state["fc.weight"].shape[0]
    else:
        out_features = num_classes

    model.fc = nn.Linear(model.fc.in_features, out_features)

    # Load weights (ignore missing if fc doesn't match perfectly)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] Missing keys: {missing}, Unexpected keys: {unexpected}")

    # Load threshold
    if thr_path and Path(thr_path).exists():
        with open(thr_path, "r") as f:
            data = json.load(f)
            best_thr = data.get("best_thr", 0.5)
    else:
        best_thr = 0.5

    device = get_device()
    model.to(device)
    model.eval()

    return model, best_thr, device

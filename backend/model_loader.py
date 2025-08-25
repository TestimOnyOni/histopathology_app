import json
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_resnet50_model(num_classes=2, pretrained=True):
    if pretrained:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def load_model_and_threshold(model_path: Path, threshold_json: Path, num_classes=2, map_location=None):
    device = map_location or get_device()
    model = get_resnet50_model(num_classes=num_classes, pretrained=False)
    state = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    best_threshold = 0.5
    if threshold_json and threshold_json.exists():
        with open(threshold_json, "r") as f:
            data = json.load(f)
        best_threshold = float(data.get("best_threshold", data.get("threshold", 0.5)))

    return model, device, best_threshold
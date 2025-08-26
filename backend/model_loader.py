import json
import torch
import torch.nn as nn
import torchvision.models as models

def load_model_and_threshold(model_path, threshold_path, num_classes=2, map_location="cpu"):
    """
    Load trained ResNet50 model and threshold value from disk.
    """
    # --- Load model ---
    model = models.resnet50(weights=None)  # initialize without pretrained weights
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    state = torch.load(model_path, map_location=map_location)
    model.load_state_dict(state, strict=False)
    model.to(map_location)
    model.eval()

    # --- Load threshold ---
    with open(threshold_path, "r") as f:
        thr = json.load(f)

    return model, map_location, thr.get("best_thr", 0.5)
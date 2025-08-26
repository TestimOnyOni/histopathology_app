import os
import torch

# Automatically detect root directory (repo root where app.py lives)
APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths to model + threshold JSON
MODEL_PATH = os.path.join(APP_ROOT, "best_resnet50_balanced.pth")
THR_PATH = os.path.join(APP_ROOT, "deploy_resnet50_slide_threshold.json")

# Model constants
DEFAULT_NUM_CLASSES = 2  # Benign vs Malignant

# Device getter
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
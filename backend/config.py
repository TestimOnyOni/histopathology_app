from pathlib import Path

# Base paths (Streamlit Cloud uses working dir == repo root)
BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
# MODELS_DIR = ASSETS_DIR / "models"
MODELS_DIR = BASE_DIR / "models"

# Default model + threshold file names (override in Streamlit sidebar if needed)
DEFAULT_MODEL_PATH = MODELS_DIR / "best_resnet50_balanced.pth"
DEFAULT_THRESHOLD_PATH = MODELS_DIR / "deploy_resnet50_slide_threshold.json"

# Inference defaults
DEFAULT_NUM_CLASSES = 2
DEFAULT_AGG_METHOD = "percentile"   # options: percentile, mean, max
DEFAULT_PERCENTILE = 90
DEFAULT_THRESHOLD = 0.39            # will be loaded from JSON if available
ALLOWED_IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Preprocessing
IMAGE_SIZE = 224                    # ResNet50 default input size



# from pathlib import Path
import torch

# Root of repo (one directory up from this file)
ROOT_DIR = Path(__file__).resolve().parent.parent  

# Explicitly point to models directory
MODEL_PATH = ROOT_DIR / "models" / "best_resnet50_balanced.pth"
THR_PATH   = ROOT_DIR / "models" / "deploy_resnet50_slide_threshold.json"

# DEFAULT_NUM_CLASSES = 2

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

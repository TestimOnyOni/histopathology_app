import os

# Base directory (backend is sibling of models/)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_resnet50_balanced.pth")
THR_PATH   = os.path.join(BASE_DIR, "models", "thr.json")

# Remote model (optional, used if MODEL_PATH missing)
MODEL_URL = "https://github.com/TestimOnyOni/histopathology_app/raw/main/models/best_resnet50_balanced.pth"

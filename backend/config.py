# from pathlib import Path

# # Paths
# BASE_DIR = Path(__file__).resolve().parent.parent
# MODEL_DIR = BASE_DIR / "models"

# # Model files
# MODEL_PATH = MODEL_DIR / "best_resnet50_balanced.pth"
# THR_PATH = MODEL_DIR / "deploy_resnet50_slide_threshold.json"

# # Model download URL (use raw GitHub link, not blob!)
# MODEL_URL = "https://raw.githubusercontent.com/TestimOnyOni/histopathology_app/main/models/best_resnet50_balanced.pth"

# # Inference settings
# DEFAULT_NUM_CLASSES = 2


from pathlib import Path

# Paths inside the container
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_resnet50_balanced.pth"
THR_PATH = BASE_DIR / "models" / "thr.json"

# Remote fallback (raw GitHub link!)
MODEL_URL = "https://raw.githubusercontent.com/TestimOnyOni/histopathology_app/main/models/best_resnet50_balanced.pth"

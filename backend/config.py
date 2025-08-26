from pathlib import Path

# Paths inside the container
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_resnet50_balanced.pth"
THR_PATH = BASE_DIR / "models" / "thr.json"

# Remote fallback (raw GitHub link!)
MODEL_URL = "https://raw.githubusercontent.com/TestimOnyOni/histopathology_app/main/models/best_resnet50_balanced.pth"
# https://github.com/TestimOnyOni/histoPathologySystem/blob/main/best_resnet50_balanced.pth
# MODEL_URL = "https://raw.githubusercontent.com/TestimOnyOni/histoPathologySystem/blob/main/best_resnet50_balanced.pth"
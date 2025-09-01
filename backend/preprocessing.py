import os
import zipfile
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as T

# ----------------- Image transform -----------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ----------------- Safe extraction -----------------
def safe_extract(zip_ref, path):
    for member in zip_ref.namelist():
        member_path = os.path.join(path, member)
        if not os.path.abspath(member_path).startswith(os.path.abspath(path)):
            raise Exception("Unsafe path in zip file!")
    zip_ref.extractall(path)

# ----------------- File handler -----------------
def handle_uploaded_file(uploaded_file):
    """Accepts either a single image or a ZIP of images. Returns list of Tensors."""
    patches = []

    if uploaded_file.name.endswith(".zip"):
        with zipfile.ZipFile(uploaded_file) as zip_ref:
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            safe_extract(zip_ref, temp_dir)

            for fname in os.listdir(temp_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                    img = Image.open(os.path.join(temp_dir, fname)).convert("RGB")
                    patches.append(transform(img))
    else:
        img = Image.open(uploaded_file).convert("RGB")
        patches.append(transform(img))

    return patches

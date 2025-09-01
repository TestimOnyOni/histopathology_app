import os
import zipfile
import tempfile
from PIL import Image
import torch
from torchvision import transforms
import streamlit as st

# --- Transform definition ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_image(img_file):
    """Load a single image file as tensor."""
    try:
        image = Image.open(img_file).convert("RGB")
        return transform(image)
    except Exception as e:
        st.error(f"❌ Error loading image {img_file}: {e}")
        return None

def handle_uploaded_file(uploaded_file):
    """
    Handles uploaded file (single image or zip).
    Returns: list of tensors
    """
    patches = []

    if uploaded_file.name.lower().endswith(".zip"):
        # --- Extract zip to a temp folder ---
        tmpdir = tempfile.mkdtemp()
        zip_path = os.path.join(tmpdir, uploaded_file.name)
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.read())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        # Walk through extracted files
        for root, _, files in os.walk(tmpdir):
            for fname in files:
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    fpath = os.path.join(root, fname)
                    tensor = load_image(fpath)
                    if tensor is not None:
                        patches.append(tensor)

    else:
        # --- Single image case ---
        tensor = load_image(uploaded_file)
        if tensor is not None:
            patches.append(tensor)

    if not patches:
        raise ValueError("No valid images found in upload.")

    return patches

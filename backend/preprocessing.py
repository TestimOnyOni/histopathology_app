import os
import zipfile
import tempfile
from PIL import Image
import streamlit as st
import io

def handle_uploaded_file(uploaded_file):
    """
    Handle uploaded image or ZIP file.
    Returns list of PIL.Image objects.
    """
    patches = []

    if uploaded_file.name.lower().endswith((".png", ".jpg", ".jpeg")):
        # Single image
        image = Image.open(uploaded_file).convert("RGB")
        patches.append(image)

    elif uploaded_file.name.lower().endswith(".zip"):
        # Extract ZIP
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, uploaded_file.name)
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                safe_extract(zip_ref, tmpdir)

            # Collect image files
            for root, _, files in os.walk(tmpdir):
                for fname in files:
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        try:
                            img_path = os.path.join(root, fname)
                            img = Image.open(img_path).convert("RGB")
                            patches.append(img)
                        except Exception as e:
                            st.warning(f"⚠️ Could not load {fname}: {e}")
    return patches

def safe_extract(zip_ref, path):
    """
    Safe extraction to prevent path traversal attacks.
    """
    for member in zip_ref.namelist():
        member_path = os.path.abspath(os.path.join(path, member))
        if not member_path.startswith(os.path.abspath(path)):
            raise Exception("Unsafe path detected in ZIP!")
    zip_ref.extractall(path)


def load_single_image(uploaded_file):
    """
    Load a single image from an uploaded file (Streamlit UploadedFile or file-like).
    Supports .read(), .getbuffer(), or direct BytesIO objects.
    """
    try:
        if hasattr(uploaded_file, "read"):  
            # Streamlit UploadedFile or file-like object
            img = Image.open(io.BytesIO(uploaded_file.read()))
        elif hasattr(uploaded_file, "getbuffer"):
            # Some mocks or special upload objects
            img = Image.open(io.BytesIO(uploaded_file.getbuffer()))
        else:
            raise ValueError("Unsupported uploaded file type")

        return img.convert("RGB")  # normalize mode
    except Exception as e:
        raise RuntimeError(f"Failed to load image: {e}")

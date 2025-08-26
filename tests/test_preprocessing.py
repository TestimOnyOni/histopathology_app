# tests/test_preprocessing.py

import io
import zipfile
import pytest
from PIL import Image

from backend.preprocessing import handle_uploaded_file


class DummyUpload:
    """
    Dummy class to simulate Streamlit's UploadedFile.
    It provides .read(), .seek(), and .name attributes.
    """
    def __init__(self, data: bytes, name: str):
        self._file = io.BytesIO(data)
        self.name = name

    def read(self):
        return self._file.read()

    def seek(self, pos, whence=0):
        return self._file.seek(pos, whence)


def create_test_image_bytes(color=(255, 0, 0), size=(64, 64)):
    """Helper: create an in-memory PNG image as bytes."""
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_single_image(tmp_path):
    """Test that a single image upload is handled correctly."""
    img_bytes = create_test_image_bytes()
    dummy_file = DummyUpload(img_bytes, "test.png")

    images = handle_uploaded_file(dummy_file, extract_dir=tmp_path)

    assert len(images) == 1
    assert isinstance(images[0], Image.Image)


def test_zip_images(tmp_path):
    """Test that a ZIP containing images is handled correctly."""
    img_bytes = create_test_image_bytes()

    # Create an in-memory zip file with two PNGs
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("img1.png", img_bytes)
        zf.writestr("img2.png", img_bytes)

    zip_buf.seek(0)
    dummy_zip = DummyUpload(zip_buf.read(), "images.zip")

    images = handle_uploaded_file(dummy_zip, extract_dir=tmp_path)

    assert len(images) == 2
    assert all(isinstance(img, Image.Image) for img in images)
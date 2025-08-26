# backend/preprocessing.py

import io
import zipfile
from pathlib import Path
from typing import List, Union

from PIL import Image


def safe_extract(zip_ref: zipfile.ZipFile, path: Union[str, Path]) -> None:
    """
    Safely extract a ZIP file into the target path.
    Prevents path traversal vulnerabilities.
    """
    path = Path(path)
    for member in zip_ref.infolist():
        member_path = path / member.filename
        if not str(member_path.resolve()).startswith(str(path.resolve())):
            raise Exception("Unsafe path detected in zip file!")
    zip_ref.extractall(path)


def _to_filelike(uploaded_file) -> io.BytesIO:
    """
    Normalize uploaded_file into a file-like object (BytesIO).
    Works for Streamlit UploadedFile, bytes, or dummy test classes.
    """
    if hasattr(uploaded_file, "read"):  # Streamlit UploadedFile or real file
        data = uploaded_file.read()
        return io.BytesIO(data)
    elif isinstance(uploaded_file, (bytes, bytearray)):
        return io.BytesIO(uploaded_file)
    else:
        raise TypeError(f"Unsupported uploaded_file type: {type(uploaded_file)}")


def handle_uploaded_file(uploaded_file, extract_dir: Union[str, Path] = "uploaded_data") -> List[Image.Image]:
    """
    Handle an uploaded file (single image or ZIP of images).

    Args:
        uploaded_file: file-like object (Streamlit UploadedFile, bytes, or dummy).
        extract_dir: folder where ZIP contents will be extracted.

    Returns:
        List of PIL.Image objects.
    """
    filename = getattr(uploaded_file, "name", "uploaded_file")

    # Convert to BytesIO to ensure .read() and .seek() exist
    file_like = _to_filelike(uploaded_file)

    # Case 1: ZIP archive
    if filename.lower().endswith(".zip"):
        images = []
        extract_dir = Path(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(file_like) as zip_ref:
            safe_extract(zip_ref, extract_dir)

            for f in zip_ref.namelist():
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    try:
                        img_path = extract_dir / f
                        img = Image.open(img_path).convert("RGB")
                        images.append(img)
                    except Exception as e:
                        print(f"⚠️ Could not open {f}: {e}")

        return images

    # Case 2: Single image
    else:
        try:
            img = Image.open(file_like).convert("RGB")
            return [img]
        except Exception as e:
            print(f"⚠️ Could not load image: {e}")
            return []
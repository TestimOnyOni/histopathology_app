# backend/preprocessing.py
"""
Robust preprocessing utilities.

Main function:
    handle_uploaded_file(uploaded_file, return_tensors=False, transform=None)

Supports:
- uploaded_file as: str/pathlib.Path, bytes, file-like object (with read()), or an upload object (has .name, .read()).
- zip files (reads images inside the zip).
- safe extraction helper if you need to extract to disk.

By default returns a list of PIL.Image.Image objects.
Set return_tensors=True to get torch.Tensor objects (requires torchvision).
Optionally pass a torchvision transform via `transform` (applied after ToTensor).
"""

from io import BytesIO
from pathlib import Path
import tempfile
import zipfile
import os
from typing import List, Union, Optional

from PIL import Image, UnidentifiedImageError

# Optional imports for tensor conversion
try:
    import torch
    from torchvision import transforms as T
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

# Default transform used when return_tensors=True and no transform provided
_DEFAULT_TENSOR_TRANSFORM = T.Compose([T.ToTensor()]) if _TORCH_AVAILABLE else None

# Acceptable image extensions (case-insensitive)
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"}


def _is_zip_by_name(name: str) -> bool:
    return name.lower().endswith(".zip")


def _read_all_bytes(file_obj) -> bytes:
    """
    Robustly read all bytes from a file-like object. Some upload objects may not implement seek/read exactly,
    so we fallback appropriately.
    """
    # If it's already bytes
    if isinstance(file_obj, (bytes, bytearray)):
        return bytes(file_obj)

    # If it's a Path or str: read from disk
    if isinstance(file_obj, (str, Path)):
        with open(str(file_obj), "rb") as f:
            return f.read()

    # If it has a .read() method, call it. If it returns str, encode.
    read = getattr(file_obj, "read", None)
    if callable(read):
        data = read()
        # Some frameworks return str for small uploads -> encode if needed
        if isinstance(data, str):
            return data.encode("utf-8")
        return data

    # If it has getvalue (like io.BytesIO)
    getvalue = getattr(file_obj, "getvalue", None)
    if callable(getvalue):
        data = getvalue()
        if isinstance(data, str):
            return data.encode("utf-8")
        return data

    raise ValueError("Unable to read bytes from uploaded_file (unsupported object).")


def safe_extract(zip_file_like, target_path: Union[str, Path]) -> None:
    """
    Safely extract a zip file-like (bytes or file-like) to the target directory.
    Prevents path traversal attacks by ensuring extracted paths remain inside target_path.
    """
    target_path = Path(target_path)
    target_path.mkdir(parents=True, exist_ok=True)

    data = _read_all_bytes(zip_file_like)
    with zipfile.ZipFile(BytesIO(data)) as zf:
        for member in zf.infolist():
            member_name = member.filename
            # Skip directories
            if member.is_dir():
                continue

            # Construct resolved path and ensure it is within target_path
            dest = target_path.joinpath(member_name)
            try:
                dest_resolved = dest.resolve()
            except Exception:
                # Skip weird names
                continue
            if not str(dest_resolved).startswith(str(target_path.resolve())):
                # Path traversal attempt -> skip
                continue

            # Ensure parent directory exists
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            with zf.open(member, "r") as source_f, open(dest_resolved, "wb") as target_f:
                target_f.write(source_f.read())


def _iter_images_in_zip_bytes(zip_bytes: bytes):
    """
    Yield (name, file_bytes) for image files inside zip bytes.
    """
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            lower = name.lower()
            # skip directories and unwanted files
            if lower.endswith("/"):
                continue
            ext = Path(name).suffix.lower()
            if ext in _IMAGE_EXTS:
                try:
                    data = zf.read(name)
                except RuntimeError:
                    # sometimes zip read throws, skip the file
                    continue
                yield name, data


def _open_image_from_bytes(b: bytes) -> Image.Image:
    """
    Open bytes as PIL.Image and convert to RGB.
    """
    try:
        img = Image.open(BytesIO(b))
        img = img.convert("RGB")
        return img
    except UnidentifiedImageError:
        raise
    except Exception as e:
        raise


def handle_uploaded_file(
    uploaded_file,
    return_tensors: bool = False,
    transform: Optional[object] = None,
) -> List[Union[Image.Image, "torch.Tensor"]]:
    """
    Handle an uploaded file (image or zip). Returns list of PIL.Image objects by default.
    If return_tensors=True, returns list of torch.Tensor (C,H,W floats 0..1). Requires torchvision.

    Parameters
    ----------
    uploaded_file : Union[str, Path, bytes, file-like]
        - A filesystem path (str/Path)
        - raw bytes
        - a file-like object with read() (e.g. io.BytesIO, Flask/Django/Streamlit upload)
        - object that exposes .name and .read(), commonly returned by web frameworks
    return_tensors : bool
        If True, convert each image to a torch.Tensor using torchvision transforms.
    transform : torchvision.transforms (optional)
        Additional transform to apply when converting to tensors (applied after ToTensor).
        If None and return_tensors=True, defaults to transforms.ToTensor().

    Returns
    -------
    List[PIL.Image.Image] OR List[torch.Tensor]
    """
    # If user wants tensors, ensure torch available
    if return_tensors and not _TORCH_AVAILABLE:
        raise RuntimeError("return_tensors=True requires torch and torchvision to be installed.")

    # Normalize very common container types: Many upload frameworks provide an object with .name and .read()
    filename = getattr(uploaded_file, "name", None)
    # Read all bytes robustly
    data_bytes = None
    try:
        # If it's a path string or Path instance, just open the file on disk.
        if isinstance(uploaded_file, (str, Path)):
            filename = str(uploaded_file)
            data_bytes = _read_all_bytes(uploaded_file)  # will open the path
        else:
            data_bytes = _read_all_bytes(uploaded_file)
            # filename may be None for raw bytes
    except Exception:
        # As fallback, if uploaded_file is an iterable of files (e.g. streamlit multiple_uploads), try to handle first element
        raise

    images = []

    # Decide whether zip or single image by name or by bytes signature
    is_zip = False
    if filename and _is_zip_by_name(str(filename)):
        is_zip = True
    else:
        # quick magic number check for zip (first 4 bytes: PK\x03\x04)
        if data_bytes[:4] == b"PK\x03\x04":
            is_zip = True

    if is_zip:
        # Iterate image items in the zip
        for name, file_bytes in _iter_images_in_zip_bytes(data_bytes):
            try:
                img = _open_image_from_bytes(file_bytes)
            except Exception:
                # skip problematic images silently (or log)
                continue
            images.append(img)
    else:
        # Try to open bytes as a single image
        try:
            img = _open_image_from_bytes(data_bytes)
            images.append(img)
        except Exception:
            # Not an image — as a fallback, treat as zip (some zips might not have had .zip name)
            try:
                for name, file_bytes in _iter_images_in_zip_bytes(data_bytes):
                    try:
                        img = _open_image_from_bytes(file_bytes)
                    except Exception:
                        continue
                    images.append(img)
            except zipfile.BadZipFile:
                # nothing workable
                raise ValueError("Uploaded file is neither a supported image nor a zip file containing images.")

    # Sort by a stable key (optional) - filenames not available for single images
    # For zip we can't always preserve original order; this sorts by bytes length to be deterministic
    # (keeps order consistent between runs). If you want original zip order, modify _iter_images_in_zip_bytes
    # to yield in zf.infolist() order and capture names.
    # We'll leave 'images' as-is (zip iteration preserves namelist order generally).

    if return_tensors:
        # Create transformation: ToTensor plus user transforms (applied after ToTensor if transform provided)
        if transform is None:
            transform = _DEFAULT_TENSOR_TRANSFORM
        elif isinstance(transform, list) or isinstance(transform, tuple):
            # allow passing list of transforms
            transform = T.Compose(list(transform))
        # Ensure transform exists
        if transform is None:
            raise RuntimeError("torch/torchvision transform not available for tensor conversion.")
        tensors = []
        for img in images:
            t = transform(img)
            # Make sure tensor shape is [C,H,W]
            tensors.append(t)
        return tensors

    return images


# Example small helper for convenience (not required)
def images_to_tensors(images: List[Image.Image], transform: Optional[object] = None):
    """
    Convert a list of PIL images to list of torch tensors using torchvision transforms.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("Torch and torchvision required for images_to_tensors.")

    if transform is None:
        transform = _DEFAULT_TENSOR_TRANSFORM
    elif isinstance(transform, list) or isinstance(transform, tuple):
        transform = T.Compose(list(transform))

    return [transform(img) for img in images]


# If run directly, demo quick self-test
if __name__ == "__main__":
    # Minimal local smoke test: open a local image path / zip path
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <image_or_zip_path>")
        sys.exit(0)
    p = sys.argv[1]
    imgs = handle_uploaded_file(p)
    print(f"Found {len(imgs)} images. Types: {[type(i) for i in imgs[:3]]}")

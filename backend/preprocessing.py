from pathlib import Path
from PIL import Image
import io, zipfile
import torch
from torchvision import transforms
from .config import ALLOWED_IMG_EXTS, IMAGE_SIZE

# Basic transform for evaluation
def build_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_IMG_EXTS

def load_image_from_bytes(content: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(content)).convert("RGB")
    return img

def preprocess_single_image(pil_img: Image.Image, transform=None) -> torch.Tensor:
    transform = transform or build_eval_transform()
    return transform(pil_img).unsqueeze(0)  # shape: [1, C, H, W]

def iter_zip_images(zip_bytes: bytes):
    """
    Yields (name, PIL.Image) for each valid image in a ZIP.
    Skips non-image files and unreadable entries.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename)
            if not is_image_file(name):
                continue
            try:
                with zf.open(info) as f:
                    img = Image.open(io.BytesIO(f.read())).convert("RGB")
                yield str(name), img
            except Exception:
                continue

def preprocess_zip_to_tensor_list(zip_bytes: bytes, transform=None):
    """
    Returns (tensor_list, names) where:
      - tensor_list: list[torch.Tensor: shape [1,C,H,W]]
      - names: list[str] original names inside the zip
    """
    transform = transform or build_eval_transform()
    tensors, names = [], []
    for name, img in iter_zip_images(zip_bytes):
        tensors.append(transform(img).unsqueeze(0))
        names.append(name)
    return tensors, names
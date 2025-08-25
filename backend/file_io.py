import re
from pathlib import Path

def secure_filename(name: str) -> str:
    # Lightweight sanitizer for display/logging purposes
    name = name.strip().replace("\\", "/").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

def human_readable_prob(p: float) -> str:
    return f"{p:.2f}"
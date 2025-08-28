from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torchvision.models as models


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _unwrap_state(raw: Any) -> Dict[str, torch.Tensor]:
    """
    Accepts many common checkpoint formats and returns a flat state_dict:
      - raw is already a state dict
      - {"state_dict": ...}, {"model_state_dict": ...}, {"model_state": ...}
      - lightning-type nested dicts
    """
    if isinstance(raw, dict):
        # Try the common keys
        for k in ("state_dict", "model_state_dict", "model_state", "net", "model"):
            if k in raw and isinstance(raw[k], dict):
                return raw[k]
        # Heuristic: if values look like tensors, assume it's already a state dict
        if any(isinstance(v, torch.Tensor) for v in raw.values()):
            return raw
    # If none match, raise a clear error
    raise ValueError("Unsupported checkpoint format: cannot locate a state_dict")


def _strip_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state.keys()):
        return state
    return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state.items() }


def _infer_out_features(state: Dict[str, torch.Tensor], default_out: int) -> int:
    """If checkpoint contains fc.weight, use its out_features; else default."""
    w = state.get("fc.weight")
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        return int(w.shape[0])
    return int(default_out)


def _safe_load(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """Load with strict=False and print a short diagnostic to stdout (visible in logs)."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[model_loader] Missing keys: {list(missing)[:8]}{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[model_loader] Unexpected keys: {list(unexpected)[:8]}{' ...' if len(unexpected) > 8 else ''}")


def _verify_forward(model: nn.Module, device: str) -> None:
    """Run a dummy forward; if it fails, raise the original error."""
    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, 224, 224, device=device)
        _ = model(x)  # will raise if head/backbone shapes are incompatible


def load_model_and_threshold(
    model_path: str | Path,
    thr_path: str | Path | None = None,
    num_classes: int = 2,
    map_location: str | torch.device | None = None,
) -> Tuple[nn.Module, float, str]:
    """
    Robustly load a ResNet50 + threshold from flexible checkpoints.

    Returns:
        model (eval mode), best_thr (float), device (str: 'cpu'/'cuda')
    """
    device = _get_device() if map_location is None else str(map_location)
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    raw = torch.load(model_path, map_location=device)
    state = _unwrap_state(raw)

    # Strip common prefixes (e.g., 'module.' from DataParallel)
    state = _strip_prefix(state, "module.")
    state = _strip_prefix(state, "model.")
    state = _strip_prefix(state, "network.")

    # Build a clean ResNet50 backbone
    model = models.resnet50(weights=None)

    # Decide output dim from checkpoint if possible; else use num_classes
    out_features = _infer_out_features(state, num_classes)
    model.fc = nn.Linear(model.fc.in_features, out_features)

    # Try to load weights
    _safe_load(model, state)

    # First verification attempt
    try:
        _verify_forward(model, device)
    except Exception as e:
        print(f"[model_loader] Forward check failed with loaded head ({e}). "
              f"Rebuilding classifier head and retrying...")

        # Rebuild a fresh head that matches the backbone output
        # Default to 2-way classifier if we couldn't infer
        fallback_out = 2 if out_features not in (1, 2) else out_features
        model.fc = nn.Linear(model.fc.in_features, fallback_out)

        # Don't load fc.* from state this time; reload backbone weights only
        backbone_only = {k: v for k, v in state.items() if not k.startswith("fc.")}
        _safe_load(model, backbone_only)

        # Second verification (let it raise if truly incompatible)
        _verify_forward(model, device)

    model.to(device)
    model.eval()

    # Threshold loading: support 'best_thr' or 'best_threshold' or 'threshold'
    best_thr = 0.5
    if thr_path:
        thr_path = Path(thr_path)
        if thr_path.exists():
            try:
                with open(thr_path, "r") as f:
                    d = json.load(f)
                best_thr = float(
                    d.get("best_thr", d.get("best_threshold", d.get("threshold", best_thr)))
                )
            except Exception as e:
                print(f"[model_loader] Could not read threshold file: {e}. Using default {best_thr}.")

    print(f"[model_loader] Loaded model @ {model_path.name} | "
          f"fc.in={model.fc.in_features}, fc.out={model.fc.out_features} | thr={best_thr} | device={device}")

    return model, best_thr, device

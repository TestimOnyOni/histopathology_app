import torch
import json
from backend.model_loader import load_model_and_threshold

def test_model_loader(tmp_path):
    # --- Create dummy model file ---
    model_path = tmp_path / "dummy_model.pth"
    dummy_state = {
        "fc.weight": torch.randn(1, 3 * 224 * 224),
        "fc.bias": torch.randn(1),
    }
    torch.save(dummy_state, model_path)

    # --- Create dummy threshold file ---
    thr_path = tmp_path / "thr.json"
    thr = 0.65
    thr_path.write_text(json.dumps({"best_thr": thr}))

    # --- Load using our function ---
    model, device, best_thr = load_model_and_threshold(
        model_path, thr_path, num_classes=2, map_location="cpu"
    )

    # --- Assertions ---
    assert model is not None
    assert hasattr(model, "fc")
    assert device == "cpu"
    assert best_thr == thr

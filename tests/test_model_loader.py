import os
import torch
import pytest
from backend.model_loader import load_model_and_threshold

def test_model_loader(tmp_path):
    # Create fake model file
    model_path = tmp_path / "dummy_model.pth"
    torch.save({}, model_path)  # save empty state dict

    # Create fake threshold file
    thr_path = tmp_path / "thr.json"
    thr_path.write_text('{"best_thr": 0.7}')

    model, device, thr = load_model_and_threshold(
        model_path, thr_path, num_classes=2, map_location="cpu"
    )

    assert model is not None
    assert device == "cpu"
    assert thr == 0.7
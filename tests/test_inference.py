import torch
from PIL import Image
from backend.inference import run_inference_on_patches

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        # Always predict class 1 with high probability
        return torch.tensor([[0.1, 0.9]])

def test_inference():
    img = Image.new("RGB", (224, 224), color="green")
    model = DummyModel()
    slide_pred, agg_prob, patch_probs = run_inference_on_patches(
        model, "cpu", [img], threshold=0.5
    )
    assert slide_pred == 1
    assert agg_prob > 0.5
    assert isinstance(patch_probs, list)
    assert patch_probs[0] > 0.5
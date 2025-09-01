import torch
from backend.inference import run_inference_on_patches

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3 * 224 * 224, 1)

    def forward(self, x):
        # Flatten and send through linear layer
        x = x.view(x.size(0), -1)
        return self.fc(x)

def test_inference_single_patch():
    # Create a dummy model + dummy input
    model = DummyModel()
    device = "cpu"
    patch = torch.randn(3, 224, 224)  # simulate one preprocessed patch

    slide_pred, agg_prob, patch_probs = run_inference_on_patches(
        model, device, [patch], threshold=0.5
    )

    # --- Assertions ---
    assert isinstance(slide_pred, int)
    assert isinstance(agg_prob, float)
    assert isinstance(patch_probs, list)
    assert len(patch_probs) == 1


def test_inference_multiple_patches():
    model = DummyModel()
    device = "cpu"
    patches = [torch.randn(3, 224, 224) for _ in range(5)]

    slide_pred, agg_prob, patch_probs = run_inference_on_patches(
        model, device, patches, threshold=0.5
    )

    # --- Assertions ---
    assert isinstance(slide_pred, int)
    assert isinstance(agg_prob, float)
    assert isinstance(patch_probs, list)
    assert len(patch_probs) == 5

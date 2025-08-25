import torch
import numpy as np
from typing import List, Tuple, Dict
from .aggregation import aggregate_probs

@torch.inference_mode()
def predict_patch_probs(model, device, batch_tensor: torch.Tensor) -> np.ndarray:
    """
    batch_tensor: [N, C, H, W]
    returns: np.ndarray of shape [N] with malignant probabilities
    """
    batch_tensor = batch_tensor.to(device, non_blocking=True)
    logits = model(batch_tensor)
    probs = torch.softmax(logits, dim=1)[:, 1]
    return probs.detach().cpu().numpy()

def run_single_image(model, device, tensor_1x: torch.Tensor) -> float:
    """
    tensor_1x: [1, C, H, W]
    returns malignant probability
    """
    return float(predict_patch_probs(model, device, tensor_1x)[0])

def run_zip_patches(
    model, device,
    tensor_list: List[torch.Tensor],
    agg_method="percentile",
    percentile=90,
    batch_size=32
) -> Tuple[float, Dict]:
    """
    tensor_list: list of [1,C,H,W] tensors
    returns: (aggregated_prob, details)
        details = {
            "patch_probs": np.ndarray [N],
            "min": float, "mean": float, "max": float, "n_patches": int
        }
    """
    if len(tensor_list) == 0:
        return None, {"patch_probs": np.array([]), "min": None, "mean": None, "max": None, "n_patches": 0}

    # Stack into [N,C,H,W]
    batch = torch.cat(tensor_list, dim=0)

    # Mini-batch inference to save memory
    all_probs = []
    for i in range(0, batch.size(0), batch_size):
        probs = predict_patch_probs(model, device, batch[i:i+batch_size])
        all_probs.append(probs)
    patch_probs = np.concatenate(all_probs, axis=0)

    agg = aggregate_probs(patch_probs, method=agg_method, percentile=percentile)
    details = {
        "patch_probs": patch_probs,
        "min": float(np.min(patch_probs)),
        "mean": float(np.mean(patch_probs)),
        "max": float(np.max(patch_probs)),
        "n_patches": int(patch_probs.shape[0]),
    }
    return agg, details
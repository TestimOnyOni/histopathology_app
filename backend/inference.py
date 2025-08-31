# backend/inference.py
from __future__ import annotations
from typing import List, Tuple
import torch

def run_inference_on_patches(
    model,
    device: str,
    patches: List[torch.Tensor],
    threshold: float = 0.5,
    batch_size: int = 32,
) -> Tuple[int, float, list]:
    """
    Inference over a list of CHW tensors.
    Supports 1-logit (sigmoid) and 2-logit (softmax) heads.
    Returns (slide_pred, agg_prob, patch_probs_list).
    """
    if not patches:
        return 0, 0.0, []

    model.eval()
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = torch.stack(patches[i : i + batch_size]).to(device)  # [B,3,H,W]
            logits = model(batch)  # [B,1] or [B,2]

            if logits.ndim != 2:
                logits = logits.view(logits.size(0), -1)

            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits).squeeze(1)        # [B]
            else:
                probs = torch.softmax(logits, dim=1)[:, 1]      # [B]

            all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs)  # [N]
    agg_prob = float(all_probs.mean().item())
    slide_pred = int(agg_prob >= float(threshold))

    return slide_pred, agg_prob, all_probs.tolist()

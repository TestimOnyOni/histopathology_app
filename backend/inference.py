import torch

@torch.no_grad()
def run_inference_on_patches(model, device, patches, threshold=0.5, batch_size=16):
    """
    Runs inference on a list of image tensors (patches).
    Returns:
        slide_pred (int): 0 or 1 (negative/positive)
        agg_prob (float): average probability across patches
        patch_probs (list of float): individual patch probabilities
    """
    model.eval()

    all_probs = []
    for i in range(0, len(patches), batch_size):
        batch = patches[i:i + batch_size]
        x = torch.stack(batch).to(device)  # (B, C, H, W)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1]  # probability of class 1
        all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs)
    agg_prob = all_probs.mean().item()
    slide_pred = int(agg_prob >= threshold)

    return slide_pred, agg_prob, all_probs.tolist()

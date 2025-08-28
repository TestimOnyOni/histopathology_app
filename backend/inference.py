import torch


def run_inference_on_patches(model, device, patches, threshold=0.5):
    """
    Run inference on extracted patches.
    Handles both binary (1 logit) and 2-class (2 logits) outputs.
    """
    all_probs = []

    for patch in patches:
        x = patch.unsqueeze(0).to(device)  # add batch dim

        with torch.no_grad():
            logits = model(x)

            # Case 1: binary classifier with 1 output neuron
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits).squeeze(1)

            # Case 2: 2-class classifier with softmax
            else:
                probs = torch.softmax(logits, dim=1)[:, 1]

        all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs)
    agg_prob = all_probs.mean().item()

    slide_pred = int(agg_prob >= threshold)

    return slide_pred, agg_prob, all_probs.tolist()

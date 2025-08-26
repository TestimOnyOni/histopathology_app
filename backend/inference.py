import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

# Preprocessing pipeline for each patch
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats
        std=[0.229, 0.224, 0.225]
    ),
])

def run_inference_on_patches(model, device, patches, threshold=0.5):
    """
    Run inference on list of patches and return:
    - slide_pred: 0 (benign) or 1 (malignant)
    - agg_prob: aggregated malignant probability
    - patch_probs: list of individual patch malignant probabilities
    """
    model.eval()
    patch_probs = []

    with torch.no_grad():
        for img in patches:
            x = transform(img).unsqueeze(0).to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            patch_probs.append(probs[0, 1].item())  # malignant prob

    patch_probs = np.array(patch_probs)
    agg_prob = patch_probs.mean() if len(patch_probs) > 0 else 0.0
    slide_pred = int(agg_prob >= threshold)

    return slide_pred, agg_prob, patch_probs.tolist()
import numpy as np

def aggregate_probs(probs, method="percentile", percentile=90):
    """
    probs: list/np.ndarray of float probabilities (malignant class).
    """
    if len(probs) == 0:
        return None

    arr = np.array(probs, dtype=float)
    if method == "percentile":
        return float(np.percentile(arr, percentile))
    elif method == "mean":
        return float(np.mean(arr))
    elif method == "max":
        return float(np.max(arr))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
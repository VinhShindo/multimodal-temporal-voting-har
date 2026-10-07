"""Temporal smoothing probs."""
import numpy as np


def smooth_probs(probs_seq, kernel=5):
    """probs_seq (N, C) → (N, C) moving average."""
    if kernel <= 1 or len(probs_seq) < kernel:
        return probs_seq
    N = probs_seq.shape[0]
    out = np.zeros_like(probs_seq)
    half = kernel // 2
    for i in range(N):
        lo = max(0, i - half)
        hi = min(N, i + half + 1)
        out[i] = probs_seq[lo:hi].mean(axis=0)
    return out
"""
3 phương pháp fusion voting:
  - hard_majority:       argmax mỗi model → đếm phiếu
  - average_probability: trung bình cộng probs
  - weighted_probability: trung bình có trọng số
"""
from collections import Counter

import numpy as np


# ============================================================
# Helpers
# ============================================================
def _get_N(probs_dict: dict) -> int:
    return next(iter(probs_dict.values())).shape[0]


def _get_C(probs_dict: dict) -> int:
    return next(iter(probs_dict.values())).shape[1]


# ============================================================
# Hard majority
# ============================================================
def hard_majority(probs_dict: dict) -> np.ndarray:
    """Mỗi model argmax, rồi vote. Trả về (N,) pred."""
    N = _get_N(probs_dict)
    mods = list(probs_dict.keys())

    # argmax cho từng model
    hard = {m: probs_dict[m].argmax(axis=-1) for m in mods}

    fused = np.zeros(N, dtype=np.int64)
    for i in range(N):
        votes = [int(hard[m][i]) for m in mods]
        # Tie-break: chọn class có tổng prob cao nhất (fallback)
        counts = Counter(votes)
        top_count = counts.most_common(1)[0][1]
        candidates = [c for c, v in counts.items() if v == top_count]
        if len(candidates) == 1:
            fused[i] = candidates[0]
        else:
            # Tie: chọn class có sum prob cao nhất trong candidates
            score = {}
            for c in candidates:
                score[c] = sum(probs_dict[m][i, c] for m in mods)
            fused[i] = max(score, key=score.get)
    return fused


# ============================================================
# Average probability
# ============================================================
def average_probability(probs_dict: dict) -> np.ndarray:
    """Trung bình cộng probs → (N, C)."""
    stack = np.stack(list(probs_dict.values()), axis=0)  # (M, N, C)
    return stack.mean(axis=0)


# ============================================================
# Weighted probability
# ============================================================
def weighted_probability(probs_dict: dict, weights: dict) -> np.ndarray:
    """Trung bình có trọng số. weights = {"skeleton": 0.4, ...}."""
    mods = list(probs_dict.keys())
    total_w = sum(weights.get(m, 0.0) for m in mods)
    if total_w <= 0:
        raise ValueError("Tổng weight phải > 0")

    N = _get_N(probs_dict)
    C = _get_C(probs_dict)
    out = np.zeros((N, C), dtype=np.float32)
    for m in mods:
        w = weights.get(m, 0.0) / total_w
        out += w * probs_dict[m]
    return out


# ============================================================
# Unified API
# ============================================================
def fuse_probs(probs_dict: dict, method: str,
               weights: dict = None) -> tuple:
    """
    Return: (fused_probs (N, C), fused_preds (N,)).
    """
    method = method.lower()

    if method == "hard":
        preds = hard_majority(probs_dict)
        C = _get_C(probs_dict)
        probs = np.eye(C, dtype=np.float32)[preds]

    elif method == "average":
        probs = average_probability(probs_dict)
        preds = probs.argmax(axis=-1)

    elif method == "weighted":
        if weights is None:
            raise ValueError("Weighted voting cần weights dict")
        probs = weighted_probability(probs_dict, weights)
        preds = probs.argmax(axis=-1)

    else:
        raise ValueError(f"Unknown method: {method}")

    return probs, preds
"""3 kiểu voting: hard / average / weighted."""
from typing import List, Optional
import numpy as np

from src.common.schema import Prediction
from src.common.classes import load_classes


def hard_majority(preds: List[Optional[Prediction]], cmap) -> int:
    votes = {}
    for p in preds:
        if p is None:
            continue
        votes[p.class_id] = votes.get(p.class_id, 0) + 1
    if not votes:
        return 0
    return max(votes, key=votes.get)


def average_probability(preds: List[Optional[Prediction]], num_classes: int) -> np.ndarray:
    valid = [np.array(p.probabilities) for p in preds if p is not None]
    if not valid:
        return np.ones(num_classes) / num_classes
    return np.mean(valid, axis=0)


def weighted_probability(preds: List[Optional[Prediction]],
                         weights: dict,
                         num_classes: int) -> np.ndarray:
    """weights: {"rgb": w1, "skeleton": w2, "imu": w3}, tổng = 1."""
    total = np.zeros(num_classes)
    for key, p in zip(["rgb", "skeleton", "imu"], preds):
        if p is None:
            continue
        total += weights[key] * np.array(p.probabilities)
    return total


def fuse(record, method: str, cmap, weights: dict = None) -> Prediction:
    preds = [record.rgb, record.skeleton, record.imu]
    N = cmap.num_classes

    if method == "hard":
        cid = hard_majority(preds, cmap)
        probs = np.zeros(N)
        probs[cid] = 1.0
    elif method == "average":
        probs = average_probability(preds, N)
        cid = int(probs.argmax())
    elif method == "weighted":
        probs = weighted_probability(preds, weights, N)
        cid = int(probs.argmax())
    else:
        raise ValueError(f"Unknown method: {method}")

    return Prediction(
        window_id=record.window_id, start=record.start, end=record.end,
        class_id=cid, class_name=cmap.to_name(cid),
        confidence=float(probs[cid]),
        probabilities=probs.tolist(),
    )

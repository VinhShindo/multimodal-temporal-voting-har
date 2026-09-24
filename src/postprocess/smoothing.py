"""Smoothing theo temporal neighborhood."""
import numpy as np
from typing import List

from src.common.schema import Prediction


def smooth(preds: List[Prediction], kernel: int = 5) -> List[Prediction]:
    out = []
    for i in range(len(preds)):
        lo = max(0, i - kernel // 2)
        hi = min(len(preds), i + kernel // 2 + 1)
        probs = np.mean([preds[j].probabilities for j in range(lo, hi)], axis=0)
        cid = int(probs.argmax())
        p = preds[i]
        out.append(Prediction(
            window_id=p.window_id, start=p.start, end=p.end,
            class_id=cid, class_name=p.class_name,
            confidence=float(probs[cid]), probabilities=probs.tolist(),
        ))
    return out

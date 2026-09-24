"""Temporal voting trên buffer N window."""
from collections import Counter
from typing import List

from src.common.schema import Prediction


def temporal_vote(preds: List[Prediction], buffer: int = 5) -> List[Prediction]:
    """Trượt buffer, chọn class nhiều vote nhất cho từng vị trí."""
    out = []
    for i in range(len(preds)):
        lo = max(0, i - buffer // 2)
        hi = min(len(preds), i + buffer // 2 + 1)
        window = preds[lo:hi]
        counter = Counter(p.class_id for p in window)
        cid, _ = counter.most_common(1)[0]
        p = preds[i]
        out.append(Prediction(
            window_id=p.window_id, start=p.start, end=p.end,
            class_id=cid, class_name=p.class_name,
            confidence=p.confidence, probabilities=p.probabilities,
        ))
    return out

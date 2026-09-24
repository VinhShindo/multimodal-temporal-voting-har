"""★ Metric cho continuous recognition."""
import numpy as np
from typing import List

from src.common.schema import Segment


def frame_wise_accuracy(pred_segments: List[Segment],
                        gt_segments: List[Segment],
                        fps: float = 30.0) -> float:
    """Đếm theo từng frame."""
    t_end = max(
        max((s.end for s in pred_segments), default=0),
        max((s.end for s in gt_segments), default=0),
    )
    n = int(t_end * fps) + 1
    pred_labels = np.full(n, -1, dtype=int)
    gt_labels = np.full(n, -1, dtype=int)

    for s in pred_segments:
        i0, i1 = int(s.start * fps), int(s.end * fps)
        pred_labels[i0:i1] = s.class_id
    for s in gt_segments:
        i0, i1 = int(s.start * fps), int(s.end * fps)
        gt_labels[i0:i1] = s.class_id

    return float((pred_labels == gt_labels).mean())


def temporal_iou(pred: Segment, gt: Segment) -> float:
    inter = max(0.0, min(pred.end, gt.end) - max(pred.start, gt.start))
    union = max(pred.end, gt.end) - min(pred.start, gt.start)
    return inter / union if union > 0 else 0.0

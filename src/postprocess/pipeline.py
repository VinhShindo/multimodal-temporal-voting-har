"""Chạy smoothing → linking → removing."""
from typing import List

from src.common.schema import Prediction, Segment
from src.postprocess.smoothing import smooth
from src.postprocess.linking import link_segments
from src.postprocess.removing import remove_short


def predictions_to_segments(preds: List[Prediction]) -> List[Segment]:
    segments = []
    current = None
    for p in preds:
        if current is None or current.class_id != p.class_id:
            if current is not None:
                segments.append(current)
            current = Segment(
                start=p.start, end=p.end,
                class_id=p.class_id, class_name=p.class_name,
                confidence=p.confidence,
            )
        else:
            current.end = p.end
    if current is not None:
        segments.append(current)
    return segments


def run_postprocess(preds: List[Prediction],
                    smooth_kernel: int = 5,
                    t_avg: float = 0.8,
                    min_duration: float = 0.3) -> List[Segment]:
    preds = smooth(preds, smooth_kernel)
    segs = predictions_to_segments(preds)
    segs = link_segments(segs, t_avg)
    segs = remove_short(segs, min_duration)
    return segs

"""
★ Đưa 3 Prediction về cùng temporal window để fusion.
30fps camera ↔ 100Hz IMU → mọi thứ quy về window chung.
"""
from typing import List
from src.common.schema import Prediction, FusionRecord, Window


def align_predictions(rgb_preds: List[Prediction],
                      sk_preds: List[Prediction],
                      imu_preds: List[Prediction],
                      windows: List[Window]) -> List[FusionRecord]:
    """Mỗi window chọn prediction gần nhất theo tâm thời gian."""
    records = []
    for w in windows:
        center = (w.start + w.end) / 2
        rec = FusionRecord(window_id=w.window_id, start=w.start, end=w.end)
        rec.rgb = _nearest(rgb_preds, center)
        rec.skeleton = _nearest(sk_preds, center)
        rec.imu = _nearest(imu_preds, center)
        records.append(rec)
    return records


def _nearest(preds: List[Prediction], t: float):
    if not preds:
        return None
    return min(preds, key=lambda p: abs((p.start + p.end) / 2 - t))

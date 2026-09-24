"""Nối 2 segment cùng nhãn nếu gap < t_avg."""
from typing import List
from src.common.schema import Segment


def link_segments(segments: List[Segment], t_avg: float = 0.8) -> List[Segment]:
    if not segments:
        return []
    segments = sorted(segments, key=lambda s: s.start)
    merged = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        if seg.class_id == last.class_id and (seg.start - last.end) <= t_avg:
            last.end = seg.end
        else:
            merged.append(seg)
    return merged

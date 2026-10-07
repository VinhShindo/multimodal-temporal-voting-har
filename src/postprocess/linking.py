"""Segment dataclass + link logic."""
from dataclasses import dataclass


@dataclass
class Segment:
    start: float
    end: float
    class_id: int
    class_name: str
    confidence: float = 1.0

    @property
    def duration(self):
        return self.end - self.start


def link_segments(segments, t_avg=0.8):
    """Nối 2 segment cùng nhãn nếu gap < t_avg."""
    if not segments: return []
    segs = sorted(segments, key=lambda s: s.start)
    merged = [segs[0]]
    for s in segs[1:]:
        last = merged[-1]
        if s.class_id == last.class_id and (s.start - last.end) <= t_avg:
            last.end = s.end
        else:
            merged.append(s)
    return merged
"""Loại segment ngắn hơn min_duration."""
from typing import List
from src.common.schema import Segment


def remove_short(segments: List[Segment], min_duration: float = 0.3) -> List[Segment]:
    return [s for s in segments if s.duration >= min_duration]

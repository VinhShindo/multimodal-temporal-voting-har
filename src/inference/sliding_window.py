"""Sinh sliding window từ continuous stream."""
from dataclasses import dataclass
from typing import List


@dataclass
class Window:
    window_id: int
    start: float
    end: float


def generate_windows(total_duration: float, size_sec: float, step_sec: float) -> List[Window]:
    windows = []
    t = 0.0
    wid = 0
    while t + size_sec <= total_duration + 1e-6:
        windows.append(Window(wid, t, t + size_sec))
        t += step_sec
        wid += 1
    return windows

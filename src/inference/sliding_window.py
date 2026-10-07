"""Sinh sliding windows từ video dài."""
from dataclasses import dataclass


@dataclass
class Window:
    window_id: int
    start: float
    end: float

    @property
    def duration(self):
        return self.end - self.start


def generate_windows(total_duration: float,
                     size_sec: float = 2.0,
                     step_sec: float = 0.5) -> list:
    if total_duration <= 0 or size_sec <= 0 or step_sec <= 0:
        return []
    windows = []
    t = 0.0
    wid = 0
    while t + size_sec <= total_duration + 1e-6:
        windows.append(Window(wid, t, t + size_sec))
        t += step_sec
        wid += 1
    # Tail: kéo dài window cuối đến hết video nếu cần
    if windows and windows[-1].end < total_duration - 1e-6:
        windows.append(Window(wid, total_duration - size_sec, total_duration))
    return windows
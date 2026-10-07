"""Loại segment quá ngắn."""


def remove_short(segments, min_duration=0.3):
    return [s for s in segments if s.duration >= min_duration]
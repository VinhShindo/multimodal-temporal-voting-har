"""Skeleton boundary detection dựa trên motion energy."""
import numpy as np


def compute_motion_energy(skel: np.ndarray) -> np.ndarray:
    """skel: (T, J, 3) → energy: (T,)"""
    diff = np.diff(skel, axis=0)
    return np.linalg.norm(diff, axis=-1).mean(axis=-1)


def detect_boundaries(skel: np.ndarray,
                      threshold: float = 0.15,
                      smooth_kernel: int = 5,
                      min_duration_sec: float = 0.5,
                      fps: float = 30.0) -> list:
    """Trả list (start_sec, end_sec)."""
    energy = compute_motion_energy(skel)
    # smooth
    k = np.ones(smooth_kernel) / smooth_kernel
    energy = np.convolve(energy, k, mode="same")
    active = energy > threshold
    # tìm đoạn liên tục
    segments = []
    i = 0
    while i < len(active):
        if active[i]:
            j = i
            while j < len(active) and active[j]:
                j += 1
            if (j - i) / fps >= min_duration_sec:
                segments.append((i / fps, j / fps))
            i = j
        else:
            i += 1
    return segments

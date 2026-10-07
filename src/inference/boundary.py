"""Boundary detection từ skeleton motion energy."""
import numpy as np


def compute_motion_energy(skel):
    """skel (T, 17, 3) → energy (T,)."""
    if skel.shape[0] < 2:
        return np.zeros(skel.shape[0], dtype=np.float32)
    diff = np.diff(skel[:, :, :2], axis=0)
    energy = np.linalg.norm(diff, axis=-1).mean(axis=-1)
    return np.concatenate([[0.0], energy]).astype(np.float32)


def smooth_energy(energy, kernel=5):
    if kernel <= 1 or len(energy) < kernel:
        return energy
    k = np.ones(kernel) / kernel
    return np.convolve(energy, k, mode="same")


def detect_boundaries(skel, fps=30.0, threshold=None,
                       smooth_kernel=5, min_duration_sec=0.3,
                       percentile=60.0):
    """skel (T, 17, 3) → list[(start_sec, end_sec)]."""
    if skel.shape[0] < 5: return []
    energy = smooth_energy(compute_motion_energy(skel), smooth_kernel)
    if threshold is None:
        threshold = np.percentile(energy, percentile)
    active = energy > threshold

    segments = []
    i, n = 0, len(active)
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]: j += 1
            if (j - i) / fps >= min_duration_sec:
                segments.append((i / fps, j / fps))
            i = j
        else:
            i += 1
    return segments


def merge_nearby(segments, max_gap_sec=0.3):
    if not segments: return []
    segs = sorted(segments)
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        if s - merged[-1][1] <= max_gap_sec:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [tuple(m) for m in merged]
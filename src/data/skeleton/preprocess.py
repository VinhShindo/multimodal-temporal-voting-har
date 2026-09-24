"""Skeleton preprocessing: root-center → scale-norm → interpolate."""
import numpy as np


def root_center(skel: np.ndarray, root_idx: int) -> np.ndarray:
    """skel: (T, J, 3). Trừ root khỏi mọi joint."""
    root = skel[:, root_idx:root_idx + 1, :]
    return skel - root


def scale_normalize(skel: np.ndarray, ref_joints: tuple) -> np.ndarray:
    """Chia theo khoảng cách giữa 2 joint tham chiếu (ví dụ 2 vai)."""
    a, b = ref_joints
    scale = np.linalg.norm(skel[:, a] - skel[:, b], axis=-1, keepdims=True)
    scale = np.clip(scale, 1e-6, None)
    return skel / scale[..., None]


def temporal_interpolate(skel: np.ndarray, target_len: int) -> np.ndarray:
    T = skel.shape[0]
    if T == target_len:
        return skel
    idx = np.linspace(0, T - 1, target_len)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, T - 1)
    w = (idx - lo)[:, None, None]
    return skel[lo] * (1 - w) + skel[hi] * w

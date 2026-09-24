"""Augmentation cho skeleton."""
import numpy as np


def random_rotate_z(skel: np.ndarray, max_deg: float = 15.0) -> np.ndarray:
    theta = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return skel @ R.T

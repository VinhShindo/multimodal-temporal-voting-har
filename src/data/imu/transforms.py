"""Augmentation cho IMU."""
import numpy as np


def add_gaussian_noise(sig: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    return sig + np.random.randn(*sig.shape) * sigma


def random_scale(sig: np.ndarray, lo: float = 0.9, hi: float = 1.1) -> np.ndarray:
    return sig * np.random.uniform(lo, hi)

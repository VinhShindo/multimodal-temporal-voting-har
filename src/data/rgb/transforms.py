"""Augmentation cho RGB."""
import numpy as np


def random_horizontal_flip(frames: np.ndarray, p: float = 0.5) -> np.ndarray:
    if np.random.rand() < p:
        return frames[:, :, ::-1, :].copy()
    return frames

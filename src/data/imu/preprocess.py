"""IMU preprocessing: remove invalid → resample → normalize."""
import numpy as np


def remove_invalid(sig: np.ndarray, ts: np.ndarray) -> tuple:
    mask = np.isfinite(sig).all(axis=1)
    return sig[mask], ts[mask]


def zscore_normalize(sig: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (sig - mu) / np.clip(sigma, 1e-6, None)


def compute_stats(sig: np.ndarray) -> tuple:
    """★ CHỈ gọi trên train set."""
    return sig.mean(axis=0), sig.std(axis=0)


def sliding_window(sig: np.ndarray, ts: np.ndarray, size: int, step: int):
    """Yield (start_idx, end_idx) cho mỗi cửa sổ."""
    for i in range(0, len(sig) - size + 1, step):
        yield i, i + size

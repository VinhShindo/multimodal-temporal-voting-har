"""RGB preprocessing: decode → resize → crop → normalize."""
import numpy as np


def temporal_sample(frames: np.ndarray, num_frames: int, stride: int) -> np.ndarray:
    """Lấy mẫu cách stride frame, đảm bảo đủ num_frames."""
    idx = np.arange(0, num_frames * stride, stride)
    idx = np.clip(idx, 0, len(frames) - 1)
    return frames[idx]


def normalize(frames: np.ndarray, mean: list, std: list) -> np.ndarray:
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 1, 3)
    return (frames.astype(np.float32) / 255.0 - mean) / std

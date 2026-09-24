"""Master timeline + đồng bộ timestamp giữa các modality."""
import numpy as np


def build_master_timeline(start: float, end: float, rate: float) -> np.ndarray:
    """Trả về mảng timestamp đều nhau từ start → end với sampling rate."""
    n = int(round((end - start) * rate)) + 1
    return np.linspace(start, end, n)


def align_to_master(src_ts: np.ndarray, master_ts: np.ndarray) -> np.ndarray:
    """Với mỗi timestamp trong master, tìm index gần nhất trong src."""
    idx = np.searchsorted(src_ts, master_ts)
    idx = np.clip(idx, 0, len(src_ts) - 1)
    left = np.clip(idx - 1, 0, len(src_ts) - 1)
    # chọn bên gần hơn
    choose_left = np.abs(src_ts[left] - master_ts) < np.abs(src_ts[idx] - master_ts)
    return np.where(choose_left, left, idx)


def resample_signal(ts: np.ndarray, sig: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    """Resample tín hiệu (T, C) về target_ts bằng nearest-neighbor."""
    idx = align_to_master(ts, target_ts)
    return sig[idx]

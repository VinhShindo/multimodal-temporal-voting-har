"""
★ Đồng bộ 3 modality theo timestamp.
Camera 30fps ↔ IMU 100Hz → resample về master timeline.
"""
import numpy as np

from src.common.timestamp import build_master_timeline, resample_signal


def synchronize(rgb_ts: np.ndarray,
                skel_ts: np.ndarray,
                imu_ts: np.ndarray,
                imu_sig: np.ndarray,
                target_rate: float = 30.0) -> dict:
    """Trả dict timestamp + tín hiệu đã align."""
    t0 = min(rgb_ts[0], skel_ts[0], imu_ts[0])
    t1 = max(rgb_ts[-1], skel_ts[-1], imu_ts[-1])
    master_ts = build_master_timeline(t0, t1, target_rate)

    imu_aligned = resample_signal(imu_ts, imu_sig, master_ts)

    return {
        "master_ts": master_ts,
        "imu_aligned": imu_aligned,
    }

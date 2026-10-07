"""Skeleton transforms (COCO 17 keypoints) — input đã được normalize bởi processor.

Quy ước: kps có shape (T, 17, 3) = (x, y, conf), đã hip-center + scale theo thân.
Mọi augmentation chạy SAU normalize (không normalize lại trong dataset).

Mức augmentation:
  "none"   : chỉ resample thời gian (dùng để sanity check overfit)
  "light"  : crop thời gian + flip + rotate/scale nhẹ
  "strong" : light + scale bất đẳng hướng + shear + joint noise + joint dropout
  "view"   : strong + x-squeeze (co trục x ngẫu nhiên, mô phỏng xoay thân quanh trục dọc
             → giả lập đổi góc camera). Dùng cho cross-view.
"""
import numpy as np


COCO_FLIP_PAIRS = [
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16),
]


# ============================================================
# Temporal
# ============================================================
def temporal_interpolate(kps: np.ndarray, target_len: int) -> np.ndarray:
    """Resample tuyến tính (T, V, C) → (target_len, V, C)."""
    T = kps.shape[0]
    if T == target_len:
        return kps
    idx = np.linspace(0, T - 1, target_len)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, T - 1)
    w = (idx - lo)[:, None, None]
    return kps[lo] * (1 - w) + kps[hi] * w


def temporal_sample(kps: np.ndarray, target_len: int = 64,
                    train: bool = False, min_ratio: float = 0.6):
    """
    Resample TOÀN BỘ chuỗi về target_len (val) hoặc một đoạn ngẫu nhiên
    chiếm [min_ratio, 1] độ dài chuỗi (train).

    Returns:
      clip: (target_len, V, C)
      step: số frame gốc tương ứng 1 bước sau resample (dùng để tính tốc độ thật)
    """
    T = kps.shape[0]
    s, w = 0, T
    if train and T > 8:
        w = max(8, int(round(T * np.random.uniform(min_ratio, 1.0))))
        w = min(w, T)
        s = np.random.randint(0, T - w + 1)
    seg = kps[s:s + w]
    clip = temporal_interpolate(seg, target_len)
    step = (w - 1) / (target_len - 1) if w > 1 else 1.0
    return clip.astype(np.float32), float(step)


# ============================================================
# Spatial
# ============================================================
def random_flip(kps: np.ndarray, p: float = 0.5) -> np.ndarray:
    if np.random.rand() > p:
        return kps
    out = kps.copy()
    out[:, :, 0] *= -1.0
    for l, r in COCO_FLIP_PAIRS:
        out[:, [l, r], :] = out[:, [r, l], :]
    return out


def random_affine_2d(kps: np.ndarray, max_rot: float = 10.0,
                     scale=(0.9, 1.1), aniso: float = 0.0,
                     shear: float = 0.0) -> np.ndarray:
    """
    Rotate + scale (có thể bất đẳng hướng) + shear, áp dụng cho (x, y).
    aniso: biên độ lệch tỉ lệ giữa trục x và y (vd 0.15 → ±15%)
    shear: hệ số shear tối đa (vd 0.2)
    """
    th = np.deg2rad(np.random.uniform(-max_rot, max_rot))
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)

    base = np.random.uniform(*scale)
    sx = base * (1 + np.random.uniform(-aniso, aniso))
    sy = base * (1 + np.random.uniform(-aniso, aniso))
    S = np.diag([sx, sy]).astype(np.float32)

    shx = np.random.uniform(-shear, shear)
    shy = np.random.uniform(-shear, shear)
    Sh = np.array([[1, shx], [shy, 1]], dtype=np.float32)

    M = R @ S @ Sh
    out = kps.copy()
    out[:, :, :2] = out[:, :, :2] @ M.T
    return out


def random_joint_noise(kps: np.ndarray, std: float = 0.01) -> np.ndarray:
    out = kps.copy()
    out[:, :, :2] += np.random.normal(0, std, size=out[:, :, :2].shape).astype(np.float32)
    return out


def random_joint_dropout(kps: np.ndarray, max_joints: int = 3,
                         min_span: int = 8) -> np.ndarray:
    """Giả lập occlusion: che 0..max_joints joint trong một đoạn thời gian."""
    out = kps.copy()
    T, V = out.shape[:2]
    for _ in range(np.random.randint(0, max_joints + 1)):
        v = np.random.randint(0, V)
        span = np.random.randint(min_span, max(min_span + 1, T // 2))
        s = np.random.randint(0, max(1, T - span + 1))
        out[s:s + span, v, :2] = 0.0
        out[s:s + span, v, 2] = 0.0
    return out


def random_x_squeeze(kps: np.ndarray, lo: float = 0.5, hi: float = 1.2,
                     p: float = 0.7) -> np.ndarray:
    """
    Co/dãn riêng trục x quanh gốc (hip) với hệ số s ~ U(lo, hi).
    Khi người xoay quanh trục dọc, bề rộng 2D (foreshortening) co lại ~ cos(yaw),
    còn chiều cao gần như giữ nguyên. Đây là xấp xỉ 2D của việc đổi góc camera.
    """
    if np.random.rand() > p:
        return kps
    s = np.random.uniform(lo, hi)
    out = kps.copy()
    out[:, :, 0] *= s
    return out


# ============================================================
# Entry points
# ============================================================
def train_transform(kps: np.ndarray, target_len: int = 64,
                    level: str = "light"):
    """Returns (clip (T, V, 3), step)."""
    if level == "none":
        return temporal_sample(kps, target_len, train=False)

    clip, step = temporal_sample(kps, target_len, train=True)
    clip = random_flip(clip, p=0.5)

    if level == "light":
        clip = random_affine_2d(clip, max_rot=10.0, scale=(0.9, 1.1))
    elif level in ("strong", "view"):
        if level == "view":
            clip = random_x_squeeze(clip, lo=0.5, hi=1.2, p=0.7)
        clip = random_affine_2d(clip, max_rot=15.0, scale=(0.85, 1.15),
                                aniso=0.15, shear=0.2)
        clip = random_joint_noise(clip, std=0.01)
        clip = random_joint_dropout(clip, max_joints=3)
    else:
        raise ValueError(f"Unknown aug level: {level}")
    return clip.astype(np.float32), step


def val_transform(kps: np.ndarray, target_len: int = 64):
    """Returns (clip (T, V, 3), step)."""
    return temporal_sample(kps, target_len, train=False)


def normalize_skeleton(kps: np.ndarray) -> np.ndarray:
    """DEPRECATED: dữ liệu đã được normalize bởi SkeletonProcessor. Giữ lại để code cũ không bị vỡ import."""
    return kps.astype(np.float32)
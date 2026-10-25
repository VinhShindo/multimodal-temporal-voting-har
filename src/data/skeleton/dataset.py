"""MMAct skeleton dataset.

Stream (chọn bằng `stream`):
  joint(2) | joint_bone(4) | joint_bone_motion(6)      ← như cũ (baseline)
  rich(11) = joint(2)+bone(2)+motion(2)+bone_motion(2)+angle(cos,sin)(2)+conf(1)
  dual(18) = rich(11) || upper(7), upper pad V 9→17 và gói chung 1 tensor (18, T, 17)
             upper(7) = local_xy(2)+bone(2)+motion(2)+conf(1), 9 khớp thân trên,
             toạ độ tính so với trung điểm 2 vai TỪNG frame (scale = thân, đã chuẩn hoá bởi processor).

Gói chung 1 tensor để full_evaluation / fusion không phải đổi chữ ký (x, y).
Dữ liệu NPY đã normalize bởi SkeletonProcessor — KHÔNG normalize lại ở đây.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import yaml

from src.data.skeleton.transforms import train_transform, val_transform
from src.common.manifest_paths import resolve_train_val_manifests


COCO_PARENT = [-1, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14]
UB_JOINTS = [0, 1, 2, 5, 6, 7, 8, 9, 10]
UB_PARENT = [-1, 0, 0, 0, 0, 3, 4, 5, 6]      # chỉ số cục bộ (xem ctrgcn.py)

IN_CHANNELS = {"joint": 2, "joint_bone": 4, "joint_bone_motion": 6,
               "rich": 11, "dual": 18}
NUM_V = 17
UB_V = len(UB_JOINTS)


def _parent_arrays(parent):
    idx = np.array([p if p >= 0 else 0 for p in parent])
    has = np.array([p >= 0 for p in parent])
    return idx, has


_COCO_P = _parent_arrays(COCO_PARENT)
_UB_P = _parent_arrays(UB_PARENT)


def compute_bone(x: np.ndarray, parent=None) -> np.ndarray:
    """x: (T, V, 2) → (T, V, 2) = joint - parent_joint (gốc = 0)."""
    idx, has = _UB_P if (parent is not None and len(parent) == UB_V) else _COCO_P
    bone = x - x[:, idx]
    bone[:, ~has] = 0.0
    return bone


def compute_motion(x: np.ndarray, step: float = 1.0) -> np.ndarray:
    """Vi phân thời gian; chia `step` để ra tốc độ trên frame gốc. motion[0] = motion[1]."""
    m = np.zeros_like(x)
    m[1:] = x[1:] - x[:-1]
    m[0] = m[1]
    return m / max(step, 1e-6)


def rich_features(kps: np.ndarray, step: float, speed_aware: bool = True) -> np.ndarray:
    """kps (T, 17, 3) → (T, 17, 11)."""
    s = step if speed_aware else 1.0
    joint = kps[:, :, :2]
    conf = kps[:, :, 2:3]
    bone = compute_bone(joint)
    motion = compute_motion(joint, s)
    bone_motion = compute_motion(bone, s)
    n = np.linalg.norm(bone, axis=-1, keepdims=True)
    ang = np.where(n > 1e-3, bone / np.maximum(n, 1e-6), 0.0)      # (cos, sin) của bone
    return np.concatenate([joint, bone, motion, bone_motion, ang, conf],
                          axis=-1).astype(np.float32)


def upper_features(kps: np.ndarray, step: float, speed_aware: bool = True) -> np.ndarray:
    """kps (T, 17, 3) → (T, 9, 7): hệ quy chiếu cục bộ = trung điểm 2 vai mỗi frame."""
    s = step if speed_aware else 1.0
    ub = kps[:, UB_JOINTS]                                          # (T, 9, 3)
    mid_sho = (kps[:, 5, :2] + kps[:, 6, :2]) / 2.0                 # (T, 2)
    xy = ub[:, :, :2] - mid_sho[:, None, :]
    bone = compute_bone(xy, UB_PARENT)
    motion = compute_motion(xy, s)
    conf = ub[:, :, 2:3]
    return np.concatenate([xy, bone, motion, conf], axis=-1).astype(np.float32)


def make_features(kps: np.ndarray, step: float, stream: str,
                  speed_aware: bool = True) -> np.ndarray:
    """Trả về (C, T, 17)."""
    if stream in ("joint", "joint_bone", "joint_bone_motion"):
        joint = kps[:, :, :2].astype(np.float32)
        parts = [joint]
        if "bone" in stream:
            parts.append(compute_bone(joint))
        if "motion" in stream:
            parts.append(compute_motion(joint, step if speed_aware else 1.0))
        feat = np.concatenate(parts, axis=-1)                       # (T, 17, C)
        return np.ascontiguousarray(feat.transpose(2, 0, 1))

    g = rich_features(kps, step, speed_aware)                       # (T, 17, 11)
    g = g.transpose(2, 0, 1)                                        # (11, T, 17)
    if stream == "rich":
        return np.ascontiguousarray(g)
    if stream == "dual":
        u = upper_features(kps, step, speed_aware).transpose(2, 0, 1)   # (7, T, 9)
        u_pad = np.zeros((u.shape[0], u.shape[1], NUM_V), dtype=np.float32)
        u_pad[:, :, :UB_V] = u
        return np.ascontiguousarray(np.concatenate([g, u_pad], axis=0))  # (18, T, 17)
    raise ValueError(f"Unknown stream: {stream}")


class SkeletonDataset(Dataset):
    def __init__(self, manifest_path: str,
                 cfg_path: str = "configs/data/skeleton_mmact.yaml",
                 augment: bool = False,
                 stream: str = "joint_bone_motion",
                 aug_level: str = "light",
                 speed_aware_motion: bool = True,
                 cache: bool = True):
        if stream not in IN_CHANNELS:
            raise ValueError(f"Unknown stream: {stream}")
        with open(cfg_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        df = pd.read_csv(manifest_path)
        df = df[df["skeleton_path"].notna() & (df["skeleton_path"] != "")].copy()
        df["class_id"] = df["class_id"].astype(int)
        self.df = df.reset_index(drop=True)

        self.paths = self.df["skeleton_path"].tolist()
        self.labels = self.df["class_id"].values.astype(np.int64)

        self.target_len = self.cfg["sequence"]["target_len"]
        self.num_joints = self.cfg["layout"]["num_joints"]
        self.augment = augment
        self.stream = stream
        self.aug_level = aug_level
        self.speed_aware_motion = speed_aware_motion
        self.in_channels = IN_CHANNELS[stream]

        # Cache toàn bộ NPY vào RAM (nhỏ: vài trăm MB) → bỏ I/O mỗi lần __getitem__
        self._cache = None
        if cache:
            self._cache = [self._load(p) for p in self.paths]

    @staticmethod
    def _load(path):
        kps = np.load(path)
        return np.nan_to_num(kps, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        kps = self._cache[idx] if self._cache is not None else self._load(self.paths[idx])

        if self.augment:
            kps, step = train_transform(kps, self.target_len, self.aug_level)
        else:
            kps, step = val_transform(kps, self.target_len)

        feat = make_features(kps, step, self.stream, self.speed_aware_motion)   # (C, T, 17)
        return torch.from_numpy(feat).float(), int(self.labels[idx])


def build_loaders(manifest_dir: str | None, cfg_path: str,
                  batch_size: int, num_workers: int = 4,
                  stream: str = "joint_bone_motion",
                  aug_level: str = "light",
                  cache: bool = True):
    from torch.utils.data import DataLoader, WeightedRandomSampler

    train_manifest, val_manifest = resolve_train_val_manifests(
        cfg_path, manifest_dir)
    train_ds = SkeletonDataset(str(train_manifest),
                               cfg_path, augment=True, stream=stream,
                               aug_level=aug_level, cache=cache)
    val_ds = SkeletonDataset(str(val_manifest),
                             cfg_path, augment=False, stream=stream, cache=cache)

    labels = train_ds.labels
    num_classes = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=num_classes)
    weight_per_class = 1.0 / np.clip(counts, 1, None)
    weights = weight_per_class[labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds),
                                    replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=num_workers > 0)

    print(f"[Skeleton] stream={stream} aug={aug_level} "
          f"train={len(train_ds)} val={len(val_ds)} "
          f"| in_channels={train_ds.in_channels}")
    return train_loader, val_loader

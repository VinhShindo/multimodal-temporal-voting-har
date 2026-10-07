"""
MMAct RGB dataset — load video + sample T frames.
"""
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import yaml

try:
    import decord
    from decord import VideoReader, cpu
    HAS_DECORD = True
except ImportError:
    import cv2
    HAS_DECORD = False
    print("[RGB] decord không có — dùng cv2 (chậm hơn)")

MEAN = np.array([0.45, 0.45, 0.45], dtype=np.float32)
STD  = np.array([0.225, 0.225, 0.225], dtype=np.float32)


def sample_frame_indices(total: int, num: int, train: bool) -> np.ndarray:
    if total <= 0:
        return np.zeros(num, dtype=np.int64)
    if total < num:
        idx = np.arange(total)
        pad = np.full(num - total, total - 1)
        return np.concatenate([idx, pad])
    if train:
        seg = total / num
        idx = (np.arange(num) * seg + np.random.uniform(0, seg, num)).astype(int)
        return np.clip(idx, 0, total - 1)
    return np.linspace(0, total - 1, num).astype(int)


def resize_center_crop(frame: np.ndarray, size: int = 224,
                       crop_pct: float = 0.875) -> np.ndarray:
    import cv2
    H, W = frame.shape[:2]
    scale = size / crop_pct
    if H < W:
        new_h = int(scale); new_w = int(W * scale / H)
    else:
        new_w = int(scale); new_h = int(H * scale / W)
    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    y0 = (new_h - size) // 2
    x0 = (new_w - size) // 2
    return frame[y0:y0 + size, x0:x0 + size]


def read_video_decord(path: str, num_frames: int, train: bool) -> np.ndarray:
    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    idx = sample_frame_indices(total, num_frames, train)
    return vr.get_batch(idx).asnumpy()


def read_video_cv2(path: str, num_frames: int, train: bool) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = sample_frame_indices(total, num_frames, train)
    frames = []
    for target in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        elif frames:
            frames.append(frames[-1])
        else:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()
    return np.stack(frames, axis=0)


class RGBDataset(Dataset):
    def __init__(self, manifest_path: str,
                 cfg_path: str = "configs/data/rgb_mmact.yaml",
                 augment: bool = False,
                 max_samples: int = None):
        with open(cfg_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        df = pd.read_csv(manifest_path)
        df = df[df["rgb_path"].notna() & (df["rgb_path"] != "")].copy()
        df["class_id"] = df["class_id"].astype(int)

        # ★ Giới hạn samples để verify nhanh
        if max_samples is not None and max_samples < len(df):
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
            print(f"[RGB] Giới hạn {max_samples} samples (verify mode)")

        self.df = df.reset_index(drop=True)
        self.num_frames = self.cfg["sampling"]["num_frames"]
        self.size = self.cfg["transform"]["resize"]
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["rgb_path"]

        try:
            if HAS_DECORD:
                frames = read_video_decord(path, self.num_frames, self.augment)
            else:
                frames = read_video_cv2(path, self.num_frames, self.augment)
        except Exception as e:
            frames = np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)

        if self.augment and random.random() < 0.5:
            frames = frames[:, :, ::-1, :]

        processed = [resize_center_crop(f, size=self.size) for f in frames]
        arr = np.stack(processed, axis=0).astype(np.float32) / 255.0
        arr = (arr - MEAN) / STD

        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
        return tensor, int(row["class_id"])


def build_loaders(manifest_dir: str, cfg_path: str,
                  batch_size: int, num_workers: int = 4,
                  max_samples: int = None):
    from torch.utils.data import DataLoader, WeightedRandomSampler

    train_ds = RGBDataset(f"{manifest_dir}/train_cross_view.csv",
                          cfg_path, augment=True, max_samples=max_samples)
    val_ds   = RGBDataset(f"{manifest_dir}/val_cross_view.csv",
                          cfg_path, augment=False, max_samples=max_samples)

    # Sampler — nếu dataset quá nhỏ, skip WeightedSampler
    labels = train_ds.df["class_id"].values
    num_classes = int(labels.max()) + 1
    if len(labels) >= 20:  # đủ data để sample theo class
        counts = np.bincount(labels, minlength=num_classes)
        weight_per_class = 1.0 / np.clip(counts, 1, None)
        weights = weight_per_class[labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds),
                                        replacement=True)
    else:
        sampler = None  # shuffle thường

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=num_workers, pin_memory=True,
        drop_last=False)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    print(f"[RGB] train={len(train_ds)} val={len(val_ds)} "
          f"| frames={train_ds.num_frames} | size={train_ds.size}")
    return train_loader, val_loader
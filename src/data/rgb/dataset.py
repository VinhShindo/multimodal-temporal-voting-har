"""
MMAct RGB dataset — load video + sample T frames.
"""
import random
import subprocess
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import yaml

from src.common.manifest_paths import resolve_train_val_manifests

try:
    import decord
    from decord import VideoReader, cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

if not HAS_DECORD and not HAS_CV2:
    print("[RGB] decord/opencv không có — dùng ffmpeg fallback")

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
    H, W = frame.shape[:2]
    scale = size / crop_pct
    if H < W:
        new_h = int(scale); new_w = int(W * scale / H)
    else:
        new_w = int(scale); new_h = int(H * scale / W)
    if HAS_CV2:
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        from PIL import Image
        frame = np.asarray(Image.fromarray(frame).resize(
            (new_w, new_h), Image.Resampling.BILINEAR))
    y0 = (new_h - size) // 2
    x0 = (new_w - size) // 2
    return frame[y0:y0 + size, x0:x0 + size]


def read_video_decord(path: str, num_frames: int, train: bool) -> np.ndarray:
    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    idx = sample_frame_indices(total, num_frames, train)
    return vr.get_batch(idx).asnumpy()


def read_video_cv2(path: str, num_frames: int, train: bool) -> np.ndarray:
    if not HAS_CV2:
        raise RuntimeError("opencv is not installed")
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


def read_video_ffmpeg(path: str, num_frames: int, train: bool) -> np.ndarray:
    """Read sampled RGB frames without Python video bindings."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames",
         "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames,width,height",
         "-of", "default=noprint_wrappers=1", path],
        check=True, capture_output=True, text=True,
    )
    values = {}
    for line in probe.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    total = int(values.get("nb_read_frames", 0))
    width = int(values.get("width", 0))
    height = int(values.get("height", 0))
    if total <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid ffprobe metadata for {path}")
    indices = sample_frame_indices(total, num_frames, train)
    expression = "+".join(f"eq(n\\,{int(index)})" for index in indices)
    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path,
         "-vf", f"select={expression}", "-vsync", "0",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        check=True, capture_output=True,
    ).stdout
    frame_bytes = width * height * 3
    frames = np.frombuffer(decoded, dtype=np.uint8)
    if frames.size % frame_bytes:
        raise RuntimeError(f"Invalid ffmpeg frame output for {path}")
    frames = frames.reshape(-1, height, width, 3)
    if len(frames) == 0:
        raise RuntimeError(f"ffmpeg returned no frames for {path}")
    if len(frames) < num_frames:
        pad = np.repeat(frames[-1:], num_frames - len(frames), axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    return frames[:num_frames]


class RGBDataset(Dataset):
    def __init__(self, manifest_path: str,
                 cfg_path: str = "configs/data/rgb_mmact.yaml",
                 augment: bool = False,
                 max_samples: int = None,
                 cache_in_memory: bool = False,
                 cache_max_clips: int = 4):
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
        self.paths = []
        for _, row in self.df.iterrows():
            cache_path = row.get("rgb_cache_path", "")
            cache_path = "" if pd.isna(cache_path) else str(cache_path)
            video_path = str(row["rgb_path"])
            self.paths.append(cache_path if cache_path and Path(cache_path).is_file()
                              else video_path)
        self.num_frames = self.cfg["sampling"]["num_frames"]
        self.size = self.cfg["transform"]["resize"]
        self.augment = augment
        self._cache = None
        # Disk-backed NPY caches are opened with mmap and kept in a small
        # per-worker LRU. This avoids preloading all clips into RAM while
        # avoiding repeated open/close overhead for frequently sampled clips.
        self.cache_max_clips = max(0, int(cache_max_clips))
        self._mmap_cache = OrderedDict()
        if cache_in_memory:
            non_cache = [path for path in self.paths
                         if Path(path).suffix.lower() != ".npy"]
            if non_cache:
                raise RuntimeError(
                    f"RGB frame cache missing for {len(non_cache)} clips. "
                    "Run scripts/01_process_data.py --modality rgb "
                    "--rgb-mode extract --rgb-frames 16 --rgb-size 224 first."
                )
            self._cache = [np.load(path) for path in self.paths]
            cache_gib = sum(frames.nbytes for frames in self._cache) / 1024**3
            print(f"[RGB] Preloaded {len(self._cache)} clips into RAM "
                  f"({cache_gib:.1f} GiB)")
        elif self.cache_max_clips > 0:
            mmap_count = sum(Path(path).suffix.lower() == ".npy"
                             for path in self.paths)
            print(f"[RGB] Disk-backed cache: {mmap_count} clips, "
                  f"max {self.cache_max_clips} mmap handles per worker")

    def __len__(self):
        return len(self.df)

    def _get_cached_frames(self, idx: int, path: str):
        if self._cache is not None:
            return self._cache[idx]

        if Path(path).suffix.lower() != ".npy":
            return None

        if self.cache_max_clips <= 0:
            return np.load(path, mmap_mode="r")

        cached = self._mmap_cache.get(path)
        if cached is None:
            cached = np.load(path, mmap_mode="r")
            self._mmap_cache[path] = cached
            if len(self._mmap_cache) > self.cache_max_clips:
                self._mmap_cache.popitem(last=False)
        else:
            self._mmap_cache.move_to_end(path)
        return cached

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.paths[idx]

        try:
            if Path(path).suffix.lower() == ".npy":
                cached = self._get_cached_frames(idx, path)
                indices = sample_frame_indices(
                    len(cached), self.num_frames, self.augment)
                frames = np.asarray(cached[indices])
            elif HAS_DECORD:
                frames = read_video_decord(path, self.num_frames, self.augment)
            elif HAS_CV2:
                frames = read_video_cv2(path, self.num_frames, self.augment)
            else:
                frames = read_video_ffmpeg(path, self.num_frames, self.augment)
        except Exception as e:
            frames = np.zeros((self.num_frames, self.size, self.size, 3),
                              dtype=np.uint8)

        if self.augment and random.random() < 0.5:
            frames = frames[:, :, ::-1, :]

        processed = [f if f.shape[:2] == (self.size, self.size)
                 else resize_center_crop(f, size=self.size) for f in frames]
        arr = np.stack(processed, axis=0).astype(np.float32) / 255.0
        arr = (arr - MEAN) / STD

        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
        return tensor, int(row["class_id"])


def build_loaders(manifest_dir: str | None, cfg_path: str,
                  batch_size: int, num_workers: int = 4,
                  max_samples: int = None,
                  cache_in_memory: bool = False,
                  cache_max_clips: int = 4,
                  prefetch_factor: int = 1,
                  persistent_workers: bool = True):
    from torch.utils.data import DataLoader, WeightedRandomSampler

    train_manifest, val_manifest = resolve_train_val_manifests(
        cfg_path, manifest_dir)
    train_ds = RGBDataset(str(train_manifest),
                          cfg_path, augment=True, max_samples=max_samples,
                          cache_in_memory=cache_in_memory,
                          cache_max_clips=cache_max_clips)
    val_ds   = RGBDataset(str(val_manifest),
                          cfg_path, augment=False, max_samples=max_samples,
                          cache_in_memory=cache_in_memory,
                          cache_max_clips=cache_max_clips)

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

    loader_options = {
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": False,
    }
    if num_workers > 0:
        loader_options["persistent_workers"] = bool(persistent_workers)
        loader_options["prefetch_factor"] = max(1, int(prefetch_factor))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, shuffle=(sampler is None),
        **loader_options)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        **loader_options)

    print(f"[RGB] train={len(train_ds)} val={len(val_ds)} "
          f"| frames={train_ds.num_frames} | size={train_ds.size}")
    return train_loader, val_loader

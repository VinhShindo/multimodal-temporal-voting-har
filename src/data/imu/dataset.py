"""MMAct IMU dataset."""
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml
import pandas as pd

from src.data.imu.transforms import add_gaussian_noise, random_scale
from src.common.manifest_paths import resolve_train_val_manifests


IMU_EVENT_COLUMNS = ["subject", "scene", "session", "action"]
DERIVED_FEATURE_CHANNELS = 10


def transform_imu_features(sig: np.ndarray, feature_set: str,
                           sampling_rate: float = 50.0) -> np.ndarray:
    if feature_set == "raw":
        return sig.astype(np.float32, copy=False)

    if feature_set not in {"gravity_jerk", "gravity_jerk_plus"}:
        raise ValueError(f"Unsupported IMU feature set: {feature_set}")

    gravity_window = max(3, int(round(sampling_rate * 0.5)) | 1)
    pad = gravity_window // 2
    padded = np.pad(sig, ((pad, pad), (0, 0)), mode="edge")
    cumulative = np.vstack((np.zeros((1, sig.shape[1]), dtype=np.float64),
                            np.cumsum(padded, axis=0, dtype=np.float64)))
    gravity = ((cumulative[gravity_window:] - cumulative[:-gravity_window])
               / gravity_window).astype(np.float32)
    dynamic = sig - gravity
    magnitude = np.linalg.norm(sig, axis=1, keepdims=True)
    jerk = np.gradient(sig, 1.0 / sampling_rate, axis=0)
    base = np.concatenate((sig, magnitude, dynamic, jerk), axis=1).astype(np.float32)

    if feature_set == "gravity_jerk":
        return base

    delta = np.concatenate([
        sig[1:] - sig[:-1],
        np.zeros((1, sig.shape[1]), dtype=np.float32),
    ], axis=0)
    signal_strength = np.abs(sig) + np.abs(delta)
    return np.concatenate((base, delta, signal_strength), axis=1).astype(np.float32)


def _load_imu_manifest(manifest_path: str,
                       exclude_events: set[tuple] | None = None) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    df = df[df["imu_path"].notna() & (df["imu_path"] != "")].copy()
    df["class_id"] = df["class_id"].astype(int)
    df = df.drop_duplicates(subset=IMU_EVENT_COLUMNS)
    if exclude_events:
        event_rows = map(tuple, df[IMU_EVENT_COLUMNS].to_numpy())
        keep = [event not in exclude_events for event in event_rows]
        df = df.loc[keep]
    return df.reset_index(drop=True)


class IMUDataset(Dataset):
    def __init__(self, manifest_path: str, cfg_path: str = "configs/data/imu_mmact.yaml",
                 stats: dict = None, training: bool = False,
                 feature_set: str = "raw",
                 exclude_events: set[tuple] | None = None):
        with open(cfg_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.df = _load_imu_manifest(manifest_path, exclude_events)
        self.window_size = self.cfg["window"]["size"]
        self.stats = stats
        self.training = training
        self.num_eval_crops = int(self.cfg["window"].get("eval_crops", 1))
        self.feature_set = feature_set
        self.sampling_rate = float(self.cfg["sensor"]["sampling_rate"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sig = np.load(row["imu_path"])                 # (T, 3)
        sig = transform_imu_features(sig, self.feature_set, self.sampling_rate)

        if sig.shape[0] < self.window_size:
            pad = np.zeros((self.window_size - sig.shape[0], sig.shape[1]),
                           dtype=np.float32)
            sig = np.concatenate([sig, pad], axis=0)

        if self.stats is not None:
            sig = (sig - self.stats["mu"]) / np.clip(self.stats["std"], 1e-6, None)

        max_start = sig.shape[0] - self.window_size
        if self.training:
            starts = [np.random.randint(max_start + 1)]
        else:
            starts = np.linspace(0, max_start, self.num_eval_crops).round().astype(int)

        crops = np.stack([
            sig[start:start + self.window_size] for start in starts
        ])
        if self.training:
            aug = self.cfg.get("augmentation", {})
            if aug.get("scale_range"):
                crops = np.stack([
                    random_scale(crop, *aug["scale_range"]) for crop in crops
                ])
            noise_std = float(aug.get("gaussian_noise_std", 0.0))
            if noise_std > 0:
                crops = np.stack([
                    add_gaussian_noise(crop, noise_std) for crop in crops
                ])

        tensor = torch.from_numpy(crops.astype(np.float32)).permute(0, 2, 1)
        if self.training:
            tensor = tensor[0]  # (C, T)
        label = int(row["class_id"])
        return tensor, label


def compute_stats(manifest_path: str, feature_set: str = "raw",
                  sampling_rate: float = 50.0,
                  exclude_events: set[tuple] | None = None) -> dict:
    """Tính mean/std trên train set — dùng cho zscore."""
    df = _load_imu_manifest(manifest_path, exclude_events)
    if df.empty:
        raise ValueError(f"No IMU training events remain in {manifest_path}")

    # Sample 200 clip để tính nhanh
    sample = df.sample(n=min(200, len(df)), random_state=42)
    sigs = [transform_imu_features(np.load(p), feature_set, sampling_rate)
            for p in sample["imu_path"].tolist()]
    all_sig = np.concatenate(sigs, axis=0)

    return {"mu": all_sig.mean(axis=0).astype(np.float32),
            "std": all_sig.std(axis=0).astype(np.float32)}


def build_loaders(manifest_dir: str | None, cfg_path: str, batch_size: int,
                  num_workers: int = 4, protocol: str | None = None,
                  feature_set: str = "raw"):
    from torch.utils.data import DataLoader, WeightedRandomSampler

    train_manifest_path, val_manifest_path = resolve_train_val_manifests(
        cfg_path, manifest_dir, protocol)
    train_manifest = str(train_manifest_path)
    val_manifest = str(val_manifest_path)
    with open(cfg_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    sampling_rate = float(data_cfg["sensor"]["sampling_rate"])
    train_manifest_df = _load_imu_manifest(train_manifest)
    val_manifest_df = _load_imu_manifest(val_manifest)
    train_events = set(map(tuple, train_manifest_df[IMU_EVENT_COLUMNS].to_numpy()))
    val_events = set(map(tuple, val_manifest_df[IMU_EVENT_COLUMNS].to_numpy()))
    event_overlap = train_events & val_events
    if event_overlap:
        raise ValueError(
            f"IMU manifest leakage: {len(event_overlap)} physical events overlap "
            f"between {protocol or 'configured'} train and val; "
            f"example={next(iter(event_overlap))}"
        )

    stats = compute_stats(train_manifest, feature_set, sampling_rate)
    print(f"[IMU] stats: mu={stats['mu']}, std={stats['std']}")

    train_ds = IMUDataset(train_manifest, cfg_path, stats, training=True,
                          feature_set=feature_set)
    val_ds = IMUDataset(val_manifest, cfg_path, stats, training=False,
                        feature_set=feature_set)

    train_events = set(map(tuple, train_ds.df[IMU_EVENT_COLUMNS].to_numpy()))
    val_events = set(map(tuple, val_ds.df[IMU_EVENT_COLUMNS].to_numpy()))
    overlap = train_events & val_events
    if overlap:
        example = next(iter(overlap))
        raise ValueError(
            f"IMU leakage: {len(overlap)} physical recordings occur in both "
            f"{split} train and val; example={example}. Use a group-disjoint split."
        )
    if not len(train_ds) or not len(val_ds):
        raise ValueError(f"Empty IMU dataset for protocol={protocol}")

    print(f"[IMU] protocol={protocol or 'configured'} "
          f"unique train={len(train_ds)} val={len(val_ds)}")

    labels = train_ds.df["class_id"].values
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    weight_per_class = 1.0 / np.clip(counts, 1, None)
    weights = weight_per_class[labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)
    return train_loader, val_loader

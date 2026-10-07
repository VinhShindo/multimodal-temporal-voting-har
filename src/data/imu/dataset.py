"""MMAct IMU dataset."""
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml
import pandas as pd


class IMUDataset(Dataset):
    def __init__(self, manifest_path: str, cfg_path: str = "configs/data/imu_mmact.yaml",
                 stats: dict = None):
        with open(cfg_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        df = pd.read_csv(manifest_path)
        df = df[df["imu_path"].notna() & (df["imu_path"] != "")].copy()
        df["class_id"] = df["class_id"].astype(int)
        self.df = df.reset_index(drop=True)
        self.window_size = self.cfg["window"]["size"]
        self.stats = stats

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sig = np.load(row["imu_path"])                 # (T, 3)

        # Pad hoặc crop về window_size
        T = sig.shape[0]
        if T < self.window_size:
            pad = np.zeros((self.window_size - T, sig.shape[1]), dtype=np.float32)
            sig = np.concatenate([sig, pad], axis=0)
        else:
            # Lấy center crop
            start = (T - self.window_size) // 2
            sig = sig[start:start + self.window_size]

        # Normalize zscore
        if self.stats is not None:
            sig = (sig - self.stats["mu"]) / np.clip(self.stats["std"], 1e-6, None)

        tensor = torch.from_numpy(sig).float().permute(1, 0)  # (C, T)
        label = int(row["class_id"])
        return tensor, label


def compute_stats(manifest_path: str) -> dict:
    """Tính mean/std trên train set — dùng cho zscore."""
    df = pd.read_csv(manifest_path)
    df = df[df["imu_path"].notna() & (df["imu_path"] != "")].copy()

    # Sample 200 clip để tính nhanh
    sample = df.sample(n=min(200, len(df)), random_state=42)
    sigs = [np.load(p) for p in sample["imu_path"].tolist()]
    all_sig = np.concatenate(sigs, axis=0)

    return {"mu": all_sig.mean(axis=0).astype(np.float32),
            "std": all_sig.std(axis=0).astype(np.float32)}


def build_loaders(manifest_dir: str, cfg_path: str, batch_size: int,
                  num_workers: int = 4):
    from torch.utils.data import DataLoader, WeightedRandomSampler
    stats = compute_stats(f"{manifest_dir}/train_cross_view.csv")
    print(f"[IMU] stats: mu={stats['mu']}, std={stats['std']}")

    train_ds = IMUDataset(f"{manifest_dir}/train_cross_view.csv", cfg_path, stats)
    val_ds   = IMUDataset(f"{manifest_dir}/val_cross_view.csv",   cfg_path, stats)

    labels = train_ds.df["class_id"].values
    counts = np.bincount(labels, minlength=32)
    weight_per_class = 1.0 / np.clip(counts, 1, None)
    weights = weight_per_class[labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
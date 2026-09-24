"""IMU dataset."""
import numpy as np
import torch
from torch.utils.data import Dataset


class IMUDataset(Dataset):
    def __init__(self, manifest, window_size=200, num_channels=6):
        self.manifest = manifest
        self.window_size = window_size
        self.num_channels = num_channels

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        sig = np.zeros((self.window_size, self.num_channels), dtype=np.float32)
        # TODO: load thật từ data/processed/imu
        tensor = torch.from_numpy(sig).permute(1, 0)  # C,T
        return tensor, int(row["class_id"]), {"subject": row["subject"]}

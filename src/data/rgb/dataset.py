"""RGB dataset cho isolated training."""
import numpy as np
import torch
from torch.utils.data import Dataset


class RGBDataset(Dataset):
    def __init__(self, manifest, num_frames=16, stride=4, transform=None):
        self.manifest = manifest
        self.num_frames = num_frames
        self.stride = stride
        self.transform = transform

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        # TODO: decode video segment [start, end], sample frames
        frames = np.zeros((self.num_frames, 224, 224, 3), dtype=np.float32)
        if self.transform:
            frames = self.transform(frames)
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)  # T,C,H,W
        return tensor, int(row["class_id"]), {"subject": row["subject"]}

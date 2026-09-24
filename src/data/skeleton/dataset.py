"""Skeleton dataset."""
import numpy as np
import torch
from torch.utils.data import Dataset


class SkeletonDataset(Dataset):
    def __init__(self, manifest, seq_len=64, num_joints=48):
        self.manifest = manifest
        self.seq_len = seq_len
        self.num_joints = num_joints

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        skel = np.zeros((self.seq_len, self.num_joints, 3), dtype=np.float32)
        # TODO: load thật từ data/processed/skeleton
        tensor = torch.from_numpy(skel).permute(2, 0, 1)  # C,T,V
        return tensor, int(row["class_id"]), {"subject": row["subject"]}

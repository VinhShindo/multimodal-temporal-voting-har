"""Compatibility layer for paper-style skeleton architectures.

This file preserves the previous benchmark interface while standardizing each architecture
into its own dedicated Python module under src/models/skeleton/.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.skeleton.barlow_twins import BarlowTwinsSkeleton
from src.models.skeleton.ms_g3d import MSG3D
from src.models.skeleton.sta_gcn import STAGCN
from src.models.skeleton.stgcn import STGCN


class SkeletonGraphClassifier(nn.Module):
    """Factory wrapper for the paper-inspired graph families."""

    def __init__(self, in_channels: int = 6, num_classes: int = 35,
                 target_len: int = 64, variant: str = "stgcn",
                 dropout: float = 0.3):
        super().__init__()
        variant = variant.lower()
        if variant == "stgcn":
            self.model = STGCN(num_classes=num_classes, num_joints=17,
                              in_channels=in_channels, dropout=dropout)
        elif variant == "ms_g3d":
            self.model = MSG3D(num_classes=num_classes, num_joints=17,
                              in_channels=in_channels, dropout=dropout)
        elif variant == "sta_gcn":
            self.model = STAGCN(num_classes=num_classes, num_joints=17,
                               in_channels=in_channels, dropout=dropout)
        else:
            raise ValueError(f"Unsupported paper graph variant: {variant}")

    def forward(self, x):
        return self.model(x)


__all__ = ["SkeletonGraphClassifier", "BarlowTwinsSkeleton"]

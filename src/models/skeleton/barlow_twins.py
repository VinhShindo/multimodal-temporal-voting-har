"""Barlow Twins-style contrastive skeleton head. Keeps the paper idea: representation alignment and projection head."""

import numpy as np
import torch
import torch.nn as nn

from src.models.skeleton.stgcn import STGCN


class BarlowTwinsSkeleton(nn.Module):
    def __init__(self, in_channels: int = 6, num_classes: int = 35,
                 target_len: int = 64, dropout: float = 0.3):
        super().__init__()
        self.backbone = STGCN(num_classes=num_classes, num_joints=17,
                             in_channels=in_channels, dropout=dropout)
        self.projector = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )
        self.cls = nn.Linear(256, num_classes)

    def augment(self, x):
        return x + 0.01 * torch.randn_like(x)

    def loss(self, z1, z2):
        z1 = (z1 - z1.mean(0, keepdim=True)) / (z1.std(0, keepdim=True) + 1e-6)
        z2 = (z2 - z2.mean(0, keepdim=True)) / (z2.std(0, keepdim=True) + 1e-6)
        c = torch.matmul(z1.T, z2) / z1.shape[0]
        on_diag = torch.diagonal(c).add(-1).pow(2).sum()
        off_diag = (c - torch.diag(torch.diag(c))).pow(2).sum()
        return on_diag + 0.01 * off_diag

    def forward(self, x, return_projection: bool = False):
        feat = self.backbone.forward_features(x)
        proj = self.projector(feat)
        logits = self.cls(feat)
        if return_projection:
            return logits, proj
        return logits


def build_barlow_twins(num_classes: int = 35, num_joints: int = 17,
                      in_channels: int = 6, **kwargs):
    return BarlowTwinsSkeleton(in_channels=in_channels,
                              num_classes=num_classes,
                              target_len=kwargs.get("target_len", 64),
                              dropout=kwargs.get("dropout", 0.3))


if __name__ == "__main__":
    model = build_barlow_twins(num_classes=35, in_channels=6)
    x = torch.randn(2, 6, 64, 17)
    logits, proj = model(x, return_projection=True)
    print(logits.shape, proj.shape)

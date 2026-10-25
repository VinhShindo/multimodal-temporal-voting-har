"""STA-GCN-style attention model: spatial graph + temporal attention + residual residual blocks."""

import numpy as np
import torch
import torch.nn as nn

from src.models.skeleton.ctrgcn import build_A_3subset


class SpatialTemporalAttentionBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_joints: int = 17, stride: int = 1):
        super().__init__()
        self.A = nn.Parameter(torch.as_tensor(build_A_3subset(num_joints), dtype=torch.float32).clone())
        self.gcn = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.tcn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=(5, 1), padding=(2, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, max(1, out_ch // 8), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, out_ch // 8), out_ch, kernel_size=1),
            nn.Sigmoid(),
        )
        self.res = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride, 1)),
            nn.BatchNorm2d(out_ch),
        ) if in_ch != out_ch or stride != 1 else nn.Identity()
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_res = self.res(x)
        x_g = self.gcn(x)
        x_g = torch.einsum("bctv,vw->bctw", x_g, self.A[0])
        x_t = self.tcn(x_g)
        x_t = x_t * self.attn(x_t)
        out = self.bn(x_t + x_res)
        return self.relu(out)


class STAGCN(nn.Module):
    def __init__(self, num_classes: int = 35, num_joints: int = 17,
                 in_channels: int = 6, dropout: float = 0.3):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([
            SpatialTemporalAttentionBlock(64, 64, num_joints, stride=1),
            SpatialTemporalAttentionBlock(64, 64, num_joints, stride=1),
            SpatialTemporalAttentionBlock(64, 128, num_joints, stride=2),
            SpatialTemporalAttentionBlock(128, 128, num_joints, stride=1),
            SpatialTemporalAttentionBlock(128, 256, num_joints, stride=2),
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        B, C, T, V = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C * V, T)
        x = self.data_bn(x)
        x = x.view(B, C, V, T).permute(0, 1, 3, 2).contiguous()
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=(2, 3))
        x = self.dropout(x)
        return self.fc(x)


def build_sta_gcn(num_classes: int = 35, num_joints: int = 17,
                 in_channels: int = 6, **kwargs):
    return STAGCN(num_classes=num_classes,
                  num_joints=num_joints,
                  in_channels=in_channels,
                  dropout=kwargs.get("dropout", 0.3))


if __name__ == "__main__":
    model = build_sta_gcn(num_classes=35, num_joints=17, in_channels=6)
    x = torch.randn(2, 6, 64, 17)
    print(model(x).shape)

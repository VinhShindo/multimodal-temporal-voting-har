"""MS-G3D-inspired graph model. The implementation keeps the paper's core idea: multi-scale temporal kernels over a learnable spatial graph."""

import numpy as np
import torch
import torch.nn as nn

from src.models.skeleton.ctrgcn import build_A_3subset


class GraphTemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_joints: int = 17, stride: int = 1):
        super().__init__()
        self.A = nn.Parameter(torch.as_tensor(build_A_3subset(num_joints), dtype=torch.float32).clone())
        self.gcn = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        d0 = out_ch // 3
        d1 = out_ch // 3
        d2 = out_ch - d0 - d1
        self.temporal = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_ch, d0, kernel_size=(3, 1), padding=(1, 0), stride=(stride, 1)),
                nn.BatchNorm2d(d0), nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(out_ch, d1, kernel_size=(5, 1), padding=(2, 0), stride=(stride, 1)),
                nn.BatchNorm2d(d1), nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(out_ch, d2, kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1)),
                nn.BatchNorm2d(d2), nn.ReLU(inplace=True),
            ),
        ])
        self.res = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride, 1)),
            nn.BatchNorm2d(out_ch),
        ) if in_ch != out_ch or stride != 1 else nn.Identity()
        self.out_bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_res = self.res(x)
        x_s = self.gcn(x)
        x_s = torch.einsum("bctv,vw->bctw", x_s, self.A[0])
        branch_out = torch.cat([t(x_s) for t in self.temporal], dim=1)
        out = self.out_bn(branch_out + x_res)
        return self.relu(out)


class MSG3D(nn.Module):
    def __init__(self, num_classes: int = 35, num_joints: int = 17,
                 in_channels: int = 6, dropout: float = 0.3):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([
            GraphTemporalBlock(64, 64, num_joints, stride=1),
            GraphTemporalBlock(64, 64, num_joints, stride=1),
            GraphTemporalBlock(64, 128, num_joints, stride=2),
            GraphTemporalBlock(128, 128, num_joints, stride=1),
            GraphTemporalBlock(128, 256, num_joints, stride=2),
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

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


def build_msg3d(num_classes: int = 35, num_joints: int = 17,
               in_channels: int = 6, **kwargs):
    return MSG3D(num_classes=num_classes,
                 num_joints=num_joints,
                 in_channels=in_channels,
                 dropout=kwargs.get("dropout", 0.3))


if __name__ == "__main__":
    model = build_msg3d(num_classes=35, num_joints=17, in_channels=6)
    x = torch.randn(2, 6, 64, 17)
    print(model(x).shape)

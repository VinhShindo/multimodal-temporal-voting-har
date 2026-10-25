"""ST-GCN for COCO-17 skeleton sequences.

This is the paper-faithful ST-GCN family implementation: a three-subset
spatial graph convolution followed by a temporal convolution and residual
connection. The graph layout is adapted to the project's COCO-17 keypoints;
the classifier head and input channel count are project-specific.
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn


COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def _normalize(A: np.ndarray) -> np.ndarray:
    degree = A.sum(axis=1)
    inv_degree = np.zeros_like(degree)
    nonzero = degree > 0
    inv_degree[nonzero] = 1.0 / degree[nonzero]
    return inv_degree[:, None] * A


def build_stgcn_graph(num_joints: int = 17) -> torch.Tensor:
    """Build self/inward/outward ST-GCN graph partitions."""
    V = num_joints
    neighbors = {i: [] for i in range(V)}
    for i, j in COCO_EDGES:
        if i < V and j < V:
            neighbors[i].append(j)
            neighbors[j].append(i)

    distance = [-1] * V
    distance[0] = 0
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for other in neighbors[node]:
            if distance[other] < 0:
                distance[other] = distance[node] + 1
                queue.append(other)
    if min(distance) < 0:
        raise ValueError("COCO graph is disconnected for the requested V")

    self_link = np.eye(V, dtype=np.float32)
    inward = np.zeros((V, V), dtype=np.float32)
    outward = np.zeros((V, V), dtype=np.float32)
    for i, j in COCO_EDGES:
        if i >= V or j >= V:
            continue
        if distance[i] > distance[j]:
            inward[i, j] = 1.0
            outward[j, i] = 1.0
        else:
            inward[j, i] = 1.0
            outward[i, j] = 1.0

    return torch.from_numpy(np.stack([
        _normalize(self_link), _normalize(inward), _normalize(outward)
    ]))


class STGCNBlock(nn.Module):
    """One spatial graph convolution + temporal convolution block."""

    def __init__(self, in_channels: int, out_channels: int,
                 A: torch.Tensor, stride: int = 1,
                 residual: bool = True, dropout: float = 0.0):
        super().__init__()
        self.register_buffer("A", A, persistent=False)
        self.gcn = nn.Conv2d(in_channels, out_channels * A.shape[0], 1)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1),
                      padding=(4, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gcn(x)
        B, _, T, V = y.shape
        y = y.view(B, self.A.shape[0], -1, T, V)
        y = torch.einsum("bkctv,kvw->bctw", y, self.A)
        y = self.tcn(y)
        if self.residual is not None:
            y = y + self.residual(x)
        return self.relu(y)


class STGCN(nn.Module):
    """Original ST-GCN block schedule adapted to COCO-17/MMAct."""

    def __init__(self, num_classes: int = 35, num_joints: int = 17,
                 in_channels: int = 6, dropout: float = 0.3):
        super().__init__()
        A = build_stgcn_graph(num_joints)
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        # Ten blocks; temporal downsampling at blocks 5 and 8.
        schedule = [
            (in_channels, 64, 1, False),
            (64, 64, 1, True),
            (64, 64, 1, True),
            (64, 64, 1, True),
            (64, 128, 2, True),
            (128, 128, 1, True),
            (128, 128, 1, True),
            (128, 256, 2, True),
            (256, 256, 1, True),
            (256, 256, 1, True),
        ]
        self.layers = nn.ModuleList([
            STGCNBlock(i, o, A, stride=s, residual=r,
                       dropout=dropout if o == 256 else 0.0)
            for i, o, s, r in schedule
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, V = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C * V, T)
        x = self.data_bn(x)
        return x.view(B, C, V, T).permute(0, 1, 3, 2).contiguous()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare(x)
        for layer in self.layers:
            x = layer(x)
        return x.mean(dim=(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        return self.fc(self.dropout(feat))


def build_stgcn(num_classes: int = 35, num_joints: int = 17,
               in_channels: int = 6, **kwargs):
    return STGCN(num_classes=num_classes, num_joints=num_joints,
                 in_channels=in_channels,
                 dropout=kwargs.get("dropout", 0.3))


if __name__ == "__main__":
    model = build_stgcn(num_classes=35, in_channels=6)
    x = torch.randn(2, 6, 64, 17)
    print(tuple(model(x).shape))

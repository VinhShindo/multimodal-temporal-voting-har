"""
ST-GCN++ — Improved Spatio-Temporal Graph Convolutional Network.

Cải tiến so với ST-GCN gốc:
  1. Learnable adjacency = A_phys + A_attn (attention học được)
  2. Multi-scale temporal: 3 branches kernel 9/5/3 → concat
  3. SE channel attention (squeeze-excitation)
  4. Deeper: 6 stages

Input:  (B, C_in, T, V) — ví dụ (B, 6, 64, 17)
Output: (B, num_classes)
"""
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# COCO 17-keypoint skeleton
# ============================================================
COCO_EDGES = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Neck (nose → shoulders)
    (0, 5), (0, 6),
    # Shoulder-shoulder
    (5, 6),
    # Left arm
    (5, 7), (7, 9),
    # Right arm
    (6, 8), (8, 10),
    # Shoulder → hip
    (5, 11), (6, 12),
    # Hip-hip
    (11, 12),
    # Left leg
    (11, 13), (13, 15),
    # Right leg
    (12, 14), (14, 16),
]


def build_physical_adjacency(V=17):
    """A_phys = symmetric normalized physical adjacency."""
    A = np.eye(V, dtype=np.float32)
    for i, j in COCO_EDGES:
        if i < V and j < V:
            A[i, j] = 1.0
            A[j, i] = 1.0
    # D^(-1/2) A D^(-1/2)
    D = A.sum(axis=1)
    D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
    A_norm = np.diag(D_inv_sqrt) @ A @ np.diag(D_inv_sqrt)
    return torch.from_numpy(A_norm).float()


# ============================================================
# Multi-scale Temporal Conv
# ============================================================
class MultiScaleTCN(nn.Module):
    """3 branches với kernel 9 / 5 / 3, concat lại."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        mid = out_ch // 3
        rem = out_ch - 2 * mid
        pad9 = 4
        pad5 = 2
        pad3 = 1
        self.b9 = nn.Conv2d(in_ch, mid, kernel_size=(9, 1),
                            stride=(stride, 1), padding=(pad9, 0))
        self.b5 = nn.Conv2d(in_ch, mid, kernel_size=(5, 1),
                            stride=(stride, 1), padding=(pad5, 0))
        self.b3 = nn.Conv2d(in_ch, rem, kernel_size=(3, 1),
                            stride=(stride, 1), padding=(pad3, 0))
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(
            torch.cat([self.b9(x), self.b5(x), self.b3(x)], dim=1)))


# ============================================================
# SE Channel Attention
# ============================================================
class SEAttention(nn.Module):
    """Squeeze-Excitation channel attention."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(1, channels // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // reduction), channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


# ============================================================
# ST-GCN++ Block
# ============================================================
class STGCNPPBlock(nn.Module):
    """
    Block gồm:
      1. Spatial GCN: y = A @ (W_gcn · x)
         với A = A_phys + A_attn (attention học được)
      2. Multi-scale TCN
      3. SE attention
      4. Residual
    """

    def __init__(self, in_ch, out_ch, V, A_phys, stride=1):
        super().__init__()
        self.V = V
        self.stride = stride

        # Learnable attention adjacency
        self.A_attn = nn.Parameter(torch.zeros(V, V))
        nn.init.constant_(self.A_attn, 0.01)

        # Physical adjacency — non-persistent (không lưu state_dict)
        self.register_buffer("A_phys", A_phys, persistent=False)

        # Spatial GCN (1x1 conv → learn per-channel transform)
        self.gcn = nn.Conv2d(in_ch, out_ch, 1)

        # Multi-scale temporal
        self.tcn = MultiScaleTCN(out_ch, out_ch, stride=stride)

        # SE attention
        self.se = SEAttention(out_ch, reduction=8)

        # Residual
        if in_ch == out_ch and stride == 1:
            self.res = nn.Identity()
            self.res_bn = None
        else:
            self.res = nn.Conv2d(in_ch, out_ch, kernel_size=1,
                                 stride=(stride, 1))
            self.res_bn = nn.BatchNorm2d(out_ch)

        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, C_in, T, V)
        # ── Residual ──
        res = self.res(x)
        if self.res_bn is not None:
            res = self.res_bn(res)

        # ── Spatial GCN ──
        x_gcn = self.gcn(x)                       # (B, C_out, T, V)
        A = self.A_phys + self.A_attn             # (V, V)
        # y[b,c,t,w] = Σ_v x[b,c,t,v] · A[v,w]
        x_gcn = torch.einsum("bctv,vw->bctw", x_gcn, A)

        # ── Multi-scale TCN ──
        x_tcn = self.tcn(x_gcn)

        # ── SE ──
        x_se = self.se(x_tcn)

        # ── Residual + BN + ReLU ──
        out = self.bn(x_se + res)
        return self.relu(out)


# ============================================================
# ST-GCN++ Full Model
# ============================================================
class STGCNPP(nn.Module):
    """
    6 stages với channel tăng dần:
      Stage 1: in   → 64    (stride=1)
      Stage 2: 64   → 64    (stride=1)
      Stage 3: 64   → 128   (stride=2)   T: 64→32
      Stage 4: 128  → 128   (stride=1)
      Stage 5: 128  → 256   (stride=2)   T: 32→16
      Stage 6: 256  → 256   (stride=1)
    """

    def __init__(self, num_classes=32, num_joints=17, in_channels=6):
        super().__init__()
        V = num_joints
        A_phys = build_physical_adjacency(V)

        # Data BN
        self.data_bn = nn.BatchNorm1d(in_channels * V)

        # Stages
        self.stage1 = STGCNPPBlock(in_channels, 64, V, A_phys)
        self.stage2 = STGCNPPBlock(64, 64, V, A_phys)
        self.stage3 = STGCNPPBlock(64, 128, V, A_phys, stride=2)
        self.stage4 = STGCNPPBlock(128, 128, V, A_phys)
        self.stage5 = STGCNPPBlock(128, 256, V, A_phys, stride=2)
        self.stage6 = STGCNPPBlock(256, 256, V, A_phys)

        # Head
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, C, T, V)
        B, C, T, V = x.shape

        # Data BN — reshape (B, C, V, T) → (B, C*V, T)
        x = x.permute(0, 1, 3, 2).contiguous()  # (B, C, V, T)
        x = x.view(B, C * V, T)
        x = self.data_bn(x)
        x = x.view(B, C, V, T).permute(0, 1, 3, 2).contiguous()  # (B, C, T, V)

        # Stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)

        # GAP over T and V
        x = x.mean(dim=2)      # (B, 256, V)
        x = x.mean(dim=-1)     # (B, 256)

        x = self.dropout(x)
        return self.fc(x)


def build_stgcnpp(num_classes=32, num_joints=17, in_channels=6):
    return STGCNPP(num_classes=num_classes,
                   num_joints=num_joints,
                   in_channels=in_channels)


# ============================================================
# Sanity check
# ============================================================
if __name__ == "__main__":
    m = build_stgcnpp(num_classes=32, num_joints=17, in_channels=6)
    x = torch.randn(2, 6, 64, 17)
    y = m(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")
    n_params = sum(p.numel() for p in m.parameters())
    print(f"Params: {n_params / 1e6:.2f} M")
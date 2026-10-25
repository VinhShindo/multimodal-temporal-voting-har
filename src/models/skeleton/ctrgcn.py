"""
CTR-GCN cho MMAct (COCO 17 keypoint, 2D) — 3 kiến trúc chọn bằng `build_model(arch=...)`:

  "ctrgcn"       : baseline 10 layer + GAP (class CTRGCN, GIỮ NGUYÊN tên tham số
                   data_bn / layers / fc → load được checkpoint của run cũ)
  "ctrgcn_head"  : global CTR-GCN 10 layer + PartTransformerHead thay GAP        (ablation R2)
  "dual"         : global branch + upper-body branch + part transformer head     (ablation R3/R4)

Input:
  ctrgcn / ctrgcn_head : (B, C, T, 17)            C = 6 (base) hoặc 11 (rich)
  dual                 : (B, 18, T, 17)           = 11 kênh global || 7 kênh upper (pad V 9→17)
                         → gói chung 1 tensor để full_evaluation / fusion không phải sửa.
Output: logits (B, num_classes). `forward(x, return_feat=True)` trả thêm feature trước FC.
"""
import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Graph definitions
# ============================================================
# COCO: 0 nose | 1 l_eye 2 r_eye | 3 l_ear 4 r_ear | 5 l_sho 6 r_sho | 7 l_elb 8 r_elb
#       9 l_wri 10 r_wri | 11 l_hip 12 r_hip | 13 l_knee 14 r_knee | 15 l_ank 16 r_ank
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

# 6 bộ phận để pooling trước transformer: đầu, tay T, tay P, thân, chân T, chân P
PARTS = [[0, 1, 2, 3, 4], [5, 7, 9], [6, 8, 10], [5, 6, 11, 12],
         [11, 13, 15], [12, 14, 16]]

# Upper-body (V=9): mũi, 2 mắt, 2 vai, 2 khuỷu, 2 cổ tay (chỉ số COCO gốc)
UB_JOINTS = [0, 1, 2, 5, 6, 7, 8, 9, 10]
# Trong chỉ số cục bộ 0..8: 0 nose,1 leye,2 reye,3 lsho,4 rsho,5 lelb,6 relb,7 lwri,8 rwri
UB_EDGES = [(0, 1), (0, 2), (0, 3), (0, 4), (3, 4),
            (3, 5), (5, 7), (4, 6), (6, 8)]
UB_CENTER = 3


def build_A_3subset(num_joints: int = 17, edges=COCO_EDGES,
                    center: int = 11) -> np.ndarray:
    """(3, V, V) = [self, inward, outward], row-normalized. out[u] = sum_v A[u,v] x[v]."""
    V = num_joints
    nbr = {i: [] for i in range(V)}
    for i, j in edges:
        nbr[i].append(j)
        nbr[j].append(i)

    dist = [-1] * V
    dist[center] = 0
    q = deque([center])
    while q:
        u = q.popleft()
        for w in nbr[u]:
            if dist[w] < 0:
                dist[w] = dist[u] + 1
                q.append(w)
    assert min(dist) >= 0, "Đồ thị không liên thông"

    A_in = np.zeros((V, V), np.float32)
    A_out = np.zeros((V, V), np.float32)
    for i, j in edges:
        if dist[i] > dist[j]:
            A_in[i, j] = 1.0
            A_out[j, i] = 1.0
        elif dist[i] < dist[j]:
            A_in[j, i] = 1.0
            A_out[i, j] = 1.0
        else:
            A_in[i, j] = 1.0
            A_in[j, i] = 1.0

    def norm(M):
        d = M.sum(axis=1, keepdims=True)
        d[d == 0] = 1.0
        return M / d

    return np.stack([np.eye(V, dtype=np.float32), norm(A_in), norm(A_out)])


# ============================================================
# CTR-GC / GCN unit / MS-TCN / Block
# ============================================================
class CTRGC(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 rel_reduction: int = 8):
        super().__init__()
        # in_channels < 32 (3, 6, 7, 11, 18...) → 8 kênh rel; còn lại như bản gốc
        rel = 8 if in_channels < 32 else max(1, in_channels // rel_reduction)
        self.conv1 = nn.Conv2d(in_channels, rel, 1)
        self.conv2 = nn.Conv2d(in_channels, rel, 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv4 = nn.Conv2d(rel, out_channels, 1)
        self.tanh = nn.Tanh()

    def forward(self, x, A, alpha):
        x1 = self.conv1(x).mean(-2)
        x2 = self.conv2(x).mean(-2)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + A
        x3 = self.conv3(x)
        return torch.einsum("ncuv,nctv->nctu", x1, x3)


class UnitGCN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A3: np.ndarray):
        super().__init__()
        self.convs = nn.ModuleList(
            [CTRGC(in_channels, out_channels) for _ in range(A3.shape[0])])
        self.A = nn.Parameter(torch.as_tensor(A3, dtype=torch.float32).clone())
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        if in_channels == out_channels:
            self.down = nn.Identity()
        else:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = 0
        for i, conv in enumerate(self.convs):
            y = y + conv(x, self.A[i], self.alpha)
        return self.relu(self.bn(y) + self.down(x))


class _TemporalConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=5, stride=1, dilation=1):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(in_c, out_c, (kernel_size, 1),
                              stride=(stride, 1), padding=(pad, 0),
                              dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class MSTCN(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=5, stride=1, dilations=(1, 2)):
        super().__init__()
        nb = len(dilations) + 2
        assert out_c % nb == 0, f"out_c={out_c} phải chia hết cho {nb}"
        bc = out_c // nb
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_c, bc, 1), nn.BatchNorm2d(bc), nn.ReLU(inplace=True),
                _TemporalConv(bc, bc, kernel_size, stride, d)))
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_c, bc, 1), nn.BatchNorm2d(bc), nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(bc)))
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_c, bc, 1, stride=(stride, 1)), nn.BatchNorm2d(bc)))

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], dim=1)


class CTRGCBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A3: np.ndarray,
                 stride: int = 1, residual: bool = True,
                 kernel_size: int = 5, dilations=(1, 2)):
        super().__init__()
        self.gcn = UnitGCN(in_channels, out_channels, A3)
        self.tcn = MSTCN(out_channels, out_channels, kernel_size=kernel_size,
                         stride=stride, dilations=dilations)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        y = self.tcn(self.gcn(x))
        if self.residual is not None:
            y = y + self.residual(x)
        return self.relu(y)


def _init_backbone(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    for m in module.modules():
        if isinstance(m, UnitGCN):
            nn.init.constant_(m.bn.weight, 1e-6)


def _t_out(T: int, n_stride2: int = 2) -> int:
    for _ in range(n_stride2):
        T = (T - 1) // 2 + 1
    return T


# ============================================================
# Baseline (GIỮ NGUYÊN cấu trúc tên tham số của run cũ)
# ============================================================
class CTRGCN(nn.Module):
    def __init__(self, num_classes: int = 35, num_joints: int = 17,
                 in_channels: int = 6, base_channels: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        A3 = build_A_3subset(num_joints)
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        c = base_channels
        cfgs = [
            (in_channels, c,     1, False),
            (c,           c,     1, True),
            (c,           c,     1, True),
            (c,           c,     1, True),
            (c,           c * 2, 2, True),
            (c * 2,       c * 2, 1, True),
            (c * 2,       c * 2, 1, True),
            (c * 2,       c * 4, 2, True),
            (c * 4,       c * 4, 1, True),
            (c * 4,       c * 4, 1, True),
        ]
        self.layers = nn.ModuleList([
            CTRGCBlock(i, o, A3, stride=s, residual=r) for i, o, s, r in cfgs])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c * 4, num_classes)

        _init_backbone(self)
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, return_feat: bool = False):
        B, C, T, V = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C * V, T)
        x = self.data_bn(x)
        x = x.view(B, C, V, T).permute(0, 1, 3, 2).contiguous()
        for layer in self.layers:
            x = layer(x)
        feat = x.mean(dim=(2, 3))
        out = self.fc(self.dropout(feat))
        return (out, feat) if return_feat else out


# ============================================================
# Backbone dùng chung cho các kiến trúc mới
# ============================================================
class GCNBackbone(nn.Module):
    def __init__(self, in_channels: int, num_joints: int, A3: np.ndarray,
                 layer_cfg):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        self.layers = nn.ModuleList([
            CTRGCBlock(i, o, A3, stride=s, residual=r)
            for i, o, s, r in layer_cfg])
        _init_backbone(self)

    def forward(self, x):                      # (B, C, T, V) → (B, C_out, T', V)
        B, C, T, V = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C * V, T)
        x = self.data_bn(x)
        x = x.view(B, C, V, T).permute(0, 1, 3, 2).contiguous()
        for layer in self.layers:
            x = layer(x)
        return x


def global_layer_cfg(in_c: int, c: int = 64):
    # 10 layer: T 64→32→16
    return [(in_c, c, 1, False), (c, c, 1, True), (c, c, 1, True), (c, c, 1, True),
            (c, 2 * c, 2, True), (2 * c, 2 * c, 1, True), (2 * c, 2 * c, 1, True),
            (2 * c, 4 * c, 2, True), (4 * c, 4 * c, 1, True), (4 * c, 4 * c, 1, True)]


def upper_layer_cfg(in_c: int, c: int = 64):
    # 6 layer: T 64→32→16
    return [(in_c, c, 1, False), (c, c, 1, True),
            (c, 2 * c, 2, True), (2 * c, 2 * c, 1, True),
            (2 * c, 4 * c, 2, True), (4 * c, 4 * c, 1, True)]


# ============================================================
# Part-token Transformer head
# ============================================================
class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.shape[0], 1, 1).bernoulli_(keep)
        return x * mask / keep


class TBlock(nn.Module):
    """Pre-norm transformer block với DropPath và key_padding_mask."""

    def __init__(self, c, heads, drop, drop_path):
        super().__init__()
        self.ln1 = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, dropout=drop, batch_first=True)
        self.ln2 = nn.LayerNorm(c)
        self.ffn = nn.Sequential(
            nn.Linear(c, 2 * c), nn.GELU(), nn.Dropout(drop),
            nn.Linear(2 * c, c), nn.Dropout(drop))
        self.dp = DropPath(drop_path)

    def forward(self, x, kpm=None):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=kpm, need_weights=False)
        x = x + self.dp(a)
        return x + self.dp(self.ffn(self.ln2(x)))


class PartTransformerHead(nn.Module):
    """
    (B, C, T', V=17) → pool theo 6 bộ phận → T'×6 token + CLS → transformer → CLS (B, C).
    part_drop: xác suất che ngẫu nhiên từng token bộ phận khi train (CLS không bao giờ bị che).
    """

    def __init__(self, c: int = 256, t_out: int = 16, num_joints: int = 17,
                 depth: int = 2, heads: int = 4, drop: float = 0.1,
                 drop_path: float = 0.1, part_drop: float = 0.1):
        super().__init__()
        P = len(PARTS)
        pm = torch.zeros(P, num_joints)
        for i, js in enumerate(PARTS):
            pm[i, js] = 1.0 / len(js)
        self.register_buffer("pm", pm)
        self.part_drop = part_drop
        self.in_ln = nn.LayerNorm(c)
        self.cls = nn.Parameter(torch.zeros(1, 1, c))
        self.pos_t = nn.Parameter(torch.zeros(1, t_out, 1, c))
        self.pos_p = nn.Parameter(torch.zeros(1, 1, P, c))
        for p in (self.cls, self.pos_t, self.pos_p):
            nn.init.trunc_normal_(p, std=0.02)
        dps = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList([TBlock(c, heads, drop, dps[i]) for i in range(depth)])
        self.ln = nn.LayerNorm(c)
        for m in self.blocks.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        p = torch.einsum("bctv,pv->btpc", x, self.pm)           # (B, T', P, C)
        B, T, P, C = p.shape
        p = self.in_ln(p) + self.pos_t[:, :T] + self.pos_p
        tok = torch.cat([self.cls.expand(B, -1, -1), p.reshape(B, T * P, C)], dim=1)

        kpm = None
        if self.training and self.part_drop > 0:
            kpm = torch.rand(B, T * P, device=x.device) < self.part_drop
            kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), kpm], dim=1)

        for blk in self.blocks:
            tok = blk(tok, kpm)
        return self.ln(tok[:, 0])


# ============================================================
# ctrgcn_head: global branch + part transformer head (ablation R2)
# ============================================================
class CTRGCNPart(nn.Module):
    def __init__(self, num_classes: int = 32, num_joints: int = 17,
                 in_channels: int = 11, target_len: int = 64,
                 base_channels: int = 64, dropout: float = 0.4,
                 depth: int = 2, heads: int = 4, drop_path: float = 0.1,
                 part_drop: float = 0.1):
        super().__init__()
        A3 = build_A_3subset(num_joints)
        c4 = base_channels * 4
        self.backbone = GCNBackbone(in_channels, num_joints, A3,
                                    global_layer_cfg(in_channels, base_channels))
        self.head = PartTransformerHead(c4, _t_out(target_len), num_joints,
                                        depth, heads, 0.1, drop_path, part_drop)
        self.norm = nn.LayerNorm(c4)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c4, num_classes)
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, return_feat: bool = False):
        feat = self.norm(self.head(self.backbone(x)))
        out = self.fc(self.dropout(feat))
        return (out, feat) if return_feat else out


# ============================================================
# dual: global (11ch, 17 khớp) + upper-body (7ch, 9 khớp) + head
# ============================================================
class DualBranchCTRGCN(nn.Module):
    G_CH = 11
    U_CH = 7
    U_V = len(UB_JOINTS)      # 9

    def __init__(self, num_classes: int = 32, num_joints: int = 17,
                 in_channels: int = 18, target_len: int = 64,
                 base_channels: int = 64, dropout: float = 0.4,
                 depth: int = 2, heads: int = 4, drop_path: float = 0.1,
                 part_drop: float = 0.1):
        super().__init__()
        assert in_channels == self.G_CH + self.U_CH, \
            f"dual cần in_channels={self.G_CH + self.U_CH}, nhận {in_channels}"
        c4 = base_channels * 4

        self.global_branch = GCNBackbone(
            self.G_CH, num_joints, build_A_3subset(num_joints),
            global_layer_cfg(self.G_CH, base_channels))
        self.upper_branch = GCNBackbone(
            self.U_CH, self.U_V,
            build_A_3subset(self.U_V, UB_EDGES, UB_CENTER),
            upper_layer_cfg(self.U_CH, base_channels))
        self.head = PartTransformerHead(c4, _t_out(target_len), num_joints,
                                        depth, heads, 0.1, drop_path, part_drop)
        self.norm = nn.LayerNorm(2 * c4)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * c4, num_classes)
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, return_feat: bool = False):
        g = x[:, :self.G_CH]                                            # (B, 11, T, 17)
        u = x[:, self.G_CH:self.G_CH + self.U_CH, :, :self.U_V]         # (B, 7, T, 9)
        fg = self.head(self.global_branch(g))                           # (B, 256)
        fu = self.upper_branch(u).mean(dim=(2, 3))                      # (B, 256)
        feat = self.norm(torch.cat([fg, fu], dim=1))                    # (B, 512)
        out = self.fc(self.dropout(feat))
        return (out, feat) if return_feat else out


# ============================================================
# Factory
# ============================================================
def build_model(arch: str = "dual", num_classes: int = 35, num_joints: int = 17,
                in_channels: int = None, target_len: int = 64, **kw):
    arch = str(arch).lower()
    if arch == "ctrgcn":
        return CTRGCN(num_classes, num_joints, in_channels or 6,
                      dropout=kw.get("dropout", 0.3))
    common = dict(dropout=kw.get("dropout", 0.4), depth=kw.get("tf_depth", 2),
                  heads=kw.get("tf_heads", 4), drop_path=kw.get("drop_path", 0.1),
                  part_drop=kw.get("part_drop", 0.1))
    if arch == "ctrgcn_head":
        return CTRGCNPart(num_classes, num_joints, in_channels or 11,
                          target_len, **common)
    if arch == "dual":
        return DualBranchCTRGCN(num_classes, num_joints, in_channels or 18,
                                target_len, **common)
    raise ValueError(f"Unknown arch: {arch}")


def build_ctrgcn(num_classes: int = 35, num_joints: int = 17,
                 in_channels: int = 6) -> CTRGCN:
    """Giữ lại cho code cũ (fusion/eval) — baseline."""
    return CTRGCN(num_classes=num_classes, num_joints=num_joints,
                  in_channels=in_channels)


if __name__ == "__main__":
    for arch, ch in [("ctrgcn", 6), ("ctrgcn_head", 11), ("dual", 18)]:
        m = build_model(arch, 32, 17, ch, target_len=64)
        x = torch.randn(2, ch, 64, 17)
        y, f = m(x, return_feat=True)
        n = sum(p.numel() for p in m.parameters())
        print(f"{arch:12s} in={tuple(x.shape)} out={tuple(y.shape)} "
              f"feat={tuple(f.shape)} params={n / 1e6:.2f}M")
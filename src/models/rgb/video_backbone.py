"""
Video backbone wrapper cho MMAct RGB.

Hỗ trợ các model torchvision 0.21:
  - r2plus1d_18 (default, 33M params)
  - r3d_18      (33M)
  - mc3_18      (11M — nhẹ)
  - s3d         (8.9M)
  - mvit_v2_s   (34M — cần VRAM lớn)
"""
import torch
import torch.nn as nn


SUPPORTED_ARCHS = ["r2plus1d_18", "r3d_18", "mc3_18", "s3d", "mvit_v2_s"]


def build_backbone(arch: str, num_classes: int,
                   pretrained: bool = True, dropout: float = 0.3):
    """Trả về model với classifier thay thế."""
    from torchvision import models

    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"arch phải là 1 trong {SUPPORTED_ARCHS}")

    if arch == "r2plus1d_18":
        weights = models.video.R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        model = models.video.r2plus1d_18(weights=weights)
        in_feat = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout),
                                 nn.Linear(in_feat, num_classes))
        input_layout = "BCTHW"   # (B, C, T, H, W)

    elif arch == "r3d_18":
        weights = models.video.R3D_18_Weights.KINETICS400_V1 if pretrained else None
        model = models.video.r3d_18(weights=weights)
        in_feat = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout),
                                 nn.Linear(in_feat, num_classes))
        input_layout = "BCTHW"

    elif arch == "mc3_18":
        weights = models.video.MC3_18_Weights.KINETICS400_V1 if pretrained else None
        model = models.video.mc3_18(weights=weights)
        in_feat = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout),
                                 nn.Linear(in_feat, num_classes))
        input_layout = "BCTHW"

    elif arch == "s3d":
        weights = models.video.S3D_Weights.KINETICS400_V1 if pretrained else None
        model = models.video.s3d(weights=weights)
        # S3D có `classifier` là Sequential với Linear cuối
        in_feat = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feat, num_classes)
        input_layout = "BCTHW"

    elif arch == "mvit_v2_s":
        weights = models.video.MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None
        model = models.video.mvit_v2_s(weights=weights)
        in_feat = model.head[-1].in_features
        model.head[-1] = nn.Linear(in_feat, num_classes)
        input_layout = "BCTHW"

    return model, input_layout


class VideoClassifier(nn.Module):
    """Wrapper chuẩn hóa input (B, T, C, H, W) → (B, C, T, H, W)."""

    def __init__(self, arch: str, num_classes: int,
                 pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.arch = arch
        self.model, self.input_layout = build_backbone(
            arch, num_classes, pretrained, dropout)

    def forward(self, x):
        # x: (B, T, C, H, W) từ dataset
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # → (B, C, T, H, W)
        return self.model(x)


def build_video_model(arch: str = "r2plus1d_18",
                      num_classes: int = 32,
                      pretrained: bool = True,
                      dropout: float = 0.3) -> VideoClassifier:
    return VideoClassifier(arch, num_classes, pretrained, dropout)


if __name__ == "__main__":
    for arch in SUPPORTED_ARCHS:
        try:
            m = build_video_model(arch, 32, pretrained=False)
            x = torch.randn(1, 16, 3, 224, 224)   # (B, T, C, H, W)
            y = m(x)
            n = sum(p.numel() for p in m.parameters()) / 1e6
            print(f"  {arch:15s}  params={n:6.2f}M  out={tuple(y.shape)}")
        except Exception as e:
            print(f"  {arch:15s}  ERROR: {e}")
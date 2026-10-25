"""Video backbone wrapper for MMAct RGB.

This module preserves the legacy API while routing each architecture to its dedicated
per-file implementation under src/models/rgb/.
"""

import torch
import torch.nn as nn

from src.models.rgb.mc3_18 import MC3_18Classifier
from src.models.rgb.mvit_v2_s import MViTV2SClassifier
from src.models.rgb.r2plus1d_18 import R2Plus1D18Classifier
from src.models.rgb.r3d_18 import R3D18Classifier
from src.models.rgb.s3d import S3DClassifier
from src.models.rgb.videomae import VideoMAEClassifier
from src.models.rgb.videomae2 import VideoMAE2Classifier

SUPPORTED_ARCHS = [
    "r2plus1d_18", "r3d_18", "mc3_18", "s3d", "mvit_v2_s",
    "videomae", "videomae2",
]
ARCH_BUILDERS = {
    "r2plus1d_18": R2Plus1D18Classifier,
    "r3d_18": R3D18Classifier,
    "mc3_18": MC3_18Classifier,
    "s3d": S3DClassifier,
    "mvit_v2_s": MViTV2SClassifier,
    "videomae": VideoMAEClassifier,
    "videomae2": VideoMAE2Classifier,
}


class VideoClassifier(nn.Module):
    """Compatibility wrapper that accepts input (B, T, C, H, W) and permutes it."""

    def __init__(self, arch: str, num_classes: int,
                 pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        if arch not in ARCH_BUILDERS:
            raise ValueError(f"arch must be one of {SUPPORTED_ARCHS}")
        self.arch = arch
        self.model = ARCH_BUILDERS[arch](num_classes=num_classes,
                                        pretrained=pretrained,
                                        dropout=dropout)

    def forward(self, x):
        return self.model(x)


def build_backbone(arch: str, num_classes: int,
                   pretrained: bool = True, dropout: float = 0.3):
    return ARCH_BUILDERS[arch](num_classes=num_classes,
                              pretrained=pretrained,
                              dropout=dropout), "BCTHW"


def build_video_model(arch: str = "r2plus1d_18",
                      num_classes: int = 32,
                      pretrained: bool = True,
                      dropout: float = 0.3) -> VideoClassifier:
    return VideoClassifier(arch, num_classes, pretrained, dropout)


if __name__ == "__main__":
    for arch in SUPPORTED_ARCHS:
        try:
            m = build_video_model(arch, 32, pretrained=False)
            x = torch.randn(1, 16, 3, 224, 224)
            y = m(x)
            n = sum(p.numel() for p in m.parameters()) / 1e6
            print(f"  {arch:15s}  params={n:6.2f}M  out={tuple(y.shape)}")
        except Exception as e:
            print(f"  {arch:15s}  ERROR: {e}")
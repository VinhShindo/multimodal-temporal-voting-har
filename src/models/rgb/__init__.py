from .mc3_18 import (
    MC3_18Classifier,
    MC3_18TemporalPyramidClassifier,
    build_mc3_18,
    build_mc3_18_temporal_pyramid,
)
from .mvit_v2_s import MViTV2SClassifier, build_mvit_v2_s
from .r3d_18 import R3D18Classifier, build_r3d_18
from .s3d import S3DClassifier, build_s3d
from .videomae import VideoMAEClassifier, build_videomae
from .videomae2 import VideoMAE2Classifier, build_videomae2

__all__ = [
    "R3D18Classifier",
    "MC3_18Classifier",
    "MC3_18TemporalPyramidClassifier",
    "S3DClassifier",
    "MViTV2SClassifier",
    "VideoMAEClassifier",
    "VideoMAE2Classifier",
    "build_model",
]


def build_model(arch: str = "mc3_18", num_classes: int = 35,
               pretrained: bool = True, dropout: float = 0.3):
    arch = arch.lower()
    if arch == "r3d_18":
        return build_r3d_18(num_classes=num_classes,
                            pretrained=pretrained,
                            dropout=dropout)
    if arch == "mc3_18":
        return build_mc3_18(num_classes=num_classes,
                           pretrained=pretrained,
                           dropout=dropout)
    if arch == "mc3_18_temporal_pyramid":
        return build_mc3_18_temporal_pyramid(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )
    if arch == "s3d":
        return build_s3d(num_classes=num_classes,
                        pretrained=pretrained,
                        dropout=dropout)
    if arch == "mvit_v2_s":
        return build_mvit_v2_s(num_classes=num_classes,
                              pretrained=pretrained,
                              dropout=dropout)
    if arch == "videomae":
        return build_videomae(num_classes=num_classes,
                             pretrained=pretrained,
                             dropout=dropout)
    if arch == "videomae2":
        return build_videomae2(num_classes=num_classes,
                              pretrained=pretrained,
                              dropout=dropout)
    raise ValueError(f"Unknown RGB arch: {arch}")

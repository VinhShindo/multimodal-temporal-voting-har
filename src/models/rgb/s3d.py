"""S3D video model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class S3DClassifier(nn.Module):
    def __init__(self, num_classes: int = 35, pretrained: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        weights = models.video.S3D_Weights.KINETICS400_V1 if pretrained else None
        self.model = models.video.s3d(weights=weights)
        conv = self.model.classifier[1]
        self.model.classifier[1] = nn.Conv3d(
            in_channels=conv.in_channels,
            out_channels=num_classes,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=conv.bias is not None,
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        if x.shape[2] < 16:
            x = F.interpolate(
                x, size=(16, x.shape[3], x.shape[4]),
                mode="trilinear", align_corners=False)
        return self.model(x)


def build_s3d(num_classes: int = 35, pretrained: bool = True,
             dropout: float = 0.3):
    return S3DClassifier(num_classes=num_classes,
                         pretrained=pretrained,
                         dropout=dropout)


if __name__ == "__main__":
    model = build_s3d(num_classes=35, pretrained=False)
    x = torch.randn(2, 16, 3, 224, 224)
    print(model(x).shape)

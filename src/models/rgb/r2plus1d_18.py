"""R(2+1)D-18 video model implementation."""

import torch
import torch.nn as nn
from torchvision import models


class R2Plus1D18Classifier(nn.Module):
    def __init__(self, num_classes: int = 35, pretrained: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        weights = models.video.R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        self.model = models.video.r2plus1d_18(weights=weights)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return self.model(x)


def build_r2plus1d_18(num_classes: int = 35, pretrained: bool = True,
                     dropout: float = 0.3):
    return R2Plus1D18Classifier(num_classes=num_classes,
                               pretrained=pretrained,
                               dropout=dropout)


if __name__ == "__main__":
    model = build_r2plus1d_18(num_classes=35, pretrained=False)
    x = torch.randn(2, 16, 3, 224, 224)
    print(model(x).shape)

"""MViT V2 Small video model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MViTV2SClassifier(nn.Module):
    def __init__(self, num_classes: int = 35, pretrained: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        weights = models.video.MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None
        self.model = models.video.mvit_v2_s(weights=weights)
        in_features = self.model.head[-1].in_features
        self.model.head[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        if x.shape[2] < 16:
            x = F.interpolate(
                x, size=(16, x.shape[3], x.shape[4]),
                mode="trilinear", align_corners=False)
        return self.model(x)


def build_mvit_v2_s(num_classes: int = 35, pretrained: bool = True,
                   dropout: float = 0.3):
    return MViTV2SClassifier(num_classes=num_classes,
                            pretrained=pretrained,
                            dropout=dropout)


if __name__ == "__main__":
    model = build_mvit_v2_s(num_classes=35, pretrained=False)
    x = torch.randn(2, 16, 3, 224, 224)
    print(model(x).shape)

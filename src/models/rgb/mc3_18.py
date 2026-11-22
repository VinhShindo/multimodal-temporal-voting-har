"""MC3-18 video model implementation."""

import torch
import torch.nn as nn
from torchvision import models


class MC3_18Classifier(nn.Module):
    def __init__(self, num_classes: int = 35, pretrained: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        weights = models.video.MC3_18_Weights.KINETICS400_V1 if pretrained else None
        self.model = models.video.mc3_18(weights=weights)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return self.model(x)


class MC3_18TemporalPyramidClassifier(nn.Module):
    """MC3-18 with a classifier head that preserves coarse temporal phases."""

    def __init__(self, num_classes: int = 35, pretrained: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        weights = models.video.MC3_18_Weights.KINETICS400_V1 if pretrained else None
        self.model = models.video.mc3_18(weights=weights)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.pool_sizes = (1, 2, 4)
        pooled_features = in_features * sum(self.pool_sizes)
        self.temporal_head = nn.Sequential(
            nn.LayerNorm(pooled_features),
            nn.Dropout(dropout),
            nn.Linear(pooled_features, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        features = self.model.stem(x)
        features = self.model.layer1(features)
        features = self.model.layer2(features)
        features = self.model.layer3(features)
        features = self.model.layer4(features)
        features = features.mean(dim=(-1, -2))
        pyramid = [
            nn.functional.adaptive_avg_pool1d(features, bins).flatten(1)
            for bins in self.pool_sizes
        ]
        return self.temporal_head(torch.cat(pyramid, dim=1))


def build_mc3_18(num_classes: int = 35, pretrained: bool = True,
                dropout: float = 0.3):
    return MC3_18Classifier(num_classes=num_classes,
                           pretrained=pretrained,
                           dropout=dropout)


def build_mc3_18_temporal_pyramid(num_classes: int = 35,
                                  pretrained: bool = True,
                                  dropout: float = 0.3):
    return MC3_18TemporalPyramidClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )


if __name__ == "__main__":
    model = build_mc3_18(num_classes=35, pretrained=False)
    x = torch.randn(2, 16, 3, 224, 224)
    print(model(x).shape)

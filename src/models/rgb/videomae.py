"""VideoMAE wrapper cho RGB."""
import torch
import torch.nn as nn

from src.models.base_model import BaseModel


class VideoMAEClassifier(BaseModel):
    def __init__(self, backbone: str = "ViT-L/16", num_classes: int = 14, pretrained: bool = True):
        super().__init__()
        # TODO: load backbone từ transformers hoặc timm
        self.backbone = None
        self.head = nn.Linear(1024, num_classes)

    def forward(self, x):
        # x: (B, C, T, H, W)
        feat = torch.zeros(x.size(0), 1024, device=x.device)  # placeholder
        return self.head(feat)

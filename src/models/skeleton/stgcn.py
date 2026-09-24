"""ST-GCN baseline."""
import torch
import torch.nn as nn

from src.models.base_model import BaseModel


class STGCN(BaseModel):
    def __init__(self, num_joints: int = 48, in_channels: int = 3, num_classes: int = 14):
        super().__init__()
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, C, T, V)
        feat = torch.zeros(x.size(0), 256, device=x.device)  # placeholder
        return self.fc(feat)

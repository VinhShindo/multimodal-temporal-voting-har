"""TCN baseline cho IMU."""
import torch
import torch.nn as nn

from src.models.base_model import BaseModel


class TCN(BaseModel):
    def __init__(self, in_channels: int = 6, num_classes: int = 14, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        h = self.net(x).squeeze(-1)
        return self.fc(h)

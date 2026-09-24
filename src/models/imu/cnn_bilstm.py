"""1D CNN + BiLSTM cho IMU."""
import torch
import torch.nn as nn

from src.models.base_model import BaseModel


class CNNBiLSTM(BaseModel):
    def __init__(self,
                 in_channels: int = 6,
                 num_classes: int = 14,
                 cnn_channels=(64, 128, 128),
                 lstm_hidden: int = 128,
                 lstm_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = in_channels
        for c in cnn_channels:
            layers += [nn.Conv1d(prev, c, 5, padding=2), nn.BatchNorm1d(c), nn.ReLU()]
            prev = c
        self.cnn = nn.Sequential(*layers)
        self.lstm = nn.LSTM(prev, lstm_hidden, lstm_layers,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        # x: (B, C, T)
        h = self.cnn(x).transpose(1, 2)      # B,T,C
        h, _ = self.lstm(h)                  # B,T,2H
        h = h.mean(dim=1)                    # global pool
        return self.fc(self.dropout(h))

"""Alternative IMU architectures for controlled architecture comparisons."""
import torch
import torch.nn as nn

from src.models.base_model import BaseModel
from src.models.imu.cnn_bilstm import CNNBiLSTM


class CBAM1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8, kernel_size: int = 7):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.channel_mlp = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=False),
        )
        self.temporal = nn.Conv1d(2, 1, kernel_size=kernel_size,
                                  padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=-1, keepdim=True)
        peak = x.amax(dim=-1, keepdim=True)
        x = x * torch.sigmoid(self.channel_mlp(avg) + self.channel_mlp(peak))
        avg = x.mean(dim=1, keepdim=True)
        peak = x.amax(dim=1, keepdim=True)
        return x * torch.sigmoid(self.temporal(torch.cat([avg, peak], dim=1)))


class CNNBiLSTMCBAM(BaseModel):
    """CNN-BiLSTM baseline with channel and temporal attention."""
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, 5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.attention = CBAM1D(128)
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        h = self.attention(self.cnn(x)).transpose(1, 2)
        h, _ = self.lstm(h)
        return self.fc(self.dropout(h.mean(dim=1)))


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        padding = 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=padding,
                      dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 5, padding=padding,
                      dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.net(x))


class IMUTCN(BaseModel):
    """Residual dilated temporal convolution network."""
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.25):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, 5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[
            TemporalResidualBlock(64, dilation, dropout)
            for dilation in (1, 2, 4, 8, 16)
        ])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(64, num_classes))

    def forward(self, x):
        h = self.blocks(self.stem(x)).mean(dim=-1)
        return self.head(h)


class IMUTransformer(BaseModel):
    """Convolutional stem followed by a compact temporal Transformer encoder."""
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.2,
                 d_model: int = 96, max_length: int = 512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, d_model, 7, padding=3, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.position = nn.Parameter(torch.zeros(1, max_length, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward= d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3,
                                              enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, x):
        h = self.stem(x).transpose(1, 2)
        h = self.encoder(h + self.position[:, :h.shape[1]])
        return self.head(self.norm(h).mean(dim=1))


class XTinyHAR(BaseModel):
    """Compact patch-based inertial Transformer student."""
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.15,
                 d_model: int = 64, patch_size: int = 25,
                 max_patches: int = 32):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=patch_size,
                      stride=patch_size, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.position = nn.Parameter(torch.zeros(1, max_patches, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        h = self.patch_embed(x).transpose(1, 2)
        if h.shape[1] > self.position.shape[1]:
            raise ValueError(
                f"XTinyHAR got {h.shape[1]} patches; max is "
                f"{self.position.shape[1]}"
            )
        h = self.encoder(h + self.position[:, :h.shape[1]])
        return self.head(self.norm(h).mean(dim=1))


class SamePadConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.left_pad = (kernel_size - 1) // 2
        self.right_pad = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False)

    def forward(self, x):
        x = nn.functional.pad(x, (self.left_pad, self.right_pad))
        return self.conv(x)


class InceptionTimeModule(nn.Module):
    def __init__(self, in_channels: int, bottleneck: int = 32,
                 filters_per_branch: int = 32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck, kernel_size=1,
                                    bias=False)
        self.branches = nn.ModuleList([
            SamePadConv1d(bottleneck, filters_per_branch, kernel_size)
            for kernel_size in (10, 20, 40)
        ])
        self.pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, filters_per_branch, kernel_size=1, bias=False),
        )
        output_channels = filters_per_branch * 4
        self.norm = nn.BatchNorm1d(output_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        bottleneck = self.bottleneck(x)
        features = [branch(bottleneck) for branch in self.branches]
        features.append(self.pool(x))
        return self.activation(self.norm(torch.cat(features, dim=1)))


class InceptionResidualBlock(nn.Module):
    def __init__(self, in_channels: int, dropout: float):
        super().__init__()
        self.modules_ = nn.Sequential(*[
            InceptionTimeModule(in_channels if index == 0 else 128)
            for index in range(3)
        ])
        self.shortcut = (nn.Identity() if in_channels == 128 else
                         nn.Sequential(
                             nn.Conv1d(in_channels, 128, kernel_size=1, bias=False),
                             nn.BatchNorm1d(128)))
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.dropout(self.modules_(x)) + self.shortcut(x))


class InceptionTime1D(BaseModel):
    """Multivariate InceptionTime-inspired classifier for inertial signals."""
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.blocks = nn.Sequential(
            InceptionResidualBlock(in_channels, dropout),
            InceptionResidualBlock(128, dropout),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.head(self.blocks(x).mean(dim=-1))


class FCKDChannelAttention(nn.Module):
    """Channel attention module used in the compact FCKD-style IMU student."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        attn = self.mlp(x).view(x.size(0), x.size(1), 1)
        return x * attn


class FCKDResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn = FCKDChannelAttention(out_channels)
        self.shortcut = (
            nn.Identity() if in_channels == out_channels else
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv2(self.conv1(x))
        x = self.attn(x)
        return x + residual


class FCKD1D(BaseModel):
    """Compact FCKD-style IMU student with residual temporal blocks and channel attention."""
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        hidden = 64
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.encoder = nn.Sequential(
            FCKDResidualBlock(hidden, hidden, dropout),
            FCKDResidualBlock(hidden, hidden * 2, dropout),
            FCKDResidualBlock(hidden * 2, hidden * 2, dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.encoder(x)
        return self.head(x)


def build_imu_model(arch: str, in_channels: int, num_classes: int,
                    dropout: float = 0.3) -> nn.Module:
    models = {
        "cnn_bilstm": CNNBiLSTM,
        "cnn_bilstm_cbam": CNNBiLSTMCBAM,
        "tcn": IMUTCN,
        "transformer": IMUTransformer,
        "xtinyhar": XTinyHAR,
        "inceptiontime": InceptionTime1D,
        "fckd": FCKD1D,
    }
    if arch not in models:
        raise ValueError(f"Unsupported IMU architecture {arch!r}; choose {tuple(models)}")
    return models[arch](in_channels=in_channels, num_classes=num_classes,
                        dropout=dropout)

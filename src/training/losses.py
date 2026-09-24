"""Custom losses."""
import torch
import torch.nn as nn


def label_smoothing_ce(eps: float = 0.1):
    return nn.CrossEntropyLoss(label_smoothing=eps)

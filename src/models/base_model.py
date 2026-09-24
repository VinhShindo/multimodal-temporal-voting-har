"""Interface chung cho mọi model."""
from abc import ABC, abstractmethod
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module, ABC):
    """Mọi model phải có forward → logits và predict_proba → softmax."""

    def predict_proba(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

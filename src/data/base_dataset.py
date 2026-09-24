"""Abstract dataset interface."""
from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseHARDataset(Dataset, ABC):
    """__getitem__ phải trả (sample, label, meta)."""

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

"""EarlyStopping + ModelCheckpoint đơn giản."""
from pathlib import Path
import torch


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = -float("inf") if mode == "max" else float("inf")
        self.counter = 0

    def step(self, value: float) -> bool:
        better = value > self.best if self.mode == "max" else value < self.best
        if better:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class ModelCheckpoint:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, model, **meta):
        torch.save({"state_dict": model.state_dict(), **meta}, self.path)

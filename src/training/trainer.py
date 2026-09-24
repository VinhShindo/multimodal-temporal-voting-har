"""Generic trainer cho cả 3 modality."""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.logger import get_logger

log = get_logger("trainer")


class Trainer:
    def __init__(self, model, optimizer, scheduler=None, device="cuda"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y, _ in tqdm(loader, desc="train"):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(-1) == y).sum().item()
            total += x.size(0)
        if self.scheduler:
            self.scheduler.step()
        return loss_sum / total, correct / total

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total, correct = 0, 0
        for x, y, _ in tqdm(loader, desc="eval"):
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            correct += (logits.argmax(-1) == y).sum().item()
            total += x.size(0)
        return correct / total

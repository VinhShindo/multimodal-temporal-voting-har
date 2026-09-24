"""Training loop cho Skeleton modality."""
from src.models.skeleton.ctrgcn import CTRGCN
from src.training.trainer import Trainer


def build_skeleton_model(cfg) -> CTRGCN:
    return CTRGCN(
        num_joints=cfg["num_joints"],
        in_channels=cfg["in_channels"],
        num_classes=cfg["num_classes"],
    )


def train_skeleton(cfg, train_loader, val_loader, device="cuda"):
    model = build_skeleton_model(cfg["model"])
    optimizer = __import__("torch").optim.SGD(
        model.parameters(), lr=cfg["train"]["lr"], momentum=0.9)
    trainer = Trainer(model, optimizer, device=device)
    for epoch in range(cfg["train"]["epochs"]):
        loss, acc = trainer.train_epoch(train_loader)
        val_acc = trainer.evaluate(val_loader)
        print(f"[SK][E{epoch}] loss={loss:.4f} acc={acc:.4f} val={val_acc:.4f}")

"""Training loop cho RGB modality."""
from src.models.rgb.videomae import VideoMAEClassifier
from src.training.trainer import Trainer


def build_rgb_model(cfg) -> VideoMAEClassifier:
    return VideoMAEClassifier(
        backbone=cfg.get("backbone", "ViT-L/16"),
        num_classes=cfg["num_classes"],
        pretrained=cfg.get("pretrained", True),
    )


def train_rgb(cfg, train_loader, val_loader, device="cuda"):
    model = build_rgb_model(cfg["model"])
    optimizer = __import__("torch").optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    trainer = Trainer(model, optimizer, device=device)
    for epoch in range(cfg["train"]["epochs"]):
        loss, acc = trainer.train_epoch(train_loader)
        val_acc = trainer.evaluate(val_loader)
        print(f"[RGB][E{epoch}] loss={loss:.4f} acc={acc:.4f} val={val_acc:.4f}")

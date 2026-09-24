"""Training loop cho IMU modality."""
from src.models.imu.cnn_bilstm import CNNBiLSTM
from src.training.trainer import Trainer


def build_imu_model(cfg) -> CNNBiLSTM:
    return CNNBiLSTM(
        in_channels=cfg["in_channels"],
        num_classes=cfg["num_classes"],
        dropout=cfg.get("dropout", 0.3),
    )


def train_imu(cfg, train_loader, val_loader, device="cuda"):
    model = build_imu_model(cfg["model"])
    optimizer = __import__("torch").optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    trainer = Trainer(model, optimizer, device=device)
    for epoch in range(cfg["train"]["epochs"]):
        loss, acc = trainer.train_epoch(train_loader)
        val_acc = trainer.evaluate(val_loader)
        print(f"[IMU][E{epoch}] loss={loss:.4f} acc={acc:.4f} val={val_acc:.4f}")

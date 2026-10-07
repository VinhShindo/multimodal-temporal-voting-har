"""
Train CNN-BiLSTM trên MMAct IMU — full logging + evaluation.

Output: outputs/runs/imu/{timestamp}/

Usage:
    python scripts/12_train_imu.py
    python scripts/12_train_imu.py --epochs 60 --tag exp01
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
import yaml
from tqdm import tqdm

from src.common.run_logger import RunLogger, Timer, load_class_names
from src.data.imu.dataset import build_loaders
from src.models.imu.cnn_bilstm import CNNBiLSTM
from src.evaluation.isolated_eval import full_evaluation


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=5.0):
    model.train()
    losses, preds_all, labels_all = [], [], []
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(loss.item() * x.size(0))
        preds_all.append(logits.argmax(-1).detach().cpu().numpy())
        labels_all.append(y.cpu().numpy())
    preds = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    return (sum(losses) / len(labels),
            accuracy_score(labels, preds),
            f1_score(labels, preds, average="macro", zero_division=0))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses, preds_all, labels_all = [], [], []
    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        losses.append(criterion(logits, y).item() * x.size(0))
        preds_all.append(logits.argmax(-1).cpu().numpy())
        labels_all.append(y.cpu().numpy())
    preds = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    return (sum(losses) / len(labels),
            accuracy_score(labels, preds),
            f1_score(labels, preds, average="macro", zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train/imu_mmact.yaml")
    ap.add_argument("--data-cfg", default="configs/data/imu_mmact.yaml")
    ap.add_argument("--classes-cfg", default="configs/classes_mmact.yaml")
    ap.add_argument("--manifest-dir", default="data/processed/MMAct/manifests")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size

    # ── Run logger ──
    run = RunLogger("imu", config=cfg, tag=args.tag)
    run.info("=" * 70)
    run.info(f"  IMU TRAINING — {run.run_name}")
    run.info("=" * 70)
    run.info(f"  Device:  {args.device}")
    run.info(f"  Model:   CNN-BiLSTM")
    run.info(f"  Epochs:  {cfg['train']['epochs']}")
    run.info(f"  Batch:   {cfg['train']['batch_size']}")
    run.info(f"  LR:      {cfg['train']['lr']}")
    run.info(f"  Run dir: {run.path}")
    run.info("")

    # ── Data ──
    run.info("[Data] Loading...")
    train_loader, val_loader = build_loaders(
        args.manifest_dir, args.data_cfg, cfg["train"]["batch_size"])
    run.info(f"[Data] train={len(train_loader.dataset)} "
             f"val={len(val_loader.dataset)}")

    # ── Model ──
    model = CNNBiLSTM(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"].get("dropout", 0.3),
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run.info(f"[Model] CNN-BiLSTM  params={n_params / 1e6:.2f}M")

    # ── Optim ──
    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg["train"].get("label_smoothing", 0.0))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"])

    epochs = cfg["train"]["epochs"]
    warmup = cfg["train"].get("warmup_epochs", 3)

    def lr_lambda(e):
        if e < warmup:
            return (e + 1) / warmup
        prog = (e - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training ──
    run.info("")
    run.info("─" * 70)
    run.info("  TRAINING")
    run.info("─" * 70)
    run.info("")

    history = []
    best_f1, best_epoch, patience = 0.0, -1, 0
    patience_max = cfg["train"]["early_stop_patience"]

    for epoch in range(epochs):
        with Timer() as t:
            tr_loss, tr_acc, tr_f1 = train_one_epoch(
                model, train_loader, optimizer, criterion, args.device)
            va_loss, va_acc, va_f1 = evaluate(
                model, val_loader, criterion, args.device)
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch, "lr": round(lr_now, 8),
            "train_loss": round(tr_loss, 6),
            "train_acc":  round(tr_acc, 6),
            "train_f1":   round(tr_f1, 6),
            "val_loss":   round(va_loss, 6),
            "val_acc":    round(va_acc, 6),
            "val_f1":     round(va_f1, 6),
            "time_s":     round(t.elapsed, 2),
        }
        history.append(row)

        run.info(f"[E{epoch:02d}] lr={lr_now:.6f} | "
                 f"tr loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} | "
                 f"va loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f} | "
                 f"{t.elapsed_str}")

        if va_f1 > best_f1:
            best_f1, best_epoch = va_f1, epoch
            patience = 0
            run.save_best_from_model(model, epoch, best_f1, name="best.pt")
            run.info(f"  ✓ Saved best (val_f1={best_f1:.4f})")
        else:
            patience += 1
            if patience >= patience_max:
                run.info(f"[EarlyStopping] Dừng ở epoch {epoch}")
                break

    run.save_history(history)
    run.info("")
    run.info(f"✓ Training done. Best val F1 = {best_f1:.4f} (epoch {best_epoch})")

    # ── Full evaluation ──
    run.info("")
    run.info("─" * 70)
    run.info("  FULL EVALUATION")
    run.info("─" * 70)

    best_ckpt = torch.load(run.path / "best.pt", map_location=args.device,
                            weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    run.info(f"  Loaded best from epoch {best_ckpt['epoch']}")

    class_names = load_class_names(args.classes_cfg)
    val_metrics = full_evaluation(
        model, train_loader, val_loader, class_names,
        out_dir=run.path, device=args.device,
        collect_train=True, log_fn=run.info)

    # ── Summary ──
    run.info("")
    run.info("=" * 70)
    run.info("  SUMMARY")
    run.info("=" * 70)
    run.info(f"  Best val F1 (macro): {best_f1:.4f}  @ epoch {best_epoch}")
    run.info(f"  Final val accuracy:  {val_metrics['accuracy']:.4f}")
    run.info(f"  Final val F1 macro:  {val_metrics['f1_macro']:.4f}")
    run.info(f"  Run dir:             {run.path}")


if __name__ == "__main__":
    main()
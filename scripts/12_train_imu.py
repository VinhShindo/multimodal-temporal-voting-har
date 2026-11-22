"""
Train CNN-BiLSTM trên MMAct IMU — full logging + evaluation.

Output: outputs/runs/imu/{timestamp}/

Usage:
    python scripts/12_train_imu.py
    python scripts/12_train_imu.py --epochs 60 --tag exp01
    python scripts/12_train_imu.py --protocol x_view --feature-set gravity_jerk \
        --arch tcn_temporal_pyramid --seed 42 --tag tcn_temporal_pyramid_seed42
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import yaml
from tqdm import tqdm

from src.common.run_logger import RunLogger, Timer, load_class_names
from src.common.manifest_paths import MM_ACT_PROTOCOLS
from src.data.imu.dataset import DERIVED_FEATURE_CHANNELS, build_loaders
from src.models.imu.architectures import build_imu_model
from src.evaluation.isolated_eval import full_evaluation


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=5.0,
                    amp=False, scaler=None, show_progress=True):
    model.train()
    losses, preds_all, labels_all = [], [], []
    for x, y in tqdm(loader, desc="train", leave=False,
                      disable=not show_progress):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
        with torch.autocast(device_type=dev_type, dtype=torch.float16,
                            enabled=amp):
            logits = model(x)
        loss = criterion(logits.float(), y)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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
def evaluate(model, loader, criterion, device, amp=False, show_progress=True):
    model.eval()
    losses, preds_all, labels_all = [], [], []
    for x, y in tqdm(loader, desc="val", leave=False,
                      disable=not show_progress):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
        with torch.autocast(device_type=dev_type, dtype=torch.float16,
                            enabled=amp):
            if x.ndim == 4:
                batch_size, num_crops = x.shape[:2]
                logits = model(x.flatten(0, 1)).reshape(
                    batch_size, num_crops, -1)
                probs = torch.softmax(logits.float(), dim=-1).mean(dim=1)
                loss = criterion(probs.clamp_min(1e-8).log(), y)
                preds = probs.argmax(-1)
            else:
                logits = model(x)
                loss = criterion(logits.float(), y)
                preds = logits.argmax(-1)
        losses.append(loss.item() * x.size(0))
        preds_all.append(preds.cpu().numpy())
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
    ap.add_argument("--manifest-dir", default=None,
                    help="Override paths.manifest_dir from the data config")
    ap.add_argument("--protocol", choices=MM_ACT_PROTOCOLS, default=None)
    ap.add_argument("--arch", choices=["cnn_bilstm", "cnn_bilstm_cbam", "tcn",
                                        "tcn_temporal_pyramid", "transformer",
                                        "xtinyhar", "inceptiontime", "fckd"],
                    default=None)
    ap.add_argument("--feature-set", choices=["raw", "gravity_jerk", "gravity_jerk_plus"],
                    default=None)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--quiet-progress", action="store_true",
                    help="suppress per-batch tqdm output")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    with open(args.data_cfg, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    protocol = args.protocol or cfg.get(
        "protocol", data_cfg["paths"].get("protocol", "x_view")
    )
    cfg["protocol"] = protocol
    arch = args.arch or cfg["model"].get("arch", "cnn_bilstm")
    cfg["model"]["arch"] = arch
    feature_set = args.feature_set or cfg.get("feature_set", "raw")
    cfg["feature_set"] = feature_set
    channel_counts = {
        "raw": 3,
        "gravity_jerk": DERIVED_FEATURE_CHANNELS,
        "gravity_jerk_plus": 16,
    }
    cfg["model"]["in_channels"] = channel_counts[feature_set]
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    cfg["seed"] = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    num_workers = (args.num_workers if args.num_workers is not None
                   else cfg["train"].get("num_workers", 4))
    amp = (cfg["train"].get("amp", False) if args.amp is None else args.amp)
    amp = bool(amp) and str(args.device).startswith("cuda")
    if amp:
        torch.backends.cudnn.benchmark = True

    # ── Run logger ──
    run = RunLogger("imu", config=cfg, tag=args.tag)
    run.info("=" * 70)
    run.info(f"  IMU TRAINING — {run.run_name}")
    run.info("=" * 70)
    run.info(f"  Device:  {args.device}")
    run.info(f"  Model:   {arch}")
    run.info(f"  Features: {feature_set} ({cfg['model']['in_channels']} channels)")
    run.info(f"  Protocol: {protocol}")
    run.info(f"  Seed:    {seed}")
    run.info(f"  Epochs:  {cfg['train']['epochs']}")
    run.info(f"  Batch:   {cfg['train']['batch_size']}")
    run.info(f"  LR:      {cfg['train']['lr']}")
    run.info(f"  Run dir: {run.path}")
    run.info("")

    # ── Data ──
    run.info("[Data] Loading...")
    train_loader, val_loader = build_loaders(
        args.manifest_dir, args.data_cfg, cfg["train"]["batch_size"],
        num_workers=num_workers, protocol=protocol, feature_set=feature_set)
    run.info(f"[Data] train={len(train_loader.dataset)} "
             f"val={len(val_loader.dataset)}")

    # ── Model ──
    model = build_imu_model(
        arch=arch,
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"].get("dropout", 0.3),
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run.info(f"[Model] {arch}  params={n_params / 1e6:.2f}M")

    # ── Optim ──
    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg["train"].get("label_smoothing", 0.0))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

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
                model, train_loader, optimizer, criterion, args.device,
                amp=amp, scaler=scaler,
                show_progress=not args.quiet_progress)
            va_loss, va_acc, va_f1 = evaluate(
                model, val_loader, criterion, args.device, amp=amp,
                show_progress=not args.quiet_progress)
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
            run.save_best_from_model(model, epoch, best_f1, arch=arch,
                                     name="best.pt")
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

    # Training uses a sampler and stochastic crops.  Switch the same dataset
    # to deterministic evaluation mode and iterate it in manifest order so
    # train predictions can be aligned safely for attention fusion.
    train_eval_dataset = train_loader.dataset
    train_eval_dataset.training = False
    train_eval_loader = DataLoader(
        train_eval_dataset, batch_size=train_loader.batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
        drop_last=False, persistent_workers=num_workers > 0)
    train_clip_ids = train_eval_dataset.df["clip_id"].astype(str).tolist()

    class_names = load_class_names(args.classes_cfg)
    val_metrics = full_evaluation(
        model, train_eval_loader, val_loader, class_names,
        out_dir=run.path, device=args.device,
        collect_train=True, train_clip_ids=train_clip_ids,
        log_fn=run.info)

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

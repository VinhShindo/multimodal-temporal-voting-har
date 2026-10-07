"""
Train video model trên MMAct RGB — full logging + evaluation + RESUME support.

Output: outputs/runs/rgb/{timestamp}/

Usage:
    python scripts/11_train_rgb.py --quick-test
    python scripts/11_train_rgb.py --arch r2plus1d_18 --tag exp01
    python scripts/11_train_rgb.py --resume outputs/runs/rgb/20260927_235612_mc3_18/best.pt
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
from src.data.rgb.dataset import build_loaders
from src.models.rgb.video_backbone import build_video_model, SUPPORTED_ARCHS
from src.evaluation.isolated_eval import full_evaluation


# ============================================================
# Train / eval
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device,
                    grad_clip=5.0):
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


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train/rgb_mmact.yaml")
    ap.add_argument("--data-cfg", default="configs/data/rgb_mmact.yaml")
    ap.add_argument("--classes-cfg", default="configs/classes_mmact.yaml")
    ap.add_argument("--manifest-dir", default="data/processed/MMAct/manifests")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--arch", default=None, choices=SUPPORTED_ARCHS)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--quick-test", action="store_true",
                    help="100 samples, 1 epoch, batch=2, mc3_18")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=42)
    # ★ NEW
    ap.add_argument("--resume", type=str, default=None,
                    help="Path tới checkpoint best.pt để resume training")
    args = ap.parse_args()

    # ── Set seed ──
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ── Quick-test overrides ──
    if args.quick_test:
        args.epochs = 1
        args.batch_size = 2
        args.arch = args.arch or "mc3_18"
        args.num_workers = 0
        args.max_samples = 100
        args.no_pretrained = True
        args.tag = args.tag or "quicktest"

    # ── Load config ──
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    arch = args.arch or cfg["model"].get("arch", "mc3_18")
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size

    # ── RESUME: load checkpoint info ──
    resume_ckpt = None
    start_epoch = 0
    best_f1_init = 0.0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"✗ Không tìm thấy checkpoint: {resume_path}")
            return
        print(f"[Resume] Loading {resume_path}...")
        resume_ckpt = torch.load(resume_path, map_location=args.device,
                                  weights_only=False)
        # Nếu user không đổi arch → dùng arch trong checkpoint
        if args.arch is None:
            arch = resume_ckpt.get("arch", arch)
        best_f1_init = float(resume_ckpt.get("best_f1", 0.0))
        # Resume từ epoch kế tiếp của best, KHÔNG phải epoch cuối
        # (vì best.pt chỉ lưu khi có F1 mới tốt nhất)
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        print(f"[Resume] best_f1={best_f1_init:.4f}, "
              f"best_epoch={resume_ckpt.get('epoch')}, "
              f"start from epoch {start_epoch}")

    # ── Run logger ──
    tag = args.tag or arch
    if args.resume:
        tag = f"{tag}_resume_from_e{start_epoch}"
    run = RunLogger("rgb", config=cfg, tag=tag)
    run.info("=" * 70)
    run.info(f"  RGB TRAINING — {run.run_name}")
    run.info("=" * 70)
    run.info(f"  Device:      {args.device}")
    run.info(f"  Arch:        {arch}")
    run.info(f"  Pretrained:  {not args.no_pretrained}")
    run.info(f"  Epochs:      {cfg['train']['epochs']}")
    run.info(f"  Batch:       {cfg['train']['batch_size']}")
    run.info(f"  Max samples: {args.max_samples or 'all'}")
    run.info(f"  Seed:        {args.seed}")
    if args.resume:
        run.info(f"  ★ Resume from: {args.resume}")
        run.info(f"  ★ Start epoch: {start_epoch}  "
                 f"(init best_f1={best_f1_init:.4f})")
    run.info(f"  Run dir:     {run.path}")
    run.info("")

    # ── Data ──
    run.info("[Data] Loading...")
    train_loader, val_loader = build_loaders(
        args.manifest_dir, args.data_cfg, cfg["train"]["batch_size"],
        num_workers=args.num_workers, max_samples=args.max_samples)
    run.info(f"[Data] train={len(train_loader.dataset)} "
             f"val={len(val_loader.dataset)}")

    # ── Model ──
    model = build_video_model(
        arch=arch,
        num_classes=cfg["model"]["num_classes"],
        pretrained=not args.no_pretrained,
        dropout=cfg["model"].get("dropout", 0.3),
    ).to(args.device)

    # ★ Load weights nếu resume
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state"])
        run.info(f"[Model] Loaded weights từ checkpoint")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run.info(f"[Model] {arch}  params={n_params / 1e6:.2f}M")

    # ── Optim ──
    label_smooth = cfg["train"].get("label_smoothing", 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if any(k in name for k in ("fc", "classifier", "head")):
            head_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg["train"]["lr"]},
        {"params": head_params,     "lr": cfg["train"]["lr"] * 5},
    ], weight_decay=cfg["train"]["weight_decay"])

    epochs = cfg["train"]["epochs"]
    warmup = min(cfg["train"].get("warmup_epochs", 2), max(1, epochs // 2))

    def lr_lambda(e):
        if e < warmup:
            return (e + 1) / warmup
        prog = (e - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ★ Fast-forward scheduler đến start_epoch (quan trọng khi resume)
    if start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()
        run.info(f"[Resume] Scheduler fast-forwarded to epoch {start_epoch}, "
                 f"lr={optimizer.param_groups[0]['lr']:.6f}")

    # ── Training ──
    run.info("")
    run.info("─" * 70)
    run.info("  TRAINING")
    run.info("─" * 70)
    run.info("")

    history = []
    best_f1 = best_f1_init
    best_epoch = start_epoch - 1 if start_epoch > 0 else -1
    patience = 0
    patience_max = cfg["train"]["early_stop_patience"]

    for epoch in range(start_epoch, epochs):
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
            if not args.quick_test:
                run.save_best_from_model(
                    model, epoch, best_f1,
                    arch=arch, name="best.pt")
                # ★ Cũng lưu optimizer + scheduler state để resume tốt hơn lần sau
                torch.save({
                    "epoch": epoch,
                    "best_f1": best_f1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "arch": arch,
                    "config": cfg,
                }, run.path / "best_full.pt")
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
    if not args.quick_test:
        run.info("")
        run.info("─" * 70)
        run.info("  FULL EVALUATION")
        run.info("─" * 70)

        best_ckpt_path = run.path / "best.pt"
        if best_ckpt_path.exists():
            best_ckpt = torch.load(best_ckpt_path, map_location=args.device,
                                    weights_only=False)
            model.load_state_dict(best_ckpt["model_state"])
            run.info(f"  Loaded best from epoch {best_ckpt['epoch']}")
        else:
            run.info(f"  Không có best.pt — dùng model cuối")

        class_names = load_class_names(args.classes_cfg)
        val_metrics = full_evaluation(
            model, train_loader, val_loader, class_names,
            out_dir=run.path, device=args.device,
            collect_train=False,
            log_fn=run.info)

        run.info("")
        run.info("=" * 70)
        run.info("  SUMMARY")
        run.info("=" * 70)
        run.info(f"  Best val F1 (macro): {best_f1:.4f}  @ epoch {best_epoch}")
        run.info(f"  Final val accuracy:  {val_metrics['accuracy']:.4f}")
        run.info(f"  Run dir:             {run.path}")
    else:
        run.info("")
        run.info("✓ QUICK TEST PASSED — Pipeline hoạt động!")
        run.info(f"  Run dir: {run.path}")


if __name__ == "__main__":
    main()
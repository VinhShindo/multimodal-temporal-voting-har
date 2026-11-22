"""
Train CTR-GCN trên MMAct Skeleton — full logging + evaluation.

Kiến trúc (--arch):
  ctrgcn       baseline 10 layer + GAP          (stream: joint / joint_bone / joint_bone_motion / rich)
  ctrgcn_head  + PartTransformerHead            (stream: rich)            ← ablation R2

Ablation gợi ý (cùng seed, cùng wd):
  R0  python scripts/10_train_skeleton.py --arch ctrgcn      --stream joint_bone_motion --tag R0
  R1  python scripts/10_train_skeleton.py --arch ctrgcn      --stream rich              --tag R1
  R2  python scripts/10_train_skeleton.py --arch ctrgcn_head --stream rich              --tag R2

Output: outputs/runs/skeleton/{timestamp}_{tag}/ (config, log, history, best.pt, metrics, confusion matrix...)
"""
import argparse
import copy
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
from src.data.skeleton.dataset import build_loaders, IN_CHANNELS
from src.models.skeleton import BarlowTwinsSkeleton, build_model
from src.evaluation.isolated_eval import full_evaluation


DEFAULT_STREAM = {
    "stgcn": "joint_bone_motion", "ms_g3d": "joint_bone_motion",
    "ctrgcn": "joint_bone_motion", "sta_gcn": "joint_bone_motion",
    "barlow_twins": "joint_bone_motion", "ctrgcn_head": "rich",
}


# ============================================================
# EMA
# ============================================================
class EMA:
    """EMA của trọng số + buffer (BN stats)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.n = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for e, m in zip(self.module.state_dict().values(),
                        model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)


# ============================================================
# Optimizer: không weight decay cho BN / LN / bias / alpha / A / cls / pos_*
# ============================================================
def _no_decay(name: str, p: torch.Tensor) -> bool:
    return (p.ndim <= 1 or name.endswith(".bias") or name.endswith(".alpha")
            or name.endswith(".A") or name.endswith("cls") or ".pos_" in name)


def build_optimizer(model, tcfg: dict):
    lr = tcfg["lr"]
    wd = tcfg["weight_decay"]
    head_mult = tcfg.get("head_lr_mult", 1.0)

    groups = {"body_decay": [], "body_nodecay": [],
              "head_decay": [], "head_nodecay": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        head = n.startswith("fc.")
        key = ("head_" if head else "body_") + ("nodecay" if _no_decay(n, p) else "decay")
        groups[key].append(p)

    param_groups = []
    for key, params in groups.items():
        if not params:
            continue
        param_groups.append({
            "params": params,
            "lr": lr * (head_mult if key.startswith("head") else 1.0),
            "weight_decay": 0.0 if key.endswith("nodecay") else wd})

    if tcfg.get("optimizer", "adamw") == "sgd":
        return torch.optim.SGD(param_groups, lr=lr, momentum=0.9, nesterov=True)
    return torch.optim.AdamW(param_groups, lr=lr)


# ============================================================
# Train / eval
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device,
                    grad_clip=3.0, ema=None, scaler=None, amp=False,
                    show_progress=True):
    model.train()
    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
    losses, preds_all, labels_all = [], [], []
    for x, y in tqdm(loader, desc="train", leave=False,
                      disable=not show_progress):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=amp):
            if isinstance(model, BarlowTwinsSkeleton):
                logits, projection1 = model(x, return_projection=True)
                _, projection2 = model(model.augment(x), return_projection=True)
                loss = criterion(logits.float(), y) + 0.01 * model.loss(
                    projection1, projection2)
            else:
                logits = model(x)
                loss = criterion(logits.float(), y)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if ema is not None:
            ema.update(model)
        losses.append(loss.item() * x.size(0))
        preds_all.append(logits.argmax(-1).detach().cpu().numpy())
        labels_all.append(y.cpu().numpy())
    preds = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    return (sum(losses) / len(labels),
            accuracy_score(labels, preds),
            f1_score(labels, preds, average="macro", zero_division=0))


@torch.no_grad()
def evaluate(model, loader, criterion, device, show_progress=True):
    model.eval()
    losses, preds_all, labels_all = [], [], []
    for x, y in tqdm(loader, desc="val", leave=False,
                      disable=not show_progress):
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train/skeleton_mmact.yaml")
    ap.add_argument("--data-cfg", default="configs/data/skeleton_mmact.yaml")
    ap.add_argument("--classes-cfg", default="configs/classes_mmact.yaml")
    ap.add_argument("--manifest-dir", default=None,
                    help="Override paths.manifest_dir from the data config")
    ap.add_argument("--protocol", choices=MM_ACT_PROTOCOLS, default=None)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--arch", default=None, choices=[
        "stgcn", "ms_g3d", "ctrgcn", "sta_gcn", "barlow_twins",
        "ctrgcn_head"])
    ap.add_argument("--stream", default=None, choices=list(IN_CHANNELS.keys()))
    ap.add_argument("--aug", default=None, choices=["none", "light", "strong", "view"])
    ap.add_argument("--optim", default=None, choices=["adamw", "sgd"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--grad-clip", type=float, default=None)
    ap.add_argument("--ema", type=float, default=None, help="EMA decay (0 = tắt)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                    help="mixed precision fp16")
    ap.add_argument("--no-cache", action="store_true", help="không cache NPY vào RAM")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--quiet-progress", action="store_true",
                    help="suppress per-batch tqdm output")
    args = ap.parse_args()

    set_seed(args.seed)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(args.data_cfg, encoding="utf-8") as f:
        dcfg = yaml.safe_load(f)
    protocol = args.protocol or dcfg["paths"].get("protocol", "x_view")
    cfg["protocol"] = protocol
    tcfg, mcfg = cfg["train"], cfg["model"]

    # CLI override
    if args.epochs is not None:     tcfg["epochs"] = args.epochs
    if args.lr is not None:         tcfg["lr"] = args.lr
    if args.batch_size is not None: tcfg["batch_size"] = args.batch_size
    if args.wd is not None:         tcfg["weight_decay"] = args.wd
    if args.warmup is not None:     tcfg["warmup_epochs"] = args.warmup
    if args.grad_clip is not None:  tcfg["grad_clip"] = args.grad_clip
    if args.ema is not None:        tcfg["ema_decay"] = args.ema
    if args.optim is not None:      tcfg["optimizer"] = args.optim
    if args.aug is not None:        tcfg["aug_level"] = args.aug

    arch = args.arch or mcfg.get("arch", "ctrgcn_head")
    stream = args.stream or DEFAULT_STREAM[arch]
    mcfg["arch"] = arch
    mcfg["stream"] = stream
    in_ch = IN_CHANNELS[stream]
    target_len = dcfg["sequence"]["target_len"]

    aug_level = tcfg.get("aug_level", "light")
    grad_clip = tcfg.get("grad_clip", 3.0)
    ema_decay = tcfg.get("ema_decay", 0.999)
    num_workers = (args.num_workers if args.num_workers is not None
                   else tcfg.get("num_workers", 4))
    amp = (tcfg.get("amp", False) if args.amp is None else args.amp)
    amp = bool(amp) and str(args.device).startswith("cuda")
    if amp:
        torch.backends.cudnn.benchmark = True

    tag = args.tag or f"{arch}_{stream}"
    run = RunLogger("skeleton", config=cfg, tag=tag)
    run.info("=" * 70)
    run.info(f"  SKELETON TRAINING — {run.run_name}")
    run.info("=" * 70)
    run.info(f"  Device:     {args.device}   AMP={amp}   seed={args.seed}")
    run.info(f"  Arch:       {arch}   Stream: {stream}  (in_channels={in_ch})")
    run.info(f"  target_len: {target_len}   Aug: {aug_level}")
    run.info(f"  Epochs:     {tcfg['epochs']}   Batch: {tcfg['batch_size']}")
    run.info(f"  Optim:      {tcfg.get('optimizer', 'adamw')}  lr={tcfg['lr']}  "
             f"wd={tcfg['weight_decay']}  grad_clip={grad_clip}  ema={ema_decay}")
    run.info(f"  Run dir:    {run.path}")
    run.info("")

    # ── Data ──
    run.info("[Data] Loading...")
    train_loader, val_loader = build_loaders(
        args.manifest_dir, args.data_cfg, tcfg["batch_size"],
        num_workers=num_workers, stream=stream, aug_level=aug_level,
        cache=not args.no_cache, protocol=protocol)
    run.info(f"[Data] train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    # ── Model ──
    model = build_model(
        arch=arch,
        num_classes=mcfg["num_classes"],
        num_joints=mcfg["num_joints"],
        in_channels=in_ch,
        target_len=target_len,
        dropout=mcfg.get("dropout", 0.3 if arch == "ctrgcn" else 0.4),
        part_drop=mcfg.get("part_drop", 0.1),
        drop_path=mcfg.get("drop_path", 0.1),
        tf_depth=mcfg.get("tf_depth", 2),
        tf_heads=mcfg.get("tf_heads", 4),
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run.info(f"[Model] {arch}  params={n_params / 1e6:.2f}M  in_ch={in_ch}")

    ema = EMA(model, ema_decay) if ema_decay and ema_decay > 0 else None
    eval_model = ema.module if ema is not None else model

    # ── Optim ──
    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg.get("label_smoothing", 0.1))
    optimizer = build_optimizer(model, tcfg)
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if amp else None

    epochs = tcfg["epochs"]
    warmup = tcfg.get("warmup_epochs", 3)

    def lr_lambda(e):
        if e < warmup:
            return (e + 1) / warmup
        prog = (e - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training loop ──
    run.info("")
    run.info("─" * 70)
    run.info("  TRAINING")
    run.info("─" * 70)
    run.info("")

    history = []
    best_f1, best_epoch, patience = 0.0, -1, 0
    patience_max = tcfg["early_stop_patience"]

    for epoch in range(epochs):
        with Timer() as t:
            tr_loss, tr_acc, tr_f1 = train_one_epoch(
                model, train_loader, optimizer, criterion, args.device,
                grad_clip=grad_clip, ema=ema, scaler=scaler, amp=amp,
                show_progress=not args.quiet_progress)
            va_loss, va_acc, va_f1 = evaluate(
                eval_model, val_loader, criterion, args.device,
                show_progress=not args.quiet_progress)
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch, "lr": round(lr_now, 8),
            "train_loss": round(tr_loss, 6), "train_acc": round(tr_acc, 6),
            "train_f1": round(tr_f1, 6),
            "val_loss": round(va_loss, 6), "val_acc": round(va_acc, 6),
            "val_f1": round(va_f1, 6), "time_s": round(t.elapsed, 2)})

        run.info(f"[E{epoch:02d}] lr={lr_now:.6f} | "
                 f"tr loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} | "
                 f"va loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f} | "
                 f"{t.elapsed_str}")

        if va_f1 > best_f1:
            best_f1, best_epoch = va_f1, epoch
            patience = 0
            run.save_best_from_model(
                eval_model, epoch, best_f1,
                extra={"arch": arch, "stream": stream, "in_channels": in_ch,
                       "target_len": target_len, "ema": ema is not None},
                name="best.pt")
            run.info(f"  ✓ Saved best (val_f1={best_f1:.4f})")
        else:
            patience += 1
            if patience >= patience_max:
                run.info(f"[EarlyStopping] Dừng ở epoch {epoch}")
                break

    run.save_history(history)
    last5 = np.mean([h["val_f1"] for h in history[-5:]])
    run.info("")
    run.info(f"✓ Training done. Best val F1 = {best_f1:.4f} (epoch {best_epoch})  "
             f"| mean val F1 của 5 epoch cuối = {last5:.4f}")

    # ── Load best + full evaluation ──
    run.info("")
    run.info("─" * 70)
    run.info("  FULL EVALUATION")
    run.info("─" * 70)

    best_ckpt = torch.load(run.path / "best.pt", map_location=args.device,
                           weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    run.info(f"  Loaded best from epoch {best_ckpt['epoch']} "
             f"(f1={best_ckpt['best_f1']:.4f})")

    # The training loader uses a weighted sampler, augmentation and drop_last.
    # Reuse its dataset in deterministic manifest order for aligned
    # train_probs, which is required by learned attention fusion.
    train_eval_dataset = train_loader.dataset
    train_eval_dataset.augment = False
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

    run.info("")
    run.info("=" * 70)
    run.info("  SUMMARY")
    run.info("=" * 70)
    run.info(f"  Arch / stream:       {arch} / {stream}")
    run.info(f"  Best val F1 (macro): {best_f1:.4f}  @ epoch {best_epoch}")
    run.info(f"  Mean val F1 last 5:  {last5:.4f}   (dùng số này để so ablation)")
    run.info(f"  Final val accuracy:  {val_metrics['accuracy']:.4f}")
    run.info(f"  Final val F1 macro:  {val_metrics['f1_macro']:.4f}")
    run.info(f"  Run dir:             {run.path}")
    run.info("")
    run.info("  Files:")
    for f in sorted(run.path.rglob("*")):
        if f.is_file():
            run.info(f"    · {f.relative_to(run.path)}")


if __name__ == "__main__":
    main()

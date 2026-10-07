"""
Evaluation đầy đủ cho isolated recognition.
- Full metrics (acc, F1 macro/weighted, precision, recall)
- Per-class F1 → CSV
- Classification report (sklearn) → TXT
- Confusion matrix: counts + normalized → PNG
- Save probabilities NPY để dùng cho fusion
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # backend không cần display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)


# ============================================================
# Prediction
# ============================================================
def collect_predictions(model, loader, device: str = "cuda"):
    """Chạy model → trả (probs, preds, labels)."""
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(np.asarray(y))
    probs = np.concatenate(all_probs, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return probs, preds, labels


# ============================================================
# Metrics
# ============================================================
def compute_metrics(labels, preds, num_classes: int) -> dict:
    class_ids = list(range(num_classes))
    return {
        "accuracy":        float(accuracy_score(labels, preds)),
        "f1_macro":        float(f1_score(labels, preds, average="macro",
                                          labels=class_ids, zero_division=0)),
        "f1_weighted":     float(f1_score(labels, preds, average="weighted",
                                          labels=class_ids, zero_division=0)),
        "precision_macro": float(precision_score(labels, preds, average="macro",
                                                 labels=class_ids, zero_division=0)),
        "recall_macro":    float(recall_score(labels, preds, average="macro",
                                              labels=class_ids, zero_division=0)),
        "num_samples":     int(len(labels)),
    }


def save_per_class_report(labels, preds, class_names, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_ids = list(range(len(class_names)))

    f1_pc = f1_score(labels, preds, labels=class_ids, average=None,
                     zero_division=0)
    pr_pc = precision_score(labels, preds, labels=class_ids, average=None,
                            zero_division=0)
    rc_pc = recall_score(labels, preds, labels=class_ids, average=None,
                         zero_division=0)
    support = np.bincount(labels, minlength=len(class_names))

    df = pd.DataFrame({
        "class_id":   class_ids,
        "class_name": class_names,
        "precision":  pr_pc,
        "recall":     rc_pc,
        "f1":         f1_pc,
        "support":    support,
    })
    df.to_csv(out_dir / "per_class_f1.csv", index=False, encoding="utf-8")

    report = classification_report(
        labels, preds, labels=class_ids,
        target_names=class_names, zero_division=0, digits=4,
    )
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(labels, preds, labels=class_ids)
    np.save(out_dir / "confusion_matrix.npy", cm)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        out_dir / "confusion_matrix.csv", encoding="utf-8")

    return df, cm


# ============================================================
# Confusion matrix plot
# ============================================================
def plot_confusion_matrix(cm: np.ndarray, class_names: list, out_path: Path,
                          normalize: bool = False, figsize=(16, 14)):
    if normalize:
        denom = np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        cm_plot = cm.astype(float) / denom
        title = "Confusion Matrix (Normalized)"
        vmax = 1.0
    else:
        cm_plot = cm
        title = "Confusion Matrix (Counts)"
        vmax = max(cm.max(), 1)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues",
                   vmin=0, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)

    # Annotate
    thresh = vmax / 2.0
    for i in range(n):
        for j in range(n):
            v = cm_plot[i, j]
            if v == 0:
                continue
            txt = (f"{v:.2f}" if normalize else f"{int(v)}")
            if normalize and v < 0.01:
                continue
            color = "white" if v > thresh else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def collect_predictions(model, loader, device: str = "cuda",
                         tta: bool = False):
    """
    Chạy model → trả (probs, preds, labels).

    Args:
        tta: nếu True, áp dụng Test-Time Augmentation (flip ngang skeleton).
             Chỉ dùng cho model skeleton — không dùng cho IMU/RGB.
    """
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch
            x = x.to(device, non_blocking=True)

            logits = model(x)

            if tta:
                # ★ TTA: flip trục x (dim channel) và swap L/R joints
                x_flip = flip_skeleton_batch(x)
                logits_flip = model(x_flip)
                # Average probs
                probs = (torch.softmax(logits, -1)
                         + torch.softmax(logits_flip, -1)) / 2
            else:
                probs = torch.softmax(logits, -1)

            probs = probs.cpu().numpy()
            preds = probs.argmax(axis=-1)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(np.asarray(y))

    probs = np.concatenate(all_probs, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return probs, preds, labels


def flip_skeleton_batch(x):
    """
    Flip skeleton theo chiều ngang.
    x: (B, C, T, V) — channel đầu tiên là x, cần đảo dấu.
    Đồng thời swap L/R joints (theo COCO pairs).
    """
    x = x.clone()
    # Đảo dấu channel x (giả sử layout là joint(x,y) hoặc joint(x,y)+bone(x,y)+motion(x,y))
    # → đảo các channel chẵn (0, 2, 4, ...)
    C = x.shape[1]
    for c in range(0, C, 2):
        x[:, c] *= -1.0

    # Swap L/R joints
    COCO_FLIP_PAIRS = [
        (1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
        (11, 12), (13, 14), (15, 16),
    ]
    for l, r in COCO_FLIP_PAIRS:
        x[:, :, :, [l, r]] = x[:, :, :, [r, l]]
    return x

# ============================================================
# Full evaluation
# ============================================================
def full_evaluation(model, train_loader, val_loader, class_names: list,
                    out_dir, device: str = "cuda",
                    collect_train: bool = True,
                    tta: bool = False,              # ★ NEW
                    log_fn=None) -> dict:
    def log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(class_names)

    # Val
    log(f"  Collecting val predictions (TTA={tta})...")
    val_probs, val_preds, val_labels = collect_predictions(
        model, val_loader, device, tta=tta)
    val_metrics = compute_metrics(val_labels, val_preds, num_classes)

    np.save(out_dir / "val_probs.npy", val_probs.astype(np.float32))
    np.save(out_dir / "val_labels.npy", val_labels.astype(np.int64))

    _, cm_val = save_per_class_report(val_labels, val_preds, class_names, out_dir)
    plot_confusion_matrix(cm_val, class_names,
                          out_dir / "confusion_matrix.png", normalize=False)
    plot_confusion_matrix(cm_val, class_names,
                          out_dir / "confusion_matrix_normalized.png",
                          normalize=True)

    train_metrics = None
    if collect_train:
        log("  Collecting train predictions...")
        tr_probs, tr_preds, tr_labels = collect_predictions(
            model, train_loader, device, tta=False)
        train_metrics = compute_metrics(tr_labels, tr_preds, num_classes)
        np.save(out_dir / "train_probs.npy", tr_probs.astype(np.float32))
        np.save(out_dir / "train_labels.npy", tr_labels.astype(np.int64))

        tr_dir = out_dir / "train"
        tr_dir.mkdir(exist_ok=True)
        _, cm_tr = save_per_class_report(tr_labels, tr_preds, class_names, tr_dir)
        plot_confusion_matrix(cm_tr, class_names,
                              tr_dir / "confusion_matrix.png", normalize=False)

    out = {"val": val_metrics, "train": train_metrics, "tta": tta}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    log(f"  Val:   acc={val_metrics['accuracy']:.4f}  "
        f"F1_macro={val_metrics['f1_macro']:.4f}")
    if train_metrics:
        log(f"  Train: acc={train_metrics['accuracy']:.4f}  "
            f"F1_macro={train_metrics['f1_macro']:.4f}")

    return val_metrics
"""
Ensemble skeleton — average val_probs từ 3 model stream riêng.

Điều kiện: đã train 3 model với streams:
    - "joint"           → outputs/runs/skeleton/{ts}/val_probs.npy
    - "bone"            → outputs/runs/skeleton/{ts}/val_probs.npy
    - "motion"          → outputs/runs/skeleton/{ts}/val_probs.npy

Usage:
    python scripts/15_ensemble_skeleton.py --runs \
        outputs/runs/skeleton/20261001_joint \
        outputs/runs/skeleton/20261001_bone \
        outputs/runs/skeleton/20261001_motion
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.common.run_logger import RunLogger, load_class_names
from src.evaluation.isolated_eval import (
    compute_metrics, save_per_class_report, plot_confusion_matrix,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="List đường dẫn tới run folders")
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="Trọng số cho mỗi run (mặc định: bằng nhau)")
    ap.add_argument("--classes-cfg", default="configs/classes_mmact.yaml")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    # ── Load class names ──
    class_names = load_class_names(args.classes_cfg)
    num_classes = len(class_names)

    # ── Run logger ──
    run = RunLogger("skeleton_ensemble", tag=args.tag)
    run.info("=" * 70)
    run.info(f"  SKELETON ENSEMBLE — {run.run_name}")
    run.info("=" * 70)
    run.info(f"  Runs: {len(args.runs)}")

    # ── Load probs từ các run ──
    all_probs = []
    labels = None
    run_names = []

    for rp in args.runs:
        rd = Path(rp)
        if not (rd / "val_probs.npy").exists():
            run.info(f"  ✗ Không tìm thấy val_probs.npy trong {rd}")
            continue
        probs = np.load(rd / "val_probs.npy")
        lbl = np.load(rd / "val_labels.npy")
        if labels is None:
            labels = lbl
        elif not np.array_equal(labels, lbl):
            run.info(f"  ✗ Labels không khớp giữa {rd}")
            return
        all_probs.append(probs)
        run_names.append(rd.name)
        run.info(f"  ✓ Load {rd.name}  shape={probs.shape}  "
                 f"acc={(probs.argmax(-1) == labels).mean():.4f}")

    if not all_probs:
        run.info("  ✗ Không load được run nào.")
        return

    # ── Weights ──
    if args.weights is None:
        weights = [1.0 / len(all_probs)] * len(all_probs)
    else:
        if len(args.weights) != len(all_probs):
            run.info(f"  ✗ Số weights ({len(args.weights)}) ≠ số runs ({len(all_probs)})")
            return
        total = sum(args.weights)
        weights = [w / total for w in args.weights]

    run.info("")
    run.info(f"  Weights: {dict(zip(run_names, [round(w, 3) for w in weights]))}")

    # ── Ensemble ──
    fused = np.zeros_like(all_probs[0])
    for w, p in zip(weights, all_probs):
        fused += w * p

    preds = fused.argmax(-1)
    metrics = compute_metrics(labels, preds, num_classes)

    run.info("")
    run.info("─" * 70)
    run.info("  ENSEMBLE RESULT")
    run.info("─" * 70)
    run.info(f"  Accuracy:    {metrics['accuracy']:.4f}")
    run.info(f"  F1 macro:    {metrics['f1_macro']:.4f}")
    run.info(f"  F1 weighted: {metrics['f1_weighted']:.4f}")

    # ── Save artifacts ──
    np.save(run.path / "val_probs.npy", fused.astype(np.float32))
    np.save(run.path / "val_labels.npy", labels.astype(np.int64))

    _, cm = save_per_class_report(labels, preds, class_names, run.path)
    plot_confusion_matrix(cm, class_names,
                          run.path / "confusion_matrix.png", normalize=False)
    plot_confusion_matrix(cm, class_names,
                          run.path / "confusion_matrix_normalized.png",
                          normalize=True)

    with open(run.path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"val": metrics, "runs": run_names,
                   "weights": weights}, f, indent=2)

    run.info("")
    run.info(f"  Run dir: {run.path}")


if __name__ == "__main__":
    main()
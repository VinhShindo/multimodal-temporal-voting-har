"""
Benchmark fusion voting — 7 tổ hợp × 3 phương pháp.

Tổ hợp:
  E1: skeleton
  E2: imu
  E3: rgb
  E4: skeleton + imu
  E5: skeleton + rgb
  E6: imu + rgb
  E7: skeleton + imu + rgb

Mỗi tổ hợp chạy 3 phương pháp: hard / average / weighted (grid search).

Usage:
    python scripts/40_benchmark_fusion.py
    python scripts/40_benchmark_fusion.py --manifest-dir data/processed/MMAct/manifests
    python scripts/40_benchmark_fusion.py --skip-rgb
"""
import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import yaml

from src.common.run_logger import RunLogger, load_class_names
from src.fusion.align import align_all, find_latest_run
from src.fusion.voting import fuse_probs
from src.fusion.weights import grid_search_weights
from src.evaluation.isolated_eval import (
    plot_confusion_matrix, save_per_class_report,
)


# ============================================================
# Experiment definitions
# ============================================================
EXPERIMENTS = [
    ("E1", ["skeleton"],            "Skeleton only"),
    ("E2", ["imu"],                 "IMU only"),
    ("E3", ["rgb"],                 "RGB only"),
    ("E4", ["skeleton", "imu"],     "SK + IMU"),
    ("E5", ["skeleton", "rgb"],     "SK + RGB"),
    ("E6", ["imu", "rgb"],          "IMU + RGB"),
    ("E7", ["skeleton", "imu", "rgb"], "SK + IMU + RGB"),
]


# ============================================================
# Metrics
# ============================================================
def compute_all_metrics(labels, preds, num_classes):
    class_ids = list(range(num_classes))
    return {
        "accuracy":     float(accuracy_score(labels, preds)),
        "f1_macro":     float(f1_score(labels, preds, average="macro",
                                       labels=class_ids, zero_division=0)),
        "f1_weighted":  float(f1_score(labels, preds, average="weighted",
                                       labels=class_ids, zero_division=0)),
    }


# ============================================================
# Run one experiment (method + subset)
# ============================================================
def run_experiment(probs_all: dict, labels: np.ndarray, modalities: list,
                   num_classes: int, weight_step: float, log_fn):
    """
    Trả về list các dòng kết quả:
      {"method": ..., "weights": ..., "acc": ..., "f1_macro": ..., "f1_weighted": ...}
    """
    # Filter modalities có trong probs_all
    mods = [m for m in modalities if m in probs_all]
    if not mods:
        return None

    sub_probs = {m: probs_all[m] for m in mods}
    results = []

    # ── Baseline: nếu chỉ 1 modality, dùng chính nó ──
    if len(mods) == 1:
        probs = sub_probs[mods[0]]
        preds = probs.argmax(-1)
        m = compute_all_metrics(labels, preds, num_classes)
        results.append({
            "method": "single",
            "weights": {mods[0]: 1.0},
            "accuracy": m["accuracy"],
            "f1_macro": m["f1_macro"],
            "f1_weighted": m["f1_weighted"],
        })
        return results, probs, preds

    # ── Hard majority ──
    probs_hard, preds_hard = fuse_probs(sub_probs, method="hard")
    m = compute_all_metrics(labels, preds_hard, num_classes)
    results.append({
        "method": "hard",
        "weights": {m_: 1.0/len(mods) for m_ in mods},
        "accuracy": m["accuracy"],
        "f1_macro": m["f1_macro"],
        "f1_weighted": m["f1_weighted"],
    })

    # ── Average ──
    probs_avg, preds_avg = fuse_probs(sub_probs, method="average")
    m = compute_all_metrics(labels, preds_avg, num_classes)
    results.append({
        "method": "average",
        "weights": {m_: 1.0/len(mods) for m_ in mods},
        "accuracy": m["accuracy"],
        "f1_macro": m["f1_macro"],
        "f1_weighted": m["f1_weighted"],
    })

    # ── Weighted (grid search) ──
    gs = grid_search_weights(sub_probs, labels, step=weight_step,
                              metric="f1_macro", log_fn=log_fn)
    best_w = gs["weights"]
    probs_w, preds_w = fuse_probs(sub_probs, method="weighted", weights=best_w)
    m = compute_all_metrics(labels, preds_w, num_classes)
    results.append({
        "method": "weighted",
        "weights": best_w,
        "accuracy": m["accuracy"],
        "f1_macro": m["f1_macro"],
        "f1_weighted": m["f1_weighted"],
    })

    # Chọn best theo f1_macro
    best = max(results, key=lambda r: r["f1_macro"])
    if best["method"] == "hard":
        return results, probs_hard, preds_hard
    elif best["method"] == "average":
        return results, probs_avg, preds_avg
    else:
        return results, probs_w, preds_w


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fusion/voting.yaml")
    ap.add_argument("--manifest-dir", default="data/processed/MMAct/manifests")
    ap.add_argument("--classes-cfg", default="configs/classes_mmact.yaml")
    ap.add_argument("--runs-root", default="outputs/runs")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--skip-rgb", action="store_true",
                    help="Bỏ qua RGB (ví dụ chỉ có quick-test)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    weight_step = cfg["fusion"].get("weight_step", 0.1)
    min_samples = cfg["fusion"].get("min_val_samples", 500)

    # ── Run logger ──
    run = RunLogger("fusion", output_root=cfg["output"]["root"],
                    config=cfg, tag=args.tag)
    run.info("=" * 70)
    run.info(f"  FUSION BENCHMARK — {run.run_name}")
    run.info("=" * 70)
    run.info(f"  Runs root:     {args.runs_root}")
    run.info(f"  Weight step:   {weight_step}")
    run.info(f"  Min samples:   {min_samples}")
    run.info(f"  Run dir:       {run.path}")
    run.info("")

    # ── Load & align predictions ──
    run.info("─" * 70)
    run.info("  LOADING PREDICTIONS")
    run.info("─" * 70)

    manifest_csv = Path(args.manifest_dir) / "val_cross_view.csv"
    if not manifest_csv.exists():
        run.info(f"  ✗ Không tìm thấy manifest: {manifest_csv}")
        return

    candidates = cfg["fusion"]["candidates"]
    if args.skip_rgb:
        candidates = [c for c in candidates if c != "rgb"]

    try:
        probs_all, labels, clip_ids, info = align_all(
            runs_root=Path(args.runs_root),
            manifest_csv=manifest_csv,
            modalities=candidates,
            log_fn=run.info,
        )
    except Exception as e:
        run.info(f"  ✗ Lỗi khi align: {e}")
        return

    # Filter theo min_samples
    kept_mods = []
    for m in info["modalities"]:
        n = info["n_per_modality"][m]
        # n_per_modality = số clip có trong run — không phải intersection
        # Dùng intersection đã align ở trên
        if n < min_samples:
            run.info(f"  [filtered] {m}: chỉ {n} clips (< {min_samples})")
            continue
        kept_mods.append(m)

    if not kept_mods:
        run.info("  ✗ Không có modality nào pass min_samples filter.")
        return

    # Chỉ giữ các modality qua filter
    probs_all = {m: probs_all[m] for m in kept_mods}

    num_classes = next(iter(probs_all.values())).shape[1]
    class_names = load_class_names(args.classes_cfg)

    run.info("")
    run.info(f"  Modalities dùng:  {kept_mods}")
    run.info(f"  Num samples:      {len(labels)}")
    run.info(f"  Num classes:      {num_classes}")

    # ── Run experiments ──
    run.info("")
    run.info("─" * 70)
    run.info("  EXPERIMENTS")
    run.info("─" * 70)
    run.info("")

    all_results = []   # flat rows for CSV

    for exp_id, mods_req, exp_name in EXPERIMENTS:
        # Chỉ chạy nếu tất cả mods required đều available
        if not all(m in probs_all for m in mods_req):
            missing = [m for m in mods_req if m not in probs_all]
            run.info(f"[{exp_id}] {exp_name} — SKIP (thiếu {missing})")
            continue

        run.info(f"[{exp_id}] {exp_name}  ({mods_req})")
        out = run_experiment(probs_all, labels, mods_req, num_classes,
                             weight_step, run.info)
        if out is None:
            run.info("  Không chạy được.")
            continue
        results, probs_best, preds_best = out

        # In kết quả từng method
        for r in results:
            run.info(f"  · {r['method']:<10} "
                     f"acc={r['accuracy']:.4f}  "
                     f"f1_macro={r['f1_macro']:.4f}  "
                     f"f1_weighted={r['f1_weighted']:.4f}")

        # Xác định best method
        best_r = max(results, key=lambda r: r["f1_macro"])
        run.info(f"  → BEST: {best_r['method']}  "
                 f"f1_macro={best_r['f1_macro']:.4f}")

        # Lưu vào results CSV
        for r in results:
            all_results.append({
                "exp_id":      exp_id,
                "exp_name":    exp_name,
                "modalities":  "+".join(mods_req),
                "method":      r["method"],
                "weights":     json.dumps({k: round(v, 3) for k, v in r["weights"].items()}),
                "accuracy":    r["accuracy"],
                "f1_macro":    r["f1_macro"],
                "f1_weighted": r["f1_weighted"],
            })

        # Lưu artifacts cho best của experiment này
        exp_dir = run.path / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels, preds_best, labels=list(range(num_classes)))
        plot_confusion_matrix(cm, class_names,
                              exp_dir / "confusion_matrix.png", normalize=False)
        plot_confusion_matrix(cm, class_names,
                              exp_dir / "confusion_matrix_normalized.png",
                              normalize=True)

        # Per class
        save_per_class_report(labels, preds_best, class_names, exp_dir)

        # Save preds
        np.save(exp_dir / "val_probs.npy", probs_best.astype(np.float32))
        np.save(exp_dir / "val_preds.npy", preds_best.astype(np.int64))

        run.info(f"  Saved artifacts → {exp_dir.relative_to(run.path)}/")
        run.info("")

    # ── Save results CSV ──
    import csv
    if all_results:
        csv_path = run.path / "results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            w.writeheader()
            w.writerows(all_results)
        run.info(f"✓ Saved results.csv → {csv_path.relative_to(run.path)}")

    # ── Summary table ──
    run.info("")
    run.info("=" * 70)
    run.info("  SUMMARY")
    run.info("=" * 70)
    run.info("")
    run.info(f"  {'Exp':<5} {'Method':<10} {'F1_macro':>10} {'Acc':>8}  Weights")
    run.info("  " + "─" * 66)

    for r in all_results:
        run.info(f"  {r['exp_id']:<5} {r['method']:<10} "
                 f"{r['f1_macro']:>10.4f} {r['accuracy']:>8.4f}  "
                 f"{r['weights']}")

    # ── Bar chart comparison ──
    if all_results:
        plot_comparison(all_results, run.path / "comparison.png")
        run.info("")
        run.info(f"✓ Saved comparison.png → {(run.path / 'comparison.png').relative_to(run.path)}")

    # ── Best overall ──
    best_overall = max(all_results, key=lambda r: r["f1_macro"])
    run.info("")
    run.info("=" * 70)
    run.info("  BEST OVERALL")
    run.info("=" * 70)
    run.info(f"  Exp:      {best_overall['exp_id']} — {best_overall['exp_name']}")
    run.info(f"  Method:   {best_overall['method']}")
    run.info(f"  Weights:  {best_overall['weights']}")
    run.info(f"  F1_macro: {best_overall['f1_macro']:.4f}")
    run.info(f"  Accuracy: {best_overall['accuracy']:.4f}")
    run.info("")
    run.info(f"  Run dir: {run.path}")


# ============================================================
# Comparison plot
# ============================================================
def plot_comparison(results: list, out_path: Path):
    """Bar chart: mỗi experiment, best method's F1."""
    # Group by exp_id → best method
    by_exp = {}
    for r in results:
        eid = r["exp_id"]
        if eid not in by_exp or r["f1_macro"] > by_exp[eid]["f1_macro"]:
            by_exp[eid] = r

    exps = sorted(by_exp.keys(), key=lambda x: int(x[1:]))
    names = [f"{e}\n{by_exp[e]['exp_name']}" for e in exps]
    f1s = [by_exp[e]["f1_macro"] for e in exps]
    accs = [by_exp[e]["accuracy"] for e in exps]

    x = np.arange(len(exps))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, f1s, width, label="F1 macro", color="#4C72B0")
    bars2 = ax.bar(x + width/2, accs, width, label="Accuracy", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Fusion Benchmark — Best method per experiment", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # Annotate
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.3f}", ha="center", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
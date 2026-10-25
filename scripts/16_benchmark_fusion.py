"""
Benchmark voting and learned attention fusion — 7 tổ hợp.

Tổ hợp:
  E1: skeleton
  E2: imu
  E3: rgb
  E4: skeleton + imu
  E5: skeleton + rgb
  E6: imu + rgb
  E7: skeleton + imu + rgb

Voting: hard / average / weighted (grid search on the fit split only).
Attention: self / cross / adaptive / gate_cross.  Attention is fitted on
aligned fit predictions and evaluated on the untouched evaluation split.

Usage:
    python scripts/16_benchmark_fusion.py
    python scripts/16_benchmark_fusion.py --fusion-mode attention
    python scripts/16_benchmark_fusion.py --architecture imu=tcn \
        --architecture rgb=mc3_18
    python scripts/16_benchmark_fusion.py --manifest-dir data/processed/MMAct/manifests
    python scripts/16_benchmark_fusion.py --skip-rgb
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
from src.fusion.align import align_all, align_selected_runs, get_run_architecture
from src.fusion.attention import ATTENTION_METHODS, fit_attention
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


def parse_architecture_overrides(values: list[str]) -> dict:
    """Parse repeated CLI values such as ``skeleton=dual,ctrgcn_head``."""
    overrides = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --architecture '{value}'; use modality=arch[,arch]")
        modality, raw_architectures = value.split("=", 1)
        modality = modality.strip().lower()
        architectures = [item.strip() for item in raw_architectures.split(",")
                         if item.strip()]
        if modality not in {"skeleton", "imu", "rgb"} or not architectures:
            raise ValueError(
                f"Invalid --architecture '{value}'; modality must be "
                "skeleton, imu or rgb")
        overrides[modality] = architectures
    return overrides


def parse_run_dir_overrides(values: list[str]) -> dict:
    """Parse repeated ``--run-dir modality=path`` overrides."""
    overrides = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --run-dir '{value}'; use modality=path"
            )
        modality, path = value.split("=", 1)
        modality = modality.strip().lower()
        path = path.strip()
        if modality not in {"skeleton", "imu", "rgb"} or not path:
            raise ValueError(
                f"Invalid --run-dir '{value}'; modality must be skeleton, "
                "imu or rgb"
            )
        overrides[modality] = Path(path)
    return overrides


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
def run_voting_experiment(eval_probs_all: dict, eval_labels: np.ndarray,
                          fit_probs_all: dict | None, fit_labels: np.ndarray | None,
                          modalities: list, num_classes: int,
                          weight_step: float, log_fn):
    """
    Trả về list các dòng kết quả:
      {"method": ..., "weights": ..., "acc": ..., "f1_macro": ..., "f1_weighted": ...}
    """
    # Filter modalities có trong evaluation predictions.
    mods = [m for m in modalities if m in eval_probs_all]
    if not mods:
        return None

    eval_sub_probs = {m: eval_probs_all[m] for m in mods}
    results = []
    outputs = {}

    # ── Baseline: nếu chỉ 1 modality, dùng chính nó ──
    if len(mods) == 1:
        probs = eval_sub_probs[mods[0]]
        preds = probs.argmax(-1)
        m = compute_all_metrics(eval_labels, preds, num_classes)
        results.append({
            "method": "single",
            "weights": {mods[0]: 1.0},
            "accuracy": m["accuracy"],
            "f1_macro": m["f1_macro"],
            "f1_weighted": m["f1_weighted"],
        })
        outputs["single"] = (probs, preds)
        return results, outputs

    # ── Hard majority ──
    probs_hard, preds_hard = fuse_probs(eval_sub_probs, method="hard")
    m = compute_all_metrics(eval_labels, preds_hard, num_classes)
    results.append({
        "method": "hard",
        "weights": {m_: 1.0/len(mods) for m_ in mods},
        "accuracy": m["accuracy"],
        "f1_macro": m["f1_macro"],
        "f1_weighted": m["f1_weighted"],
    })
    outputs["hard"] = (probs_hard, preds_hard)

    # ── Average ──
    probs_avg, preds_avg = fuse_probs(eval_sub_probs, method="average")
    m = compute_all_metrics(eval_labels, preds_avg, num_classes)
    results.append({
        "method": "average",
        "weights": {m_: 1.0/len(mods) for m_ in mods},
        "accuracy": m["accuracy"],
        "f1_macro": m["f1_macro"],
        "f1_weighted": m["f1_weighted"],
    })
    outputs["average"] = (probs_avg, preds_avg)

    # ── Weighted: fit weights on the fit split, then freeze them ──
    if (fit_probs_all is not None and fit_labels is not None
            and all(m in fit_probs_all for m in mods)):
        fit_sub_probs = {m: fit_probs_all[m] for m in mods}
        gs = grid_search_weights(fit_sub_probs, fit_labels, step=weight_step,
                                 metric="f1_macro", log_fn=log_fn)
        best_w = gs["weights"]
        probs_w, preds_w = fuse_probs(eval_sub_probs, method="weighted",
                                      weights=best_w)
        m = compute_all_metrics(eval_labels, preds_w, num_classes)
        results.append({
            "method": "weighted",
            "weights": best_w,
            "accuracy": m["accuracy"],
            "f1_macro": m["f1_macro"],
            "f1_weighted": m["f1_weighted"],
            "fit_f1_macro": gs["score"],
        })
        outputs["weighted"] = (probs_w, preds_w)
    else:
        log_fn("  weighted SKIP — thiếu fit predictions cho đầy đủ modalities")

    # Không chọn best bằng nhãn evaluation. Mỗi method được lưu riêng để
    # việc chọn cuối cùng có thể thực hiện trên meta_val/test độc lập.
    return results, outputs


def run_attention_experiment(train_probs_all: dict, train_labels: np.ndarray,
                             val_probs_all: dict, val_labels: np.ndarray,
                             modalities: list, num_classes: int,
                             log_fn, epochs: int = 60, batch_size: int = 512,
                             hidden_dim: int = 64, num_heads: int = 4,
                             dropout: float = 0.1, lr: float = 2e-3,
                             weight_decay: float = 1e-4,
                             patience: int = 10, seed: int = 42,
                             device: str = "cpu"):
    """Fit all requested attention variants for one modality subset."""
    mods = [m for m in modalities
            if m in train_probs_all and m in val_probs_all]
    if not mods:
        return None
    if len(mods) == 1:
        probs = val_probs_all[mods[0]]
        preds = probs.argmax(-1)
        m = compute_all_metrics(val_labels, preds, num_classes)
        return [{
            "method": "single",
            "weights": {mods[0]: 1.0},
            "accuracy": m["accuracy"],
            "f1_macro": m["f1_macro"],
            "f1_weighted": m["f1_weighted"],
        }], {"single": (probs, preds)}

    train_x = np.stack([train_probs_all[m] for m in mods], axis=1)
    val_x = np.stack([val_probs_all[m] for m in mods], axis=1)
    results, outputs = [], {}
    for method in ATTENTION_METHODS:
        log_fn(f"  Fitting attention={method} on {len(train_x)} train clips...")
        fitted = fit_attention(
            train_x, train_labels, val_x, method=method,
            hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout,
            epochs=epochs, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay,
            patience=patience, seed=seed, device=device)
        metrics = compute_all_metrics(val_labels, fitted["preds"], num_classes)
        results.append({
            "method": method,
            "weights": {"attention": method},
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "fit_epoch": fitted["best_epoch"],
            "fit_f1_macro": fitted["stop_f1_macro"],
        })
        outputs[method] = (fitted["probs"], fitted["preds"], fitted)
    return results, outputs


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fusion/voting.yaml")
    ap.add_argument("--manifest-dir", default=None,
                    help="Override paths.manifest_dir from the fusion config")
    ap.add_argument("--classes-cfg", default="configs/classes_mmact.yaml")
    ap.add_argument("--runs-root", default="outputs/runs")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--fusion-mode", choices=["voting", "attention", "both"],
                    default="both",
                    help="Benchmark voting, learned attention, or both")
    ap.add_argument("--attention-device", default="cpu",
                    help="Device for the small attention meta-fuser")
    ap.add_argument("--attention-epochs", type=int, default=None)
    ap.add_argument("--attention-batch-size", type=int, default=None)
    ap.add_argument("--architecture", action="append", default=[],
                    metavar="MOD=ARCH[,ARCH]",
                    help="Restrict selected runs, repeat per modality; "
                         "e.g. --architecture imu=tcn")
    ap.add_argument("--run-dir", action="append", default=[],
                    metavar="MOD=PATH",
                    help="Use an explicit selected run directory for a modality")
    ap.add_argument("--skip-rgb", action="store_true",
                    help="Bỏ qua RGB (ví dụ chỉ có quick-test)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    args.manifest_dir = (
        args.manifest_dir
        or (cfg.get("paths", {}) or {}).get("manifest_dir")
        or "data/processed/MMAct/manifests"
    )

    try:
        architecture_filters = dict(
            cfg["fusion"].get(
                "architectures",
                cfg["fusion"].get("best_architectures", {}),
            ) or {}
        )
        architecture_filters.update(
            parse_architecture_overrides(args.architecture))
        run_dir_overrides = parse_run_dir_overrides(args.run_dir)
    except (KeyError, TypeError, ValueError) as error:
        ap.error(str(error))

    evaluation_cfg = cfg.get("evaluation", {}) or {}
    fit_split = str(evaluation_cfg.get("fit_split", "train_cross_view"))
    eval_split = str(evaluation_cfg.get("eval_split", "val_cross_view"))
    fit_source = str(evaluation_cfg.get("fit_source", "train"))
    if fit_source not in {"train", "oof"}:
        ap.error("evaluation.fit_source must be 'train' or 'oof'")

    weight_step = cfg["fusion"].get("weight_step", 0.1)
    min_samples = cfg["fusion"].get("min_val_samples", 500)
    attention_cfg = cfg.get("attention", {})
    attention_epochs = args.attention_epochs or attention_cfg.get("epochs", 60)
    attention_batch_size = (args.attention_batch_size
                             or attention_cfg.get("batch_size", 512))
    attention_hidden_dim = attention_cfg.get("hidden_dim", 64)
    attention_num_heads = attention_cfg.get("num_heads", 4)
    attention_dropout = attention_cfg.get("dropout", 0.1)
    attention_lr = attention_cfg.get("lr", 2e-3)
    attention_weight_decay = attention_cfg.get("weight_decay", 1e-4)
    attention_patience = attention_cfg.get("patience", 10)
    attention_seed = attention_cfg.get("seed", 42)

    # ── Run logger ──
    run = RunLogger("fusion", output_root=cfg["output"]["root"],
                    config=cfg, tag=args.tag)
    run.info("=" * 70)
    run.info(f"  FUSION BENCHMARK — {run.run_name}")
    run.info("=" * 70)
    run.info(f"  Runs root:     {args.runs_root}")
    run.info(f"  Weight step:   {weight_step}")
    run.info(f"  Min samples:   {min_samples}")
    run.info(f"  Fusion mode:   {args.fusion_mode}")
    run.info(f"  Fit split:     {fit_split} ({fit_source})")
    run.info(f"  Eval split:    {eval_split}")
    run.info(f"  Architecture filters: "
             f"{architecture_filters or 'all available (best F1 selected)'}")
    run.info(f"  Run dir:       {run.path}")
    run.info("")

    # ── Load & align predictions ──
    run.info("─" * 70)
    run.info("  LOADING PREDICTIONS")
    run.info("─" * 70)

    manifest_csv = Path(args.manifest_dir) / f"{eval_split}.csv"
    if not manifest_csv.exists():
        run.info(f"  ✗ Không tìm thấy manifest: {manifest_csv}")
        return

    candidates = cfg["fusion"]["candidates"]
    if args.skip_rgb:
        candidates = [c for c in candidates if c != "rgb"]

    try:
        if run_dir_overrides:
            probs_all, labels, clip_ids, info = align_selected_runs(
                run_dir_overrides,
                manifest_csv,
                modalities=candidates,
                log_fn=run.info,
            )
            info["run_dirs"] = {
                modality: str(path)
                for modality, path in run_dir_overrides.items()
                if modality in info.get("modalities", [])
            }
            info["architectures"] = {
                modality: get_run_architecture(path)
                for modality, path in run_dir_overrides.items()
                if modality in info.get("modalities", [])
            }
        else:
            probs_all, labels, clip_ids, info = align_all(
                runs_root=Path(args.runs_root),
                manifest_csv=manifest_csv,
                modalities=candidates,
                log_fn=run.info,
                architectures=architecture_filters,
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
    run.info(f"  Architectures dùng: "
             f"{ {m: info.get('architectures', {}).get(m) for m in kept_mods} }")
    run.info(f"  Num samples:      {len(labels)}")
    run.info(f"  Num classes:      {num_classes}")

    # Fit all learned fusion methods on the same aligned fit split.  Reuse the
    # exact run directories selected for evaluation so train/eval never mix
    # checkpoints. Weighted voting and attention both consume this split.
    fit_probs_all = {}
    fit_labels = None
    fit_info = {"n_common": 0, "modalities": []}
    fit_manifest = Path(args.manifest_dir) / f"{fit_split}.csv"
    if fit_manifest.exists():
        try:
            fit_probs_all, fit_labels, _, fit_info = align_selected_runs(
                {m: info["run_dirs"][m] for m in kept_mods
                 if m in info.get("run_dirs", {})},
                fit_manifest, modalities=kept_mods, log_fn=run.info,
                prediction_prefix=("oof_train" if fit_source == "oof"
                                    else None))
            run.info(f"  [fit] intersection: {fit_info['n_common']} clips; "
                     f"modalities: {fit_info['modalities']}")
            if fit_source == "train":
                run.info("  [fit] WARNING: train predictions are in-sample; "
                         "OOF is required for an unbiased final report")
        except Exception as error:
            run.info(f"  [fit] skip: cannot align fit predictions: {error}")
            fit_probs_all = {}
            fit_labels = None
    else:
        run.info(f"  [fit] skip: missing {fit_manifest}")

    if fit_source == "oof":
        run.info("  [fit] OOF mode: using oof_train_probs/labels/clip_ids.npy")

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
        exp_dir = run.path / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        def record_results(results, fusion_mode):
            for r in results:
                run.info(f"  · {fusion_mode}/{r['method']:<10} "
                         f"acc={r['accuracy']:.4f}  "
                         f"f1_macro={r['f1_macro']:.4f}  "
                         f"f1_weighted={r['f1_weighted']:.4f}")
                all_results.append({
                    "exp_id":       exp_id,
                    "exp_name":     exp_name,
                    "modalities":   "+".join(mods_req),
                    "architectures": json.dumps({
                        m: info.get("architectures", {}).get(m)
                        for m in mods_req
                    }),
                    "fusion_mode":  fusion_mode,
                    "method":       r["method"],
                    "weights":      json.dumps(r["weights"]),
                    "accuracy":     r["accuracy"],
                    "f1_macro":     r["f1_macro"],
                    "f1_weighted":  r["f1_weighted"],
                    "fit_epoch":    r.get("fit_epoch"),
                    "fit_f1_macro": r.get("fit_f1_macro"),
                    "fit_split":    fit_split,
                    "fit_source":   fit_source,
                    "eval_split":   eval_split,
                    "n_fit_clips":  fit_info.get("n_common", 0),
                    "n_eval_clips": len(labels),
                })
            run.info(f"  → Recorded {fusion_mode}: {len(results)} methods "
                     "(không chọn best trên eval split)")

        def save_artifacts(artifact_dir, probs, preds):
            artifact_dir.mkdir(parents=True, exist_ok=True)
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
            plot_confusion_matrix(cm, class_names,
                                  artifact_dir / "confusion_matrix.png",
                                  normalize=False)
            plot_confusion_matrix(cm, class_names,
                                  artifact_dir / "confusion_matrix_normalized.png",
                                  normalize=True)
            save_per_class_report(labels, preds, class_names, artifact_dir)
            np.save(artifact_dir / "eval_probs.npy", probs.astype(np.float32))
            np.save(artifact_dir / "eval_preds.npy", preds.astype(np.int64))

        if args.fusion_mode in {"voting", "both"}:
            out = run_voting_experiment(
                probs_all, labels, fit_probs_all, fit_labels, mods_req,
                num_classes, weight_step, run.info)
            if out is not None:
                results, outputs = out
                record_results(results, "voting")
                for method, output in outputs.items():
                    save_artifacts(exp_dir / "voting" / method,
                                   output[0], output[1])
            else:
                run.info("  Voting không chạy được.")

        if args.fusion_mode in {"attention", "both"}:
            missing_train = [m for m in mods_req
                             if m not in fit_probs_all]
            if missing_train:
                run.info("  Attention SKIP — thiếu fit predictions cho "
                         f"{missing_train}; cần chạy lại evaluation với "
                         "train_clip_ids.npy hoặc OOF artifacts.")
            else:
                out = run_attention_experiment(
                    fit_probs_all, fit_labels,
                    probs_all, labels, mods_req, num_classes, run.info,
                    epochs=attention_epochs, batch_size=attention_batch_size,
                    hidden_dim=attention_hidden_dim,
                    num_heads=attention_num_heads,
                    dropout=attention_dropout, lr=attention_lr,
                    weight_decay=attention_weight_decay,
                    patience=attention_patience, seed=attention_seed,
                    device=args.attention_device)
                if out is not None:
                    results, outputs = out
                    record_results(results, "attention")
                    for method, output in outputs.items():
                        save_artifacts(exp_dir / "attention" / method,
                                       output[0], output[1])
                else:
                    run.info("  Attention không chạy được.")
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

        protocol = {
            "fit_split": fit_split,
            "fit_source": fit_source,
            "eval_split": eval_split,
            "fit_intersection_clips": int(fit_info.get("n_common", 0)),
            "eval_intersection_clips": int(len(labels)),
            "modalities": kept_mods,
            "architectures": {
                m: info.get("architectures", {}).get(m) for m in kept_mods
            },
            "run_dirs": {
                m: str(info.get("run_dirs", {}).get(m)) for m in kept_mods
            },
            "selection_policy": "all methods reported; no best selected on eval",
            "weighted_voting_policy": "weights fitted on fit split and frozen",
        }
        (run.path / "protocol.json").write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
        np.save(run.path / "eval_clip_ids.npy",
                np.asarray(clip_ids, dtype=str))
        run.info("✓ Saved protocol.json and eval_clip_ids.npy")

    # ── Summary table ──
    run.info("")
    run.info("=" * 70)
    run.info("  SUMMARY")
    run.info("=" * 70)
    run.info("")
    run.info(f"  {'Exp':<5} {'Method':<10} {'F1_macro':>10} {'Acc':>8}  Weights")
    run.info("  " + "─" * 66)

    for r in all_results:
        run.info(f"  {r['exp_id']:<5} {r['fusion_mode'] + '/' + r['method']:<19} "
                 f"{r['f1_macro']:>10.4f} {r['accuracy']:>8.4f}  "
                 f"{r['weights']}")

    # ── Bar chart comparison ──
    if all_results:
        plot_comparison(all_results, run.path / "comparison.png")
        run.info("")
        run.info(f"✓ Saved comparison.png → {(run.path / 'comparison.png').relative_to(run.path)}")

    if not all_results:
        run.info("  Không có kết quả để tổng hợp.")
        return
    run.info("")
    run.info("  Không chọn BEST theo eval split để tránh leakage. ")
    run.info("  Hãy chọn method trên meta_val hoặc báo cáo từng method.")
    run.info(f"  Run dir: {run.path}")


# ============================================================
# Comparison plot
# ============================================================
def plot_comparison(results: list, out_path: Path):
    """Bar chart with every method; never selects a best method by eval score."""
    rows = sorted(results, key=lambda r: (r["exp_id"], r["fusion_mode"], r["method"]))
    names = [f"{r['exp_id']}\n{r['fusion_mode']}:{r['method']}" for r in rows]
    f1s = [r["f1_macro"] for r in rows]
    accs = [r["accuracy"] for r in rows]

    x = np.arange(len(rows))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(14, len(rows) * 0.55), 7))
    bars1 = ax.bar(x - width/2, f1s, width, label="F1 macro", color="#4C72B0")
    bars2 = ax.bar(x + width/2, accs, width, label="Accuracy", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Fusion Benchmark — all methods on the same eval split", fontsize=13)
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

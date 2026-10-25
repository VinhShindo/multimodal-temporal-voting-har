"""Audit leakage and export comparable modality-only model benchmarks.

Examples:
    python scripts/13_benchmark_models.py --modality imu
    python scripts/13_benchmark_models.py --modality skeleton
    python scripts/13_benchmark_models.py --modality rgb
"""
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml

from src.models.architecture_registry import architecture_metadata


MODALITY_DEFAULT_SPLIT = {
    "imu": "cross_view",
    "skeleton": "cross_view",
    "rgb": "cross_view",
}
DEFAULT_ARCHITECTURES = {
        "imu": ["cnn_bilstm", "cnn_bilstm_cbam", "tcn", "transformer", "xtinyhar",
            "inceptiontime", "fckd"],
    "skeleton": ["stgcn", "ms_g3d", "ctrgcn", "sta_gcn", "ctrgcn_head", "dual"],
        "rgb": ["r2plus1d_18", "r3d_18", "mc3_18", "s3d", "mvit_v2_s",
            "videomae", "videomae2"],
}
# Kept callable for a separately reported ablation, but excluded from the
# default architecture benchmark because its training objective is different.
OPTIONAL_ARCHITECTURES = {
    "skeleton": {"barlow_twins"},
    "imu": set(),
    "rgb": set(),
}
RESULT_COLUMNS = [
    "run", "result_type", "architecture", "stream", "feature_set", "split", "seed",
    "accuracy", "f1_macro", "f1_weighted", "num_samples", "parameters_m",
    "best_epoch", "train_seconds", "run_dir",
    "architecture_status", "architecture_note", "paper_reference",
]


def audit_split(manifest_dir: Path, modality: str, split: str,
                allow_camera_overlap: bool = False) -> dict:
    train_manifest = manifest_dir / f"train_{split}.csv"
    val_manifest = manifest_dir / f"val_{split}.csv"
    if not train_manifest.exists() or not val_manifest.exists():
        raise FileNotFoundError(f"Missing manifests for {modality}/{split}")

    train_df = pd.read_csv(train_manifest)
    val_df = pd.read_csv(val_manifest)
    path_column = f"{modality}_path"
    for frame in (train_df, val_df):
        if path_column not in frame:
            raise ValueError(f"Manifest missing {path_column}")
        frame.drop(frame[frame[path_column].isna() | frame[path_column].eq("")].index,
                   inplace=True)

    train_ids = set(train_df["clip_id"])
    val_ids = set(val_df["clip_id"])
    clip_overlap = train_ids & val_ids
    if clip_overlap:
        raise ValueError(f"{modality}/{split}: {len(clip_overlap)} clip_id overlaps")

    report = {
        "modality": modality,
        "split": split,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "clip_id_overlap": 0,
    }
    if modality == "imu":
        event_columns = ["subject", "scene", "session", "action"]
        train_events = set(map(tuple, train_df[event_columns].drop_duplicates().to_numpy()))
        val_events = set(map(tuple, val_df[event_columns].drop_duplicates().to_numpy()))
        event_overlap = train_events & val_events
        report["train_unique_events"] = len(train_events)
        report["val_unique_events"] = len(val_events)
        report["physical_event_overlap"] = len(event_overlap)
        if event_overlap and split == "cross_view":
            report["training_event_purge_required"] = len(event_overlap)
            report["physical_event_overlap_after_purge"] = 0
            report["event_purge_strategy"] = "exclude_validation_events_from_training"
        elif event_overlap:
            raise ValueError(
                f"IMU {split} leaks {len(event_overlap)} physical events; "
                f"example={next(iter(event_overlap))}"
            )
    elif split == "cross_view":
        train_cameras = sorted(train_df["cam"].dropna().unique().tolist())
        val_cameras = sorted(val_df["cam"].dropna().unique().tolist())
        camera_overlap = set(train_cameras) & set(val_cameras)
        report["train_cameras"] = train_cameras
        report["val_cameras"] = val_cameras
        report["camera_overlap"] = sorted(camera_overlap)
        if camera_overlap and not allow_camera_overlap:
            raise ValueError(f"{modality} cross_view camera overlap: {camera_overlap}")
        if camera_overlap and allow_camera_overlap:
            report["camera_overlap_policy"] = "allowed_for_group_holdout"
    return report


def discover_runs(run_root: Path, modality: str, split: str,
                  only_run_names: set[str] | None = None) -> tuple[list[dict], int]:
    results = []
    excluded_imu_without_split = 0
    for run_dir in run_root.iterdir():
        if not run_dir.is_dir():
            continue
        if only_run_names is not None and run_dir.name not in only_run_names:
            continue
        config_path = run_dir / "config.yaml"
        metadata_path = run_dir / "metadata.json"
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            config = (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                      if config_path.exists() else {})
            metadata = (json.loads(metadata_path.read_text(encoding="utf-8"))
                        if metadata_path.exists() else {})
            metric_data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, yaml.YAMLError):
            continue

        configured_split = config.get("split", metadata.get("split"))
        if modality == "imu" and not configured_split:
            excluded_imu_without_split += 1
            continue
        if configured_split is None:
            configured_split = MODALITY_DEFAULT_SPLIT[modality]
        if configured_split != split:
            continue

        val = metric_data.get("val", {})
        if not val or "accuracy" not in val or "f1_macro" not in val:
            continue
        history_path = run_dir / "history.csv"
        history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
        best_epoch = None
        train_seconds = None
        if not history.empty:
            if "val_f1" in history:
                best_epoch = int(history.loc[history["val_f1"].idxmax(), "epoch"])
            if "time_s" in history:
                train_seconds = round(float(history["time_s"].sum()), 2)

        log_path = run_dir / "log.txt"
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        params_match = re.search(r"params=([0-9.]+)M", log_text)
        model_config = config.get("model", {})
        is_ensemble = bool(metadata.get("ensemble"))
        provenance = architecture_metadata(
            modality,
            model_config.get("arch", model_config.get("name", "unknown")),
        )
        results.append({
            "run": run_dir.name,
            "result_type": "ensemble" if is_ensemble else "single",
            "architecture": (f"{metadata.get('architecture', 'unknown')}_ensemble"
                             if is_ensemble else model_config.get("arch", model_config.get("name", "unknown"))),
            "stream": model_config.get("stream"),
            "feature_set": config.get("feature_set", metadata.get("feature_set", "raw")),
            "split": configured_split,
            "seed": ("+".join(map(str, metadata.get("seeds", [])))
                     if is_ensemble else config.get("seed")),
            "accuracy": float(val["accuracy"]),
            "f1_macro": float(val["f1_macro"]),
            "f1_weighted": float(val.get("f1_weighted", float("nan"))),
            "num_samples": int(val.get("num_samples", 0)),
            "parameters_m": float(params_match.group(1)) if params_match else None,
            "best_epoch": best_epoch,
            "train_seconds": train_seconds,
            "run_dir": str(run_dir),
            **provenance,
        })
    return results, excluded_imu_without_split


def build_train_command(modality: str, arch: str, split: str, seed: int,
                        tag: str, epochs: int | None, manifest_dir: str,
                        num_workers: int | None = None,
                        batch_size: int | None = None,
                        lr: float | None = None) -> list[str]:
    script = {
        "imu": "scripts/12_train_imu.py",
        "skeleton": "scripts/10_train_skeleton.py",
        "rgb": "scripts/11_train_rgb.py",
    }[modality]
    command = [
        sys.executable, script,
        "--arch", arch,
        "--seed", str(seed),
        "--tag", tag,
        "--manifest-dir", manifest_dir,
    ]
    command.append("--quiet-progress")
    if modality == "imu":
        command.extend(["--split", split])
    elif split != "cross_view":
        raise ValueError(f"{modality} trainer currently supports cross_view only")
    if epochs is not None:
        command.extend(["--epochs", str(epochs)])
    if num_workers is not None:
        command.extend(["--num-workers", str(num_workers)])
    if batch_size is not None:
        command.extend(["--batch-size", str(batch_size)])
    if lr is not None:
        command.extend(["--lr", str(lr)])
    return command


def launch_architectures(modality: str, architectures: list[str], split: str,
                         seed: int, epochs: int | None, timestamp: str,
                         output_dir: Path,
                         manifest_dir: str = "data/processed/MMAct/manifests",
                         max_samples: int | None = None,
                         num_workers: int | None = None,
                         no_pretrained: bool = False,
                         batch_size: int | None = None,
                         lr: float | None = None) -> tuple[set[str], list[dict]]:
    run_root = Path("outputs/runs") / modality
    launched_run_names = set()
    failures = []
    if modality == "rgb":
        tag_prefix = f"bench_{timestamp}"
        log_path = output_dir / "rgb_benchmark.train.log"
        command = [
            sys.executable, "scripts/11_train_rgb.py",
            "--benchmark-architectures", *architectures,
            "--benchmark-tag-prefix", tag_prefix,
            "--seed", str(seed),
            "--manifest-dir", manifest_dir,
            "--quiet-progress",
        ]
        if epochs is not None:
            command.extend(["--epochs", str(epochs)])
        if max_samples is not None:
            command.extend(["--max-samples", str(max_samples)])
        if num_workers is not None:
            command.extend(["--num-workers", str(num_workers)])
        if no_pretrained:
            command.append("--no-pretrained")
        if batch_size is not None:
            command.extend(["--batch-size", str(batch_size)])
        if lr is not None:
            command.extend(["--lr", str(lr)])
        print(f"Training rgb architectures in one process; log={log_path}")
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.run(
                command, stdout=log_file, stderr=subprocess.STDOUT, check=False)
        for architecture in architectures:
            suffix = f"_{tag_prefix}_{architecture}"
            run_dirs = [
                path for path in run_root.iterdir()
                if path.is_dir() and path.name.endswith(suffix)
                and (path / "metrics.json").exists()
            ] if run_root.exists() else []
            if run_dirs:
                launched_run_names.update(path.name for path in run_dirs)
            else:
                failures.append({
                    "architecture": architecture,
                    "return_code": process.returncode,
                    "log": str(log_path),
                })
        return launched_run_names, failures

    for architecture in architectures:
        tag = f"bench_{timestamp}_{architecture}"
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / f"{architecture}.train.log"
        command = build_train_command(
            modality, architecture, split, seed, tag, epochs,
            manifest_dir,
            num_workers=num_workers, batch_size=batch_size, lr=lr)
        print(f"Training {modality}/{architecture}; log={log_path}")
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.run(
                command, stdout=log_file, stderr=subprocess.STDOUT, check=False)
        run_dirs = [
            path for path in run_root.iterdir()
            if path.is_dir() and path.name.endswith(f"_{tag}")
            and (path / "metrics.json").exists()
        ] if run_root.exists() else []
        if process.returncode != 0 or not run_dirs:
            failures.append({
                "architecture": architecture,
                "return_code": process.returncode,
                "log": str(log_path),
            })
            print(f"FAILED {modality}/{architecture}; inspect {log_path}")
            continue
        launched_run_names.update(path.name for path in run_dirs)
    return launched_run_names, failures


def save_benchmark_charts(results: list[dict], output_dir: Path) -> list[str]:
    if not results:
        return []

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    frame = pd.DataFrame(results).sort_values(
        ["f1_macro", "accuracy"], ascending=False).reset_index(drop=True)
    labels = frame["architecture"].astype(str).tolist()
    positions = np.arange(len(frame))
    metrics = [
        ("accuracy", "Accuracy"),
        ("f1_macro", "Macro-F1"),
        ("f1_weighted", "Weighted-F1"),
    ]
    height = 0.24
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(frame) * 0.65)))
    for index, (column, label) in enumerate(metrics):
        offset = (index - (len(metrics) - 1) / 2) * height
        ax.barh(positions + offset, frame[column], height=height, label=label)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Validation score")
    ax.set_title("Model validation comparison")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    metrics_path = output_dir / "metrics_comparison.png"
    fig.savefig(metrics_path, dpi=160)
    plt.close(fig)

    cost_column = next((
        column for column in ("train_seconds", "parameters_m")
        if column in frame and frame[column].notna().any()
    ), None)
    if cost_column is None:
        return [metrics_path.name]

    cost_frame = frame.dropna(subset=[cost_column]).copy()
    x_label = "Training time (minutes)" if cost_column == "train_seconds" else "Parameters (M)"
    x = cost_frame[cost_column].astype(float)
    if cost_column == "train_seconds":
        x = x / 60
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, cost_frame["f1_macro"], s=70, alpha=0.8)
    for x_value, (_, row) in zip(x, cost_frame.iterrows()):
        ax.annotate(str(row["architecture"]),
                    (x_value, row["f1_macro"]), xytext=(5, 5),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Validation Macro-F1")
    ax.set_title("Validation quality versus model cost")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    cost_path = output_dir / "performance_vs_cost.png"
    fig.savefig(cost_path, dpi=160)
    plt.close(fig)
    return [metrics_path.name, cost_path.name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", required=True,
                        choices=["imu", "skeleton", "rgb"])
    parser.add_argument("--split", choices=["cross_view", "cross_scene"], default=None)
    parser.add_argument("--mode", choices=["train", "reuse"], default="train",
                        help="train = train each architecture from scratch; reuse = collect existing runs")
    parser.add_argument("--architectures", nargs="+", default=None,
                        help="subset of architectures; defaults to all supported architectures for modality")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None,
                        help="override configured epochs for every architecture")
    parser.add_argument("--manifest-dir", default=None,
                        help="Override paths.manifest_dir from the modality data config")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="limit RGB train/validation samples for a smoke benchmark")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="override dataloader worker count")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="disable pretrained RGB weights (useful for smoke tests)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override training batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="override learning rate for every architecture")
    parser.add_argument("--min-samples", type=int, default=None,
                        help="in reuse mode, ignore runs with fewer validation samples")
    parser.add_argument("--latest-per-architecture", action="store_true",
                        help="in reuse mode, keep only the newest matching run per architecture")
    parser.add_argument("--allow-camera-overlap", action="store_true",
                        help=("Allow camera overlap for the new group-disjoint "
                              "multimodal holdout; physical events remain disjoint."))
    parser.add_argument("--run-contains", default=None,
                        help="in reuse mode, include only runs whose name contains this string")
    parser.add_argument("--run-root", default="outputs/runs")
    parser.add_argument("--output-root", default="outputs/benchmark")
    args = parser.parse_args()

    split = args.split or MODALITY_DEFAULT_SPLIT[args.modality]
    if args.manifest_dir is None:
        data_cfg_by_modality = {
            "skeleton": "configs/data/skeleton_mmact.yaml",
            "imu": "configs/data/imu_mmact.yaml",
            "rgb": "configs/data/rgb_mmact.yaml",
        }
        with open(data_cfg_by_modality[args.modality], encoding="utf-8") as stream:
            data_cfg = yaml.safe_load(stream) or {}
        args.manifest_dir = str(
            (data_cfg.get("paths", {}) or {}).get(
                "manifest_dir", "data/processed/MMAct/manifests"
            )
        )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(args.output_root) / args.modality / f"{split}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    try:
        audit = audit_split(
            Path(args.manifest_dir), args.modality, split,
            allow_camera_overlap=args.allow_camera_overlap,
        )
    except (FileNotFoundError, ValueError) as error:
        audit = {
            "modality": args.modality,
            "split": split,
            "audit_status": "FAIL",
            "error": str(error),
        }
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(
            output_dir / "results.csv", index=False)
        (output_dir / "leakage_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        (output_dir / "README.md").write_text(
            f"# {args.modality.upper()} benchmark ({split})\n\n"
            f"- Leakage/data audit: FAIL\n- Error: {error}\n",
            encoding="utf-8")
        print(output_dir)
        raise SystemExit(str(error))

    audit["audit_status"] = "PASS"
    architectures = args.architectures or DEFAULT_ARCHITECTURES[args.modality]
    allowed_architectures = (set(DEFAULT_ARCHITECTURES[args.modality])
                             | OPTIONAL_ARCHITECTURES[args.modality])
    invalid_architectures = sorted(set(architectures) - allowed_architectures)
    if invalid_architectures:
        raise SystemExit(
            f"Unsupported {args.modality} architecture(s): {invalid_architectures}; "
            f"choose from {DEFAULT_ARCHITECTURES[args.modality]}"
        )

    selected_run_names = None
    failures = []
    effective_batch_size = args.batch_size
    if (args.modality == "rgb" and effective_batch_size is None
            and {"videomae", "videomae2"}.intersection(architectures)):
        effective_batch_size = 16
    if args.mode == "train":
        if args.modality in {"rgb", "skeleton"} and split != "cross_view":
            raise SystemExit(f"{args.modality} training currently requires --split cross_view")
        selected_run_names, failures = launch_architectures(
            args.modality, architectures, split, args.seed, args.epochs,
            timestamp, output_dir, args.manifest_dir, args.max_samples,
            args.num_workers, args.no_pretrained, effective_batch_size,
            args.lr)

    results, excluded_without_split = discover_runs(
        Path(args.run_root) / args.modality, args.modality, split,
        only_run_names=selected_run_names)
    if args.mode == "reuse":
        if args.run_contains:
            results = [row for row in results
                       if args.run_contains in row["run"]]
        if args.architectures:
            results = [row for row in results
                       if row["architecture"] in set(args.architectures)]
        if args.min_samples is not None:
            results = [row for row in results
                       if row["num_samples"] >= args.min_samples]
        if args.latest_per_architecture:
            latest = {}
            for row in results:
                current = latest.get(row["architecture"])
                if current is None or row["run"] > current["run"]:
                    latest[row["architecture"]] = row
            results = list(latest.values())
    results.sort(key=lambda item: (item["f1_macro"], item["accuracy"]), reverse=True)
    pd.DataFrame(results, columns=RESULT_COLUMNS).to_csv(
        output_dir / "results.csv", index=False)
    chart_names = save_benchmark_charts(results, output_dir)
    audit["excluded_imu_runs_without_explicit_split"] = excluded_without_split
    audit["benchmark_mode"] = args.mode
    if args.modality == "rgb" and args.mode == "train":
        audit["dataset_load_strategy"] = "single_process_preload_once"
        audit["max_samples_per_split"] = args.max_samples
        audit["pretrained_weights"] = not args.no_pretrained
        audit["batch_size"] = effective_batch_size
    audit["requested_architectures"] = architectures if args.mode == "train" else []
    audit["seed"] = args.seed if args.mode == "train" else None
    audit["failed_architectures"] = failures
    (output_dir / "leakage_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    lines = [f"# {args.modality.upper()} benchmark ({split})", ""]
    lines.extend([
        f"- Leakage audit: PASS; clip overlap = {audit['clip_id_overlap']}",
        f"- Train/validation rows: {audit['train_rows']} / {audit['val_rows']}",
        f"- Completed runs: {len(results)}",
        f"- Mode: {args.mode}",
    ])
    if args.modality == "rgb" and args.mode == "train":
        lines.append("- Dataset loading: preloaded once and reused across architectures")
        if args.max_samples is not None:
            lines.append(
                f"- Smoke subset: at most {args.max_samples} samples per split; "
                "do not treat these metrics as full benchmark results."
            )
    if results:
        best = results[0]
        lines.extend([
            f"- Best run: {best['run']} ({best['architecture']})",
            f"- Best validation accuracy / macro-F1: {best['accuracy']:.4f} / {best['f1_macro']:.4f}",
        ])
    else:
        lines.append("- No completed runs matched this split.")
        if excluded_without_split:
            lines.append(
                f"- Excluded {excluded_without_split} legacy IMU run(s) without explicit split metadata; "
                "rerun training with --split cross_scene to include them."
            )
    for failure in failures:
        lines.append(
            f"- Training failed for {failure['architecture']} (exit={failure['return_code']}); "
            f"log: `{failure['log']}`"
        )
    lines.extend([
        "",
        "Full per-run results: `results.csv`.",
        "Split audit: `leakage_audit.json`.",
    ])
    if chart_names:
        lines.append("Charts: " + ", ".join(f"`{name}`" for name in chart_names) + ".")
    if args.modality == "imu":
        if split == "cross_view":
            lines.insert(
                4,
                f"- Cross-view IMU validation events purged from train: "
                f"{audit.get('training_event_purge_required', 0)} "
                f"(overlap after purge: {audit.get('physical_event_overlap_after_purge', 0)})"
            )
        else:
            lines.insert(4, f"- Unique physical IMU event overlap: {audit['physical_event_overlap']}")
        if excluded_without_split:
            lines.append(f"- Excluded legacy IMU runs without explicit split metadata: {excluded_without_split}")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_dir)
    if results:
        print(f"best={best['run']} arch={best['architecture']} "
              f"acc={best['accuracy']:.4f} macro_f1={best['f1_macro']:.4f}")
    else:
        print("No compatible completed runs; see README.md and results.csv.")


if __name__ == "__main__":
    main()

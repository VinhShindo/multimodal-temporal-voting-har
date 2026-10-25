"""Train K-fold models and generate leakage-safe OOF predictions.

OOF (out-of-fold) predictions are required when a learned fusion model is
fit on the training split.  A sample in ``oof_train_probs.npy`` is predicted
by a model whose training fold did not contain that sample.

The script uses the existing benchmark trainer for each fold and architecture
so the fold models follow the normal project training path.  It writes the
combined artifacts into the selected full-training run directories:

    oof_train_probs.npy
    oof_train_labels.npy
    oof_train_clip_ids.npy

The final test predictions must still come from the selected full-training
checkpoints; OOF models are only used to fit voting weights/attention.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


EVENT_COLUMNS = ["subject", "scene", "session", "action"]
MODALITIES = ("skeleton", "imu", "rgb")
DEFAULT_ARCHITECTURES = {
    "skeleton": "ctrgcn_head",
    "imu": "tcn",
    "rgb": "mc3_18",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 5-fold OOF train predictions for fusion"
    )
    parser.add_argument(
        "--manifest-dir",
        default="data/processed/MMAct/manifests/multimodal_cross_view_3way",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/oof/ctrgcn_head_tcn_mc3",
        help="Stores fold manifests, benchmark logs and OOF audit",
    )
    parser.add_argument("--skeleton-run-dir", required=True,
                        help="Selected full-train Skeleton run to receive OOF files")
    parser.add_argument("--imu-run-dir", required=True,
                        help="Selected full-train IMU run to receive OOF files")
    parser.add_argument("--rgb-run-dir", required=True,
                        help="Selected full-train RGB run to receive OOF files")
    parser.add_argument("--skeleton-arch", default=DEFAULT_ARCHITECTURES["skeleton"])
    parser.add_argument("--imu-arch", default=DEFAULT_ARCHITECTURES["imu"])
    parser.add_argument("--rgb-arch", default=DEFAULT_ARCHITECTURES["rgb"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--skeleton-batch-size", type=int, default=256)
    parser.add_argument("--imu-batch-size", type=int, default=256)
    parser.add_argument("--rgb-batch-size", type=int, default=4)
    parser.add_argument("--skeleton-epochs", type=int, default=None)
    parser.add_argument("--imu-epochs", type=int, default=None)
    parser.add_argument("--rgb-epochs", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Smoke test only; do not use for final OOF")
    return parser.parse_args()


def event_groups(frame: pd.DataFrame) -> pd.Series:
    missing = sorted(set(EVENT_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Manifest missing event columns: {missing}")
    return frame[EVENT_COLUMNS].astype(str).agg("|".join, axis=1)


def modality_frame(frame: pd.DataFrame, modality: str) -> pd.DataFrame:
    column = f"{modality}_path"
    keep = frame[column].notna() & frame[column].astype(str).str.strip().ne("")
    output = frame.loc[keep].copy().reset_index(drop=True)
    if modality == "imu":
        output = output.drop_duplicates(subset=EVENT_COLUMNS).reset_index(drop=True)
    return output


def manifest_clip_ids(path: Path, modality: str) -> np.ndarray:
    frame = pd.read_csv(path)
    return modality_frame(frame, modality)["clip_id"].astype(str).to_numpy()


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n$ " + " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                       check=True)


def latest_results(output_root: Path, modality: str) -> Path:
    results = list((output_root / modality).glob("cross_view_*/results.csv"))
    if not results:
        raise FileNotFoundError(
            f"No benchmark results found for {modality} under {output_root}")
    return max(results, key=lambda path: path.stat().st_mtime)


def train_fold(
    modality: str,
    architecture: str,
    fold: int,
    fold_manifest_dir: Path,
    fold_output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    output_root = fold_output_dir / "benchmark"
    command = [
        sys.executable,
        "scripts/13_benchmark_models.py",
        "--modality", modality,
        "--mode", "train",
        "--split", "cross_view",
        "--architectures", architecture,
        "--manifest-dir", str(fold_manifest_dir),
        "--run-root", "outputs/runs",
        "--output-root", str(output_root),
        "--seed", str(args.seed + fold),
        "--num-workers", str(args.num_workers),
        "--allow-camera-overlap",
    ]
    batch_size = {
        "skeleton": args.skeleton_batch_size,
        "imu": args.imu_batch_size,
        "rgb": args.rgb_batch_size,
    }[modality]
    command += ["--batch-size", str(batch_size)]
    epochs = {
        "skeleton": args.skeleton_epochs,
        "imu": args.imu_epochs,
        "rgb": args.rgb_epochs,
    }[modality]
    if epochs is not None:
        command += ["--epochs", str(epochs)]
    if args.max_samples is not None:
        command += ["--max-samples", str(args.max_samples)]
    run_command(command, fold_output_dir / f"{modality}.benchmark.log")
    result_path = latest_results(output_root, modality)
    results = pd.read_csv(result_path)
    if len(results) != 1:
        raise RuntimeError(f"Expected one {modality} OOF run in {result_path}")
    run_dir = Path(str(results.iloc[0]["run_dir"]))
    if not (run_dir / "best.pt").exists():
        raise FileNotFoundError(f"Fold checkpoint missing: {run_dir / 'best.pt'}")
    return run_dir


def collect_fold_prediction(
    run_dir: Path,
    val_manifest: Path,
    modality: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs_path = run_dir / "val_probs.npy"
    labels_path = run_dir / "val_labels.npy"
    if not probs_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"Missing validation predictions in {run_dir}")
    probs = np.load(probs_path)
    labels = np.load(labels_path).astype(np.int64)
    clip_ids = manifest_clip_ids(val_manifest, modality)
    if len(probs) != len(clip_ids) or len(labels) != len(clip_ids):
        raise ValueError(
            f"{modality} fold mismatch: probs={len(probs)}, labels={len(labels)}, "
            f"clip_ids={len(clip_ids)} in {run_dir}"
        )
    return probs.astype(np.float32), labels, clip_ids


def aggregate_oof(
    output_dir: Path,
    full_train_manifest: Path,
    modality: str,
    fold_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    selected_run_dir: Path,
) -> dict:
    full_ids = manifest_clip_ids(full_train_manifest, modality)
    full_frame = modality_frame(pd.read_csv(full_train_manifest), modality)
    full_labels = full_frame["class_id"].astype(np.int64).to_numpy()
    positions = {clip_id: index for index, clip_id in enumerate(full_ids)}
    oof_probs = np.full((len(full_ids), fold_data[0][0].shape[1]), np.nan,
                        dtype=np.float32)
    oof_labels = np.full(len(full_ids), -1, dtype=np.int64)
    seen = set()
    fold_summary = []
    for fold, (probs, labels, clip_ids) in enumerate(fold_data):
        for row, clip_id in enumerate(clip_ids):
            if clip_id not in positions:
                raise ValueError(f"Unknown OOF clip_id {clip_id} for {modality}")
            if clip_id in seen:
                raise ValueError(f"Duplicate OOF clip_id {clip_id} for {modality}")
            index = positions[clip_id]
            oof_probs[index] = probs[row]
            oof_labels[index] = labels[row]
            seen.add(clip_id)
        fold_summary.append({"fold": fold, "samples": int(len(clip_ids))})

    if len(seen) != len(full_ids) or np.isnan(oof_probs).any():
        missing = len(full_ids) - len(seen)
        raise RuntimeError(f"{modality} OOF incomplete: missing={missing}")
    if not np.array_equal(oof_labels, full_labels):
        raise RuntimeError(f"{modality} OOF labels do not match train manifest")

    selected_run_dir.mkdir(parents=True, exist_ok=True)
    np.save(selected_run_dir / "oof_train_probs.npy", oof_probs)
    np.save(selected_run_dir / "oof_train_labels.npy", oof_labels)
    np.save(selected_run_dir / "oof_train_clip_ids.npy", full_ids)
    aggregate_dir = output_dir / "aggregated" / modality
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    np.save(aggregate_dir / "oof_train_probs.npy", oof_probs)
    np.save(aggregate_dir / "oof_train_labels.npy", oof_labels)
    np.save(aggregate_dir / "oof_train_clip_ids.npy", full_ids)
    return {
        "run_dir": str(selected_run_dir),
        "samples": int(len(full_ids)),
        "classes": int(oof_probs.shape[1]),
        "folds": fold_summary,
        "files": [
            str(selected_run_dir / "oof_train_probs.npy"),
            str(selected_run_dir / "oof_train_labels.npy"),
            str(selected_run_dir / "oof_train_clip_ids.npy"),
        ],
        "aggregated_files": [
            str(aggregate_dir / "oof_train_probs.npy"),
            str(aggregate_dir / "oof_train_labels.npy"),
            str(aggregate_dir / "oof_train_clip_ids.npy"),
        ],
    }


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    manifest_dir = Path(args.manifest_dir)
    train_manifest = manifest_dir / "train_cross_view.csv"
    if not train_manifest.exists():
        raise FileNotFoundError(train_manifest)
    source = pd.read_csv(train_manifest)
    required = {"clip_id", "class_id", *EVENT_COLUMNS}
    required.update(f"{modality}_path" for modality in MODALITIES)
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Train manifest missing columns: {missing}")
    source = source.drop_duplicates(subset=["clip_id"]).reset_index(drop=True)
    if source["clip_id"].duplicated().any():
        raise ValueError("Duplicate clip_id in train manifest")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed)
    indices = np.arange(len(source))
    labels = source["class_id"].astype(int).to_numpy()
    groups = event_groups(source)
    fold_splits = list(splitter.split(indices, labels, groups))

    architectures = {
        "skeleton": args.skeleton_arch,
        "imu": args.imu_arch,
        "rgb": args.rgb_arch,
    }
    selected_runs = {
        "skeleton": Path(args.skeleton_run_dir),
        "imu": Path(args.imu_run_dir),
        "rgb": Path(args.rgb_run_dir),
    }
    fold_predictions = {modality: [] for modality in MODALITIES}
    fold_records = []

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        fold_dir = output_dir / f"fold_{fold:02d}"
        fold_manifest_dir = fold_dir / "manifests"
        fold_manifest_dir.mkdir(parents=True, exist_ok=True)
        train_part = source.iloc[train_idx].reset_index(drop=True)
        val_part = source.iloc[val_idx].reset_index(drop=True)
        train_part.to_csv(fold_manifest_dir / "train_cross_view.csv", index=False)
        val_part.to_csv(fold_manifest_dir / "val_cross_view.csv", index=False)

        record = {
            "fold": fold,
            "train_rows": int(len(train_part)),
            "val_rows": int(len(val_part)),
            "run_dirs": {},
        }
        for modality in MODALITIES:
            fold_run_dir = train_fold(
                modality, architectures[modality], fold,
                fold_manifest_dir, fold_dir, args)
            val_manifest = fold_manifest_dir / "val_cross_view.csv"
            fold_predictions[modality].append(
                collect_fold_prediction(fold_run_dir, val_manifest, modality))
            record["run_dirs"][modality] = str(fold_run_dir)
        fold_records.append(record)

    summary = {
        "manifest_dir": str(manifest_dir),
        "train_manifest": str(train_manifest),
        "folds": args.folds,
        "seed": args.seed,
        "architectures": architectures,
        "fold_records": fold_records,
        "modalities": {},
    }
    for modality in MODALITIES:
        summary["modalities"][modality] = aggregate_oof(
            output_dir, train_manifest, modality,
            fold_predictions[modality], selected_runs[modality])

    summary_path = output_dir / "oof_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nOOF complete: {summary_path}")
    for modality in MODALITIES:
        item = summary["modalities"][modality]
        print(f"  {modality}: {item['samples']} samples -> "
              f"{item['run_dir']}/oof_train_probs.npy")


if __name__ == "__main__":
    main()

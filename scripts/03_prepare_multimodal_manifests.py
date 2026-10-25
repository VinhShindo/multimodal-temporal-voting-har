"""Prepare complete RGB+Skeleton+IMU manifests for fair multimodal training.

The original MMAct manifests are modality-specific: some clips have no pose
or no sensor file.  This script creates a *new manifest directory* containing
only rows for which all requested modalities are present and whose files
actually exist.  The original manifests are never modified.

For the default cross-view protocol, the output files are:

    <output-dir>/train_cross_view.csv
    <output-dir>/val_cross_view.csv
    <output-dir>/multimodal_manifest_audit.json

The output filenames intentionally keep the standard names expected by the
existing skeleton, IMU, RGB and fusion scripts.  Pass the output directory to
those scripts with ``--manifest-dir``.

Example:
    python scripts/03_prepare_multimodal_manifests.py \
        --input-dir data/processed/MMAct/manifests \
        --output-dir data/processed/MMAct/manifests/multimodal_cross_view
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_MODALITIES = ("rgb", "skeleton", "imu")
DEFAULT_SPLIT = "cross_view"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create train/val manifests containing only complete multimodal "
            "RGB+Skeleton+IMU clips."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="data/processed/MMAct/manifests",
        help="Directory containing train_<split>.csv and val_<split>.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "data/processed/MMAct/manifests/multimodal_cross_view"
        ),
        help="New directory for the filtered manifests and audit.",
    )
    parser.add_argument(
        "--split",
        choices=("cross_view", "cross_scene"),
        default=DEFAULT_SPLIT,
        help="MMAct protocol to filter.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=DEFAULT_MODALITIES,
        default=list(DEFAULT_MODALITIES),
        help="Required modalities; default is all three modalities.",
    )
    parser.add_argument(
        "--allow-missing-files",
        action="store_true",
        help=(
            "Only require non-empty paths, without checking file existence. "
            "Not recommended for training."
        ),
    )
    return parser.parse_args()


def _path_is_valid(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and Path(text).is_file()


def _filter_manifest(
    frame: pd.DataFrame,
    modalities: list[str],
    check_files: bool,
) -> tuple[pd.DataFrame, dict]:
    required_columns = {"clip_id", "class_id"}
    required_columns.update(f"{modality}_path" for modality in modalities)
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(
            "Manifest thiếu cột bắt buộc: " + ", ".join(missing_columns)
        )

    original_rows = len(frame)
    keep = pd.Series(True, index=frame.index)
    modality_stats = {}
    for modality in modalities:
        column = f"{modality}_path"
        nonempty = frame[column].notna() & frame[column].astype(str).str.strip().ne("")
        existing = frame[column].map(_path_is_valid)
        modality_keep = nonempty if not check_files else existing
        keep &= modality_keep
        modality_stats[modality] = {
            "nonempty_paths": int(nonempty.sum()),
            "existing_files": int(existing.sum()),
            "rows_kept_after_this_modality": int(keep.sum()),
        }

    filtered = frame.loc[keep].copy()
    filtered["class_id"] = filtered["class_id"].astype(int)

    duplicate_ids = filtered.loc[
        filtered["clip_id"].duplicated(keep=False), "clip_id"
    ].astype(str).unique().tolist()
    if duplicate_ids:
        sample = duplicate_ids[:10]
        raise ValueError(
            f"Manifest có clip_id trùng sau lọc: {len(duplicate_ids)}; "
            f"ví dụ={sample}"
        )

    # Keep source order for deterministic dataset/prediction alignment.
    filtered = filtered.reset_index(drop=True)
    audit = {
        "input_rows": int(original_rows),
        "output_rows": int(len(filtered)),
        "removed_rows": int(original_rows - len(filtered)),
        "class_count": int(filtered["class_id"].nunique()),
        "class_counts": {
            str(k): int(v)
            for k, v in filtered["class_id"].value_counts().sort_index().items()
        },
        "modalities": modality_stats,
    }
    return filtered, audit


def _overlap_audit(train: pd.DataFrame, val: pd.DataFrame) -> dict:
    train_ids = set(train["clip_id"].astype(str))
    val_ids = set(val["clip_id"].astype(str))
    result = {
        "clip_id_overlap": int(len(train_ids & val_ids)),
        "train_unique_clip_ids": int(len(train_ids)),
        "val_unique_clip_ids": int(len(val_ids)),
    }

    event_columns = ["subject", "scene", "session", "action"]
    if all(column in train.columns and column in val.columns
           for column in event_columns):
        train_events = set(
            map(tuple, train[event_columns].drop_duplicates().to_numpy())
        )
        val_events = set(
            map(tuple, val[event_columns].drop_duplicates().to_numpy())
        )
        result["physical_event_overlap"] = int(len(train_events & val_events))
        result["train_unique_events"] = int(len(train_events))
        result["val_unique_events"] = int(len(val_events))

    if "cam" in train.columns and "cam" in val.columns:
        train_cameras = sorted(train["cam"].dropna().astype(str).unique().tolist())
        val_cameras = sorted(val["cam"].dropna().astype(str).unique().tolist())
        result["train_cameras"] = train_cameras
        result["val_cameras"] = val_cameras
        result["camera_overlap"] = sorted(set(train_cameras) & set(val_cameras))
    return result


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    modalities = list(dict.fromkeys(args.modalities))
    check_files = not args.allow_missing_files

    train_path = input_dir / f"train_{args.split}.csv"
    val_path = input_dir / f"val_{args.split}.csv"
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError(
            f"Cần có cả {train_path} và {val_path}"
        )

    train_source = pd.read_csv(train_path)
    val_source = pd.read_csv(val_path)
    train, train_audit = _filter_manifest(train_source, modalities, check_files)
    val, val_audit = _filter_manifest(val_source, modalities, check_files)

    overlap = _overlap_audit(train, val)
    if overlap["clip_id_overlap"]:
        raise RuntimeError(
            f"Phát hiện {overlap['clip_id_overlap']} clip_id trùng giữa train/val."
        )

    if train.empty or val.empty:
        raise RuntimeError(
            f"Manifest sau lọc bị rỗng: train={len(train)}, val={len(val)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_out = output_dir / f"train_{args.split}.csv"
    val_out = output_dir / f"val_{args.split}.csv"
    audit_out = output_dir / "multimodal_manifest_audit.json"
    train.to_csv(train_out, index=False)
    val.to_csv(val_out, index=False)

    audit = {
        "status": "PASS",
        "protocol": args.split,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "required_modalities": modalities,
        "file_existence_checked": check_files,
        "source_manifests": {
            "train": str(train_path),
            "val": str(val_path),
        },
        "output_manifests": {
            "train": str(train_out),
            "val": str(val_out),
        },
        "train": train_audit,
        "val": val_audit,
        "split_overlap": overlap,
        "notes": [
            "Original manifests were not modified.",
            "For cross_view, physical event overlap across cameras can be expected.",
            "The IMU loader applies its existing cross_view event purge policy.",
            "Use the same output directory for all three modality trainers and fusion.",
        ],
    }
    audit_out.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Multimodal manifests created successfully")
    print(f"  train: {train_out} ({len(train)} clips)")
    print(f"  val:   {val_out} ({len(val)} clips)")
    print(f"  audit: {audit_out}")
    print(f"  modalities: {', '.join(modalities)}")
    print(f"  files checked: {check_files}")
    if overlap.get("physical_event_overlap"):
        print(
            "  note: physical event overlap exists across cross-view cameras; "
            "this is expected for the current protocol."
        )


if __name__ == "__main__":
    main()

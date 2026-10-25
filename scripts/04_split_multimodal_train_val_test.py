"""Create a group-disjoint RGB+Skeleton+IMU train/val/test split.

The existing MMAct train/val manifests contain different modality coverage and
may contain multiple camera views of the same physical recording.  This script
combines the existing train and val rows, keeps only complete three-modality
clips, then creates a new 60/20/20 split grouped by
``subject/scene/session/action``.  Therefore all camera views of one physical
event stay in exactly one split.

Output files use the standard names expected by the training and fusion code:

    <output-dir>/train_cross_view.csv
    <output-dir>/val_cross_view.csv
    <output-dir>/test_cross_view.csv
    <output-dir>/multimodal_split_audit.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


EVENT_COLUMNS = ["subject", "scene", "session", "action"]
MODALITIES = ("rgb", "skeleton", "imu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a group-disjoint complete multimodal split."
    )
    parser.add_argument(
        "--input-dir", default="data/processed/MMAct/manifests",
        help="Directory containing the original train/val manifests.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/MMAct/manifests/multimodal_cross_view_3way",
        help="Directory for the new train/val/test manifests.",
    )
    parser.add_argument("--split", choices=["cross_view"], default="cross_view")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-missing-files", action="store_true",
        help="Only check non-empty paths; not recommended for training.",
    )
    return parser.parse_args()


def valid_path(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and Path(text).is_file()


def filter_complete(frame: pd.DataFrame, check_files: bool) -> pd.DataFrame:
    required = {"clip_id", "class_id", *EVENT_COLUMNS}
    required.update(f"{modality}_path" for modality in MODALITIES)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Manifest thiếu cột: {missing}")

    keep = pd.Series(True, index=frame.index)
    for modality in MODALITIES:
        values = frame[f"{modality}_path"]
        nonempty = values.notna() & values.astype(str).str.strip().ne("")
        keep &= values.map(valid_path) if check_files else nonempty
    output = frame.loc[keep].copy()
    output["class_id"] = output["class_id"].astype(int)
    if output["clip_id"].duplicated().any():
        raise ValueError("Có clip_id trùng trong dữ liệu multimodal")
    return output.reset_index(drop=True)


def group_keys(frame: pd.DataFrame) -> pd.Series:
    return frame[EVENT_COLUMNS].astype(str).agg("|".join, axis=1)


def split_group_stratified(
    frame: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, val, test with approximately 60/20/20 rows by groups."""
    groups = group_keys(frame)
    labels = frame["class_id"].to_numpy()
    indices = frame.index.to_numpy()

    # One of five folds is the final test set (~20%).
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(outer.split(indices, labels, groups))
    remaining = frame.iloc[train_val_idx].copy()

    # One of four folds in the remaining 80% is validation (~20% overall).
    inner_groups = group_keys(remaining)
    inner_labels = remaining["class_id"].to_numpy()
    inner_indices = remaining.index.to_numpy()
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True,
                                random_state=seed + 1)
    train_idx, val_idx = next(
        inner.split(inner_indices, inner_labels, inner_groups)
    )

    train = remaining.iloc[train_idx].copy()
    val = remaining.iloc[val_idx].copy()
    test = frame.iloc[test_idx].copy()
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def audit_split(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    frames = {"train": train, "val": val, "test": test}
    ids = {name: set(frame["clip_id"].astype(str)) for name, frame in frames.items()}
    groups = {name: set(group_keys(frame)) for name, frame in frames.items()}
    id_overlaps = {}
    group_overlaps = {}
    names = list(frames)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            id_overlaps[f"{left}_{right}"] = len(ids[left] & ids[right])
            group_overlaps[f"{left}_{right}"] = len(groups[left] & groups[right])

    return {
        "rows": {name: int(len(frame)) for name, frame in frames.items()},
        "classes": {
            name: int(frame["class_id"].nunique())
            for name, frame in frames.items()
        },
        "class_counts": {
            name: {str(k): int(v)
                   for k, v in frame["class_id"].value_counts().sort_index().items()}
            for name, frame in frames.items()
        },
        "clip_id_overlaps": id_overlaps,
        "physical_event_overlaps": group_overlaps,
        "cameras": {
            name: sorted(frame["cam"].dropna().astype(str).unique().tolist())
            for name, frame in frames.items() if "cam" in frame
        },
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    check_files = not args.allow_missing_files

    train_path = input_dir / f"train_{args.split}.csv"
    val_path = input_dir / f"val_{args.split}.csv"
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError(f"Cần có {train_path} và {val_path}")

    source = pd.concat([
        pd.read_csv(train_path),
        pd.read_csv(val_path),
    ], ignore_index=True)
    source_rows_before_filter = len(source)
    source = filter_complete(source, check_files)
    train, val, test = split_group_stratified(source, args.seed)

    audit = audit_split(train, val, test)
    if any(value for value in audit["clip_id_overlaps"].values()):
        raise RuntimeError(f"clip_id overlap detected: {audit['clip_id_overlaps']}")
    if any(value for value in audit["physical_event_overlaps"].values()):
        raise RuntimeError(
            "Physical-event overlap detected; split is not leakage-safe: "
            f"{audit['physical_event_overlaps']}"
        )
    if any(count != 35 for count in audit["classes"].values()):
        raise RuntimeError(
            "Một split không đủ 35 classes: " + str(audit["classes"])
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": output_dir / f"train_{args.split}.csv",
        "val": output_dir / f"val_{args.split}.csv",
        "test": output_dir / f"test_{args.split}.csv",
    }
    train.to_csv(outputs["train"], index=False)
    val.to_csv(outputs["val"], index=False)
    test.to_csv(outputs["test"], index=False)

    audit.update({
        "status": "PASS",
        "protocol": args.split,
        "seed": args.seed,
        "source_rows_before_filter": int(source_rows_before_filter),
        "source_rows_after_filter": int(len(source)),
        "file_existence_checked": check_files,
        "required_modalities": list(MODALITIES),
        "source_manifests": [str(train_path), str(val_path)],
        "output_manifests": {name: str(path) for name, path in outputs.items()},
        "split_policy": (
            "combine original train+val, filter complete RGB/Skeleton/IMU, "
            "then StratifiedGroupKFold by subject/scene/session/action"
        ),
    })
    audit_path = output_dir / "multimodal_split_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Created leakage-safe multimodal train/val/test split")
    for name, path in outputs.items():
        print(f"  {name}: {path} ({len(pd.read_csv(path))} clips)")
    print(f"  audit: {audit_path}")
    print(f"  file existence checked: {check_files}")


if __name__ == "__main__":
    main()

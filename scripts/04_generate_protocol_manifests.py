"""Generate modality-specific train/val manifests for the six MMAct protocols."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


MODALITIES = ("rgb", "skeleton", "imu")
EVENT_COLUMNS = ("subject", "scene", "session", "action")
PROTOCOL_GROUPS = {
    "x_sub_scene": ("subject", "scene"),
    "x_sub_view": ("subject", "cam"),
    "x_session_scene": ("subject", "scene", "session"),
    "x_session_view": ("subject", "cam", "session"),
}
PROTOCOLS = (
    "x_scene", "x_view", *PROTOCOL_GROUPS.keys(),
)
SOURCE_PROTOCOLS = {"x_scene": "cross_scene", "x_view": "cross_view"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build six MMAct protocol manifests, with train and val only."
    )
    parser.add_argument(
        "--source-dir", default="data/processed/MMAct/source_manifests",
        help="Directory containing train/val_cross_{scene,view}.csv.",
    )
    parser.add_argument(
        "--output-dir", default="data/processed/MMAct/manifests/protocols",
        help="Destination root; writes <protocol>/<modality>/{train,val}.csv.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--upload-complete", action="store_true",
        help="Assert that all expected raw data has finished uploading.",
    )
    return parser.parse_args()


def _read_sources(source_dir: Path) -> dict[str, pd.DataFrame]:
    sources: dict[str, pd.DataFrame] = {}
    for source_protocol in ("cross_scene", "cross_view"):
        paths = {
            subset: source_dir / f"{subset}_{source_protocol}.csv"
            for subset in ("train", "val")
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Required source manifests missing: " + ", ".join(missing))
        sources[source_protocol] = pd.concat(
            [pd.read_csv(paths["train"]), pd.read_csv(paths["val"])],
            ignore_index=True,
        )
    return sources


def _validate_source(frame: pd.DataFrame, name: str) -> None:
    required = {"clip_id", "class_id", *EVENT_COLUMNS, "cam"}
    required.update(f"{modality}_path" for modality in MODALITIES)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} source manifests lack columns: {missing}")
    if frame["clip_id"].isna().any():
        raise ValueError(f"{name} source contains empty clip_id values")
    frame["class_id"] = pd.to_numeric(frame["class_id"], errors="raise").astype(int)
    for column in (*EVENT_COLUMNS, "cam"):
        frame[column] = frame[column].fillna("").astype(str)
        if frame[column].eq("").any():
            raise ValueError(f"{name} source contains empty {column} values")


def _merge_sources(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pieces = []
    for source_protocol in ("cross_scene", "cross_view"):
        frame = sources[source_protocol].copy()
        _validate_source(frame, source_protocol)
        frame["_source_protocol"] = source_protocol
        pieces.append(frame)
    combined = pd.concat(pieces, ignore_index=True)

    identity_columns = ["class_id", *EVENT_COLUMNS, "cam"]
    for column in identity_columns:
        conflicts = combined.groupby("clip_id", dropna=False)[column].nunique(dropna=False)
        if (conflicts > 1).any():
            raise ValueError(f"Conflicting {column} values for duplicate clip_id")

    # Keep a deterministic source row and retain an available path from either source.
    path_values = {}
    for modality in MODALITIES:
        column = f"{modality}_path"
        path_values[column] = combined.groupby("clip_id", sort=False)[column].apply(
            lambda values: next(
                (str(value) for value in values if _path_present(value)),
                next((str(value) for value in values
                      if not pd.isna(value) and str(value).strip()), ""),
            )
        )
    combined = combined.drop_duplicates("clip_id", keep="first").copy()
    for column, values in path_values.items():
        combined[column] = combined["clip_id"].map(values).fillna("")
    return combined.drop(columns=["_source_protocol"]).reset_index(drop=True)


def _group_ids(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    if frame.empty:
        return pd.Series(index=frame.index, dtype=str)
    return frame[list(columns)].astype(str).agg("|".join, axis=1)


def split_group_stratified(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(frame) < 2:
        raise ValueError("At least two source rows are needed for a new protocol")
    groups = _group_ids(frame, columns)
    if groups.nunique() < 5:
        raise ValueError(
            f"Need at least five distinct groups for 80/20 split on {columns}; "
            f"found {groups.nunique()}"
        )
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, val_idx = next(
        splitter.split(frame.index.to_numpy(), frame["class_id"].to_numpy(), groups)
    )
    train = frame.iloc[train_idx].reset_index(drop=True)
    val = frame.iloc[val_idx].reset_index(drop=True)
    return train, val


def _path_present(value: Any) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and Path(text).is_file()


def _filter_modality(
    frame: pd.DataFrame,
    modality: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    path_column = f"{modality}_path"
    nonempty = frame[path_column].notna() & frame[path_column].astype(str).str.strip().ne("")
    existing = frame[path_column].map(_path_present)
    missing = int((~existing).sum())
    filtered = frame.loc[existing].copy().reset_index(drop=True)
    filtered["class_id"] = filtered["class_id"].astype(int)
    return filtered, {
        "input_rows": int(len(frame)),
        "nonempty_paths": int(nonempty.sum()),
        "existing_files": int(existing.sum()),
        "dropped_missing_paths": missing,
    }


def _deduplicate_imu(
    train: pd.DataFrame, val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    train = train.copy()
    val = val.copy()
    train_rows_before = len(train)
    val_rows_before = len(val)
    val_events = set(map(tuple, val[list(EVENT_COLUMNS)].to_numpy()))
    train_events = list(map(tuple, train[list(EVENT_COLUMNS)].to_numpy()))
    val_priority = pd.Series(
        [event not in val_events for event in train_events], index=train.index
    )
    purged = int((~val_priority).sum())
    train = train.loc[val_priority].drop_duplicates(
        subset=list(EVENT_COLUMNS), keep="first"
    ).reset_index(drop=True)
    val = val.drop_duplicates(
        subset=list(EVENT_COLUMNS), keep="first"
    ).reset_index(drop=True)

    train_events = set(map(tuple, train[list(EVENT_COLUMNS)].to_numpy()))
    val_events = set(map(tuple, val[list(EVENT_COLUMNS)].to_numpy()))
    if train_events & val_events:
        raise RuntimeError("IMU event deduplication left train/val overlap")

    # cam describes the RGB/pose row used as the alignment key, not an IMU camera.
    train = train.drop(columns=["cam"], errors="ignore")
    val = val.drop(columns=["cam"], errors="ignore")
    return train, val, {
        "train_events_purged_in_favor_of_val": purged,
        "train_rows_removed_by_deduplication": int(train_rows_before - len(train)),
        "val_rows_removed_by_deduplication": int(val_rows_before - len(val)),
    }


def _audit(
    protocol: str,
    modality: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    group_columns: tuple[str, ...] | None,
    path_stats: dict[str, dict[str, int]],
    imu_stats: dict[str, int] | None,
    seed: int,
) -> dict[str, Any]:
    event_overlap = 0
    if all(column in train.columns and column in val.columns
           for column in EVENT_COLUMNS):
        train_events = set(map(tuple, train[list(EVENT_COLUMNS)].to_numpy()))
        val_events = set(map(tuple, val[list(EVENT_COLUMNS)].to_numpy()))
        event_overlap = len(train_events & val_events)

    group_overlap = None
    group_counts = None
    if group_columns:
        train_groups = set(_group_ids(train, group_columns))
        val_groups = set(_group_ids(val, group_columns))
        group_overlap = len(train_groups & val_groups)
        group_counts = {
            "train": len(train_groups),
            "val": len(val_groups),
        }

    result: dict[str, Any] = {
        "protocol": protocol,
        "modality": modality,
        "seed": seed,
        "rows": {"train": len(train), "val": len(val)},
        "validation_row_fraction": (
            len(val) / (len(train) + len(val)) if len(train) + len(val) else None
        ),
        "class_counts": {
            subset: {
                str(key): int(value)
                for key, value in frame["class_id"].value_counts().sort_index().items()
            }
            for subset, frame in (("train", train), ("val", val))
        },
        "clip_id_overlap": len(
            set(train["clip_id"].astype(str)) & set(val["clip_id"].astype(str))
        ),
        "physical_event_overlap": event_overlap,
        "group_columns": list(group_columns) if group_columns else None,
        "group_counts": group_counts,
        "group_overlap": group_overlap,
        "path_coverage": path_stats,
    }
    if modality == "imu":
        result["imu_deduplication"] = imu_stats or {}
    if "cam" in train.columns and "cam" in val.columns:
        result["cameras"] = {
            subset: sorted(frame["cam"].astype(str).unique().tolist())
            for subset, frame in (("train", train), ("val", val))
        }
    if "scene" in train.columns and "scene" in val.columns:
        train_scenes = set(train["scene"].dropna().astype(str))
        val_scenes = set(val["scene"].dropna().astype(str))
        result["scenes"] = {
            "train": sorted(train_scenes),
            "val": sorted(val_scenes),
            "overlap": sorted(train_scenes & val_scenes),
        }
    return result


def _write_alignment_index(protocol_dir: Path) -> None:
    alignment_dir = protocol_dir / "alignment"
    alignment_dir.mkdir(parents=True, exist_ok=True)
    metadata_columns = ("clip_id", "class_id", *EVENT_COLUMNS, "cam")
    for subset in ("train", "val"):
        records: dict[str, dict[str, Any]] = {}
        for modality in MODALITIES:
            frame = pd.read_csv(protocol_dir / modality / f"{subset}.csv")
            for _, row in frame.iterrows():
                clip_id = str(row["clip_id"])
                record = records.setdefault(clip_id, {"clip_id": clip_id})
                for column in metadata_columns:
                    if column in frame.columns and not pd.isna(row[column]):
                        if column in record and str(record[column]) != str(row[column]):
                            raise ValueError(
                                f"Conflicting {column} for alignment clip {clip_id}"
                            )
                        record[column] = row[column]
                record[f"{modality}_path"] = row.get(f"{modality}_path", "")
        alignment = pd.DataFrame(records.values())
        for modality in MODALITIES:
            column = f"{modality}_path"
            if column not in alignment:
                alignment[column] = ""
            alignment[column] = alignment[column].fillna("")
        alignment.to_csv(alignment_dir / f"{subset}.csv", index=False)


def generate(
    source_dir: Path,
    output_dir: Path,
    seed: int = 42,
    upload_complete: bool = False,
) -> dict[str, Any]:
    sources = _read_sources(source_dir)
    canonical = _merge_sources(sources)
    output_audit: dict[str, Any] = {
        "status": "PASS" if upload_complete else "INCOMPLETE",
        "upload_complete_asserted": upload_complete,
        "seed": seed,
        "source_dir": str(source_dir),
        "protocols": {},
        "notes": [
            "Only train and val manifests are generated; no internal test split exists.",
            "x_scene and x_view retain the source split assignments.",
            "IMU is deduplicated by physical event and validation takes priority.",
        ],
    }

    for protocol in PROTOCOLS:
        if protocol in SOURCE_PROTOCOLS:
            source_protocol = SOURCE_PROTOCOLS[protocol]
            source_train_path = source_dir / f"train_{source_protocol}.csv"
            source_val_path = source_dir / f"val_{source_protocol}.csv"
            protocol_frames = {
                "train": pd.read_csv(source_train_path),
                "val": pd.read_csv(source_val_path),
            }
            for frame in protocol_frames.values():
                _validate_source(frame, source_protocol)
            group_columns = (
                ("subject", "cam") if protocol == "x_view"
                else ("subject", "scene", "session")
            )
        else:
            group_columns = PROTOCOL_GROUPS[protocol]
            protocol_frames = dict(zip(
                ("train", "val"),
                split_group_stratified(canonical, group_columns, seed),
            ))

        protocol_audit: dict[str, Any] = {}
        for modality in MODALITIES:
            filtered: dict[str, pd.DataFrame] = {}
            path_stats: dict[str, dict[str, int]] = {}
            for subset, frame in protocol_frames.items():
                filtered[subset], path_stats[subset] = _filter_modality(
                    frame, modality
                )
                if filtered[subset].empty and upload_complete:
                    raise ValueError(
                        f"{protocol}/{modality}/{subset} is empty after path filtering"
                    )

            imu_stats = None
            if modality == "imu":
                filtered["train"], filtered["val"], imu_stats = _deduplicate_imu(
                    filtered["train"], filtered["val"]
                )
                if (
                    (filtered["train"].empty or filtered["val"].empty)
                    and upload_complete
                ):
                    raise ValueError(
                        f"{protocol}/imu became empty after physical-event deduplication"
                    )

            protocol_dir = output_dir / protocol / modality
            protocol_dir.mkdir(parents=True, exist_ok=True)
            for subset in ("train", "val"):
                filtered[subset].to_csv(protocol_dir / f"{subset}.csv", index=False)

            item = _audit(
                protocol, modality, filtered["train"], filtered["val"],
                None if modality == "imu" else group_columns,
                path_stats, imu_stats, seed,
            )
            if item["clip_id_overlap"]:
                raise RuntimeError(f"Leakage detected in {protocol}/{modality}: {item}")
            if modality == "imu" and item["physical_event_overlap"]:
                raise RuntimeError(f"IMU event leakage in {protocol}: {item}")
            if item["group_overlap"] not in (None, 0):
                raise RuntimeError(f"Group leakage detected in {protocol}/{modality}: {item}")
            protocol_audit[modality] = item
        _write_alignment_index(output_dir / protocol)
        output_audit["protocols"][protocol] = protocol_audit

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "protocol_manifest_audit.json"
    audit_path.write_text(
        json.dumps(output_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_audit


def main() -> None:
    args = parse_args()
    audit = generate(
        Path(args.source_dir), Path(args.output_dir),
        seed=args.seed, upload_complete=args.upload_complete,
    )
    print(f"Generated six train/val protocols: {args.output_dir}")
    for protocol, modalities in audit["protocols"].items():
        counts = ", ".join(
            f"{modality}={data['rows']['train']}/{data['rows']['val']}"
            for modality, data in modalities.items()
        )
        print(f"  {protocol}: {counts}")
    print(f"  status: {audit['status']}")
    print(f"  audit: {Path(args.output_dir) / 'protocol_manifest_audit.json'}")


if __name__ == "__main__":
    main()

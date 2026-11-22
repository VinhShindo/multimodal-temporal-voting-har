"""
Align predictions từ 3 model (skeleton/imu/rgb) theo clip_id.

Mỗi run folder chứa:
  - train_probs.npy / val_probs.npy    (N, C)
  - train_labels.npy / val_labels.npy  (N,)

Reconstruct clip_id dựa trên filter logic của dataset tương ứng:
  - skeleton: rows có skeleton_path != ""
  - imu:      rows có imu_path != ""
  - rgb:      rows có rgb_path != ""
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_class_names(classes_yaml: str = "configs/classes_mmact.yaml") -> list:
    with open(classes_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    id2name = {int(k): v for k, v in cfg["classes"].items()}
    return [id2name[i] for i in range(int(cfg["num_classes"]))]


def get_run_architecture(run_dir: Path) -> str | None:
    """Read the architecture name saved by each modality trainer."""
    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"
    metadata_path = run_dir / "metadata.json"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) \
            if config_path.exists() else {}
        config = config or {}
        model_cfg = config.get("model", {}) or {}
        if model_cfg.get("arch"):
            return str(model_cfg["arch"])
        if config.get("arch"):
            return str(config["arch"])
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return str(metadata.get("arch")) if metadata.get("arch") else None
    except (OSError, ValueError, TypeError, yaml.YAMLError, json.JSONDecodeError):
        return None
    return None


def manifest_protocol_split(manifest_csv: Path) -> str:
    """Return the protocol encoded in the protocol directory."""
    stem = Path(manifest_csv).stem
    protocol = Path(manifest_csv).parent.parent.name
    if protocol in {
        "x_scene", "x_view", "x_sub_scene", "x_sub_view",
        "x_session_scene", "x_session_view",
    }:
        return protocol
    for prefix in ("train_", "val_", "test_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    raise ValueError(
        f"Cannot determine protocol from manifest path: {manifest_csv}")


def find_best_compatible_run(runs_root: Path, modality: str,
                             manifest_csv: Path, log_fn=None,
                             architectures=None):
    """Choose the highest-F1 completed run compatible with this validation manifest."""
    mod_root = Path(runs_root) / modality
    if not mod_root.exists():
        return None
    manifest_split = manifest_protocol_split(manifest_csv)
    compatible = []
    for run_dir in (d for d in mod_root.iterdir() if d.is_dir()):
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.yaml"
        metadata_path = run_dir / "metadata.json"
        if not (run_dir / "val_probs.npy").exists() or not metrics_path.exists():
            continue
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or \
                     {} if config_path.exists() else {}
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) \
                       if metadata_path.exists() else {}
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            score = float(metrics["val"]["f1_macro"])
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
            continue

        architecture = get_run_architecture(run_dir)
        allowed_architectures = architectures.get(modality) if architectures else None
        if allowed_architectures:
            allowed_architectures = set(map(str, allowed_architectures))
            if architecture not in allowed_architectures:
                if log_fn:
                    log_fn(f"  [skip] {modality}: {run_dir.name} arch={architecture} "
                           f"(allowed: {sorted(allowed_architectures)})")
                continue

        run_split = config.get(
            "protocol",
            config.get("split", metadata.get("protocol", metadata.get("split"))),
        )
        if modality == "imu" and not run_split:
            if log_fn:
                log_fn(f"  [skip] imu: {run_dir.name} has no split metadata; "
                       "cannot certify a leakage-safe fusion run")
            continue
        if run_split and run_split != manifest_split:
            if log_fn:
                log_fn(f"  [skip] {modality}: {run_dir.name} split={run_split} "
                       f"(requested {manifest_split})")
            continue
        try:
            predictions = load_run_predictions(run_dir, modality, manifest_csv)
        except (OSError, ValueError, KeyError) as error:
            if log_fn:
                log_fn(f"  [skip] {modality}: {run_dir.name}: {error}")
            continue

        eval_prefix = Path(manifest_csv).stem.split("_", 1)[0]
        expected_samples = metrics.get(eval_prefix, {}).get("num_samples")
        if expected_samples is not None and int(expected_samples) != len(predictions):
            if log_fn:
                log_fn(f"  [skip] {modality}: {run_dir.name} metrics report "
                       f"{expected_samples} validation samples, found {len(predictions)}")
            continue
        compatible.append((score, run_dir.name, run_dir, predictions, not run_split))

    if not compatible:
        return None
    compatible.sort(key=lambda item: (item[0], item[1]), reverse=True)
    score, _, run_dir, predictions, legacy_split = compatible[0]
    return run_dir, predictions, score, legacy_split


def reconstruct_clip_ids(manifest_csv: Path, modality: str) -> list:
    """Trả về list clip_id theo đúng thứ tự dataset filter."""
    manifest_csv = Path(manifest_csv)
    if manifest_csv.parent.name == "alignment":
        modality_manifest = (
            manifest_csv.parent.parent / modality / manifest_csv.name
        )
        if modality_manifest.is_file():
            manifest_csv = modality_manifest
    df = pd.read_csv(manifest_csv)

    if modality == "skeleton":
        mask = df["skeleton_path"].notna() & (df["skeleton_path"] != "")
    elif modality == "imu":
        mask = df["imu_path"].notna() & (df["imu_path"] != "")
    elif modality == "rgb":
        mask = df["rgb_path"].notna() & (df["rgb_path"] != "")
    else:
        raise ValueError(f"Unknown modality: {modality}")

    df_mod = df[mask].copy()
    if modality == "imu":
        event_columns = ["subject", "scene", "session", "action"]
        df_mod = df_mod.drop_duplicates(subset=event_columns)
    df_mod = df_mod.reset_index(drop=True)
    return df_mod["clip_id"].tolist()


def load_run_predictions(run_dir: Path, modality: str,
                         manifest_csv: Path, prediction_prefix: str | None = None):
    """
    Load 1 run → dict {clip_id: {"probs": ..., "label": ...}}.
    Raise ValueError nếu không align được.
    """
    run_dir = Path(run_dir)
    split = Path(manifest_csv).stem.split("_", 1)[0]
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Manifest phải bắt đầu bằng train_, val_ hoặc test_: {manifest_csv}")
    prefix = prediction_prefix or split
    probs_path = run_dir / f"{prefix}_probs.npy"
    labels_path = run_dir / f"{prefix}_labels.npy"
    if not probs_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Thiếu {prefix}_probs/{prefix}_labels trong {run_dir}")

    probs = np.load(probs_path)
    labels = np.load(labels_path)

    if probs.ndim != 2 or labels.ndim != 1 or len(probs) != len(labels):
        raise ValueError(
            f"Invalid prediction shapes in {run_dir}: "
            f"{prefix}_probs={probs.shape}, {prefix}_labels={labels.shape}"
        )

    # Train/OOF predictions must carry their own IDs because their order is
    # not safely reconstructable from a manifest after sampling/folding.
    if split == "train" or prediction_prefix is not None:
        clip_ids_path = run_dir / f"{prefix}_clip_ids.npy"
        if not clip_ids_path.exists():
            raise FileNotFoundError(
                f"Thiếu {prefix}_clip_ids.npy trong {run_dir}; {prefix}_probs "
                "không có thứ tự clip an toàn để fit fusion")
        clip_ids = np.load(clip_ids_path, allow_pickle=True).tolist()
    else:
        clip_ids = reconstruct_clip_ids(manifest_csv, modality)

    if len(clip_ids) != len(probs):
        raise ValueError(
            f"Length mismatch cho {modality}: clip_ids = {len(clip_ids)}, "
            f"run {prefix}_probs = {len(probs)}. Có thể do run dùng --max-samples."
        )

    out = {}
    for i, cid in enumerate(clip_ids):
        out[cid] = {
            "probs": probs[i].astype(np.float32),
            "label": int(labels[i]),
        }
    return out


def align_selected_runs(run_dirs: dict, manifest_csv: Path,
                        modalities=None, log_fn=None,
                        prediction_prefix: str | None = None):
    """Align a specific set of already selected runs on one manifest split.

    This is used by learned fusion: the validation run is selected by its
    validation F1, then the *same run directory* is loaded on the train
    manifest to fit the meta-fuser.  Selecting a different run for train would
    silently mix incompatible model checkpoints.
    """
    def log(message):
        (log_fn or print)(message)

    modalities = list(modalities or run_dirs.keys())
    per_mod = {}
    for mod in modalities:
        run_dir = run_dirs.get(mod)
        if run_dir is None:
            continue
        try:
            data = load_run_predictions(
                run_dir, mod, manifest_csv, prediction_prefix=prediction_prefix)
        except (OSError, ValueError, KeyError) as error:
            log(f"  [skip] {mod}: {run_dir}: {error}")
            continue
        per_mod[mod] = data
        log(f"  [ok]   {mod}: {len(data)} clips from {Path(run_dir).name}")

    if not per_mod:
        raise RuntimeError(
            f"Không load được predictions trên manifest {manifest_csv}")

    common = set.intersection(*(set(data) for data in per_mod.values()))
    if not common:
        raise RuntimeError(
            f"Không có clip_id chung trên manifest {manifest_csv}")
    clip_ids = sorted(common)
    aligned_probs = {
        mod: np.stack([data[cid]["probs"] for cid in clip_ids], axis=0)
        for mod, data in per_mod.items()
    }
    first_mod = next(iter(per_mod))
    labels = np.asarray(
        [per_mod[first_mod][cid]["label"] for cid in clip_ids], dtype=np.int64)
    for mod, data in per_mod.items():
        other = np.asarray([data[cid]["label"] for cid in clip_ids], dtype=np.int64)
        if not np.array_equal(labels, other):
            raise RuntimeError(
                f"Labels không khớp giữa modalities trên {manifest_csv} ({mod})")

    return aligned_probs, labels, clip_ids, {
        "modalities": list(per_mod),
        "n_per_modality": {mod: len(data) for mod, data in per_mod.items()},
        "n_common": len(clip_ids),
        "clip_ids": clip_ids,
    }


def align_all(runs_root: Path, manifest_csv: Path,
              modalities=("skeleton", "imu", "rgb"),
              log_fn=None, architectures=None):
    """
    Load tất cả modality available, align theo intersection clip_id.
    Return:
        aligned_probs:  dict[modality] = (N, C) ndarray
        aligned_labels: (N,) ndarray
        clip_ids:       list[str] length N
        info:           dict metadata
    """
    def log(m):
        (log_fn or print)(m)

    per_mod = {}   # modality → {clip_id: {probs, label}}
    selected_run_dirs = {}
    for mod in modalities:
        selection = find_best_compatible_run(
            runs_root, mod, manifest_csv, log_fn=log_fn,
            architectures=architectures)
        if selection is None:
            log(f"  [skip] {mod}: không có run hoàn chỉnh, leakage-safe và "
                f"tương thích với {manifest_protocol_split(manifest_csv)} "
                f"trong {runs_root}/{mod}")
            continue
        run_dir, data, score, legacy_split = selection
        per_mod[mod] = data
        selected_run_dirs[mod] = run_dir
        split_note = "; legacy run has no split metadata" if legacy_split else ""
        log(f"  [ok]   {mod}: {len(data)} clips  "
            f"(best compatible run={run_dir.name}, val_f1={score:.4f}{split_note})")

    if not per_mod:
        raise RuntimeError("Không load được run nào.")

    # Intersection của clip_id
    sets = [set(d.keys()) for d in per_mod.values()]
    common = set.intersection(*sets)
    if not common:
        raise RuntimeError("Không có clip_id nào chung giữa các modality.")

    clip_ids = sorted(common)
    N = len(clip_ids)

    aligned_probs = {}
    for mod, data in per_mod.items():
        arr = np.stack([data[cid]["probs"] for cid in clip_ids], axis=0)
        aligned_probs[mod] = arr

    # Label — verify consistent
    labels_list = [per_mod[list(per_mod.keys())[0]][cid]["label"] for cid in clip_ids]
    labels = np.asarray(labels_list, dtype=np.int64)
    for mod, data in per_mod.items():
        other = np.asarray([data[cid]["label"] for cid in clip_ids], dtype=np.int64)
        if not np.array_equal(labels, other):
            raise RuntimeError(f"Labels không khớp giữa modalities (modality={mod}).")

    info = {
        "modalities": list(per_mod.keys()),
        "n_per_modality": {m: len(d) for m, d in per_mod.items()},
        "n_common": N,
        "clip_ids": clip_ids,
        "run_dirs": selected_run_dirs,
        "architectures": {
            mod: get_run_architecture(run_dir)
            for mod, run_dir in selected_run_dirs.items()
        },
    }
    return aligned_probs, labels, clip_ids, info

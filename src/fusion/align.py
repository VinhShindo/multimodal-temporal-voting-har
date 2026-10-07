"""
Align predictions từ 3 model (skeleton/imu/rgb) theo clip_id.

Mỗi run folder chứa:
  - val_probs.npy    (N, 32)
  - val_labels.npy   (N,)

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


def find_latest_run(runs_root: Path, modality: str) -> Path:
    """Tìm run mới nhất của 1 modality trong outputs/runs/{modality}/."""
    mod_root = Path(runs_root) / modality
    if not mod_root.exists():
        return None
    runs = [d for d in mod_root.iterdir() if d.is_dir()]
    if not runs:
        return None
    runs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return runs[0]


def reconstruct_clip_ids(manifest_csv: Path, modality: str) -> list:
    """Trả về list clip_id theo đúng thứ tự dataset filter."""
    df = pd.read_csv(manifest_csv)

    if modality == "skeleton":
        mask = df["skeleton_path"].notna() & (df["skeleton_path"] != "")
    elif modality == "imu":
        mask = df["imu_path"].notna() & (df["imu_path"] != "")
    elif modality == "rgb":
        mask = df["rgb_path"].notna() & (df["rgb_path"] != "")
    else:
        raise ValueError(f"Unknown modality: {modality}")

    df_mod = df[mask].reset_index(drop=True)
    return df_mod["clip_id"].tolist()


def load_run_predictions(run_dir: Path, modality: str,
                         manifest_csv: Path):
    """
    Load 1 run → dict {clip_id: {"probs": ..., "label": ...}}.
    Raise ValueError nếu không align được.
    """
    run_dir = Path(run_dir)
    probs_path = run_dir / "val_probs.npy"
    labels_path = run_dir / "val_labels.npy"
    if not probs_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"Thiếu val_probs/val_labels trong {run_dir}")

    probs = np.load(probs_path)
    labels = np.load(labels_path)

    clip_ids = reconstruct_clip_ids(manifest_csv, modality)

    if len(clip_ids) != len(probs):
        raise ValueError(
            f"Length mismatch cho {modality}: manifest filter = {len(clip_ids)}, "
            f"run val_probs = {len(probs)}. Có thể do run dùng --max-samples."
        )

    out = {}
    for i, cid in enumerate(clip_ids):
        out[cid] = {
            "probs": probs[i].astype(np.float32),
            "label": int(labels[i]),
        }
    return out


def align_all(runs_root: Path, manifest_csv: Path,
              modalities=("skeleton", "imu", "rgb"),
              log_fn=None):
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
    for mod in modalities:
        run_dir = find_latest_run(runs_root, mod)
        if run_dir is None:
            log(f"  [skip] {mod}: không có run nào trong {runs_root}/{mod}")
            continue
        try:
            data = load_run_predictions(run_dir, mod, manifest_csv)
            per_mod[mod] = data
            log(f"  [ok]   {mod}: {len(data)} clips  (run={run_dir.name})")
        except Exception as e:
            log(f"  [skip] {mod}: {e}")

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
    }
    return aligned_probs, labels, clip_ids, info
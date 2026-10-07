"""Verify processed data — kiểm tra file tồn tại, shape, NaN."""
from pathlib import Path

import numpy as np
import pandas as pd


def verify_processed(manifest_csv: Path, log_fn=None,
                     modalities=("skeleton", "imu", "rgb"),
                     max_check: int = 300) -> dict:
    def log(m):
        (log_fn or print)(m)

    df = pd.read_csv(manifest_csv)
    summary = {}

    for mod in modalities:
        col = f"{mod}_path"
        if col not in df.columns:
            log(f"    [{mod}] cột {col} không tồn tại — skip")
            continue

        valid = df[df[col].notna() & (df[col] != "")]
        n = len(valid)
        pct = 100.0 * n / len(df) if len(df) else 0
        log(f"    [{mod}] có path: {n}/{len(df)} ({pct:.1f}%)")

        if n == 0:
            summary[mod] = {"n_paths": 0}
            continue

        # Check a sample
        sample = valid.sample(n=min(max_check, n), random_state=42)
        stats = {"n_paths": n, "n_checked": len(sample),
                 "missing_file": 0, "nan_found": 0, "ok": 0}

        shapes = []
        for p in sample[col]:
            f = Path(p)
            if not f.exists():
                stats["missing_file"] += 1
                continue
            try:
                if mod == "rgb" and f.suffix == ".npy":
                    arr = np.load(f, mmap_mode="r")
                elif mod == "rgb":
                    # Raw video — không load
                    stats["ok"] += 1
                    continue
                else:
                    arr = np.load(f)
            except Exception:
                stats["missing_file"] += 1
                continue

            if np.isnan(arr).any() if hasattr(arr, "shape") else False:
                stats["nan_found"] += 1
                continue
            shapes.append(arr.shape)
            stats["ok"] += 1

        if shapes:
            arr = np.array(shapes)
            stats["shape_min"] = arr.min(axis=0).tolist()
            stats["shape_max"] = arr.max(axis=0).tolist()
            stats["shape_mean"] = arr.mean(axis=0).round(1).tolist()

        log(f"    [{mod}] ok={stats['ok']}/{len(sample)}  "
            f"missing={stats['missing_file']}  nan={stats['nan_found']}")
        if "shape_mean" in stats:
            log(f"    [{mod}] shape trung bình: {stats['shape_mean']}")

        summary[mod] = stats

    return summary
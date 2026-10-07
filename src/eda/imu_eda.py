"""IMU EDA — MMAct acc_watch clip."""
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eda.common import plot_class_distribution, save_fig, save_stats


CHANNEL_NAMES = ["ax", "ay", "az"]


# ============================================================
# 1. Class distribution
# ============================================================
def analyze_class_distribution(df: pd.DataFrame, class_names: list,
                                out_dir: Path) -> dict:
    return plot_class_distribution(
        df, class_names,
        out_dir / "class_distribution.png",
        title="IMU — Class Distribution",
    )


# ============================================================
# 2. Sequence length
# ============================================================
def analyze_sequence_length(df: pd.DataFrame, out_dir: Path,
                             n_samples: int = 500) -> dict:
    sample = df.sample(n=min(n_samples, len(df)), random_state=42)

    lengths = []
    for p in sample["imu_path"]:
        try:
            sig = np.load(p)
            lengths.append(sig.shape[0])
        except Exception:
            continue

    if not lengths:
        return {"error": "No data"}

    lengths = np.array(lengths)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(lengths, bins=40, color="#4C72B0", edgecolor="black", alpha=0.8)
    ax.axvline(np.mean(lengths), color="red", linestyle="--",
               label=f"mean={np.mean(lengths):.0f}")
    ax.axvline(np.median(lengths), color="orange", linestyle="--",
               label=f"median={np.median(lengths):.0f}")
    ax.set_xlabel("Samples per clip")
    ax.set_ylabel("Frequency")
    ax.set_title(f"IMU Sequence Length (sampled n={len(lengths)})")
    ax.legend()
    save_fig(fig, out_dir / "sequence_length.png")

    return {
        "n_sampled": len(lengths),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
    }


# ============================================================
# 3. Channel stats
# ============================================================
def analyze_channel_stats(df: pd.DataFrame, out_dir: Path,
                           n_samples: int = 500) -> dict:
    sample = df.sample(n=min(n_samples, len(df)), random_state=42)

    all_sig = []
    for p in sample["imu_path"]:
        try:
            sig = np.load(p)
            if sig.ndim == 2:
                all_sig.append(sig)
        except Exception:
            continue

    if not all_sig:
        return {"error": "No data"}

    concat = np.concatenate(all_sig, axis=0)  # (N, 3)
    n_ch = concat.shape[1]

    stats = {}
    for i in range(n_ch):
        col = concat[:, i]
        stats[CHANNEL_NAMES[i] if i < len(CHANNEL_NAMES) else f"ch{i}"] = {
            "mean": float(col.mean()),
            "std":  float(col.std()),
            "min":  float(col.min()),
            "max":  float(col.max()),
            "p5":   float(np.percentile(col, 5)),
            "p95":  float(np.percentile(col, 95)),
        }

    # Plot histogram
    fig, axes = plt.subplots(1, n_ch, figsize=(5 * n_ch, 4))
    if n_ch == 1:
        axes = [axes]
    for ax, i in zip(axes, range(n_ch)):
        col = concat[:, i]
        name = CHANNEL_NAMES[i] if i < len(CHANNEL_NAMES) else f"ch{i}"
        ax.hist(col, bins=80, color="#4C72B0", alpha=0.8)
        ax.axvline(col.mean(), color="red", linestyle="--")
        ax.set_xlabel(name)
        ax.set_ylabel("Frequency")
        ax.set_title(f"{name}: μ={col.mean():.2f}, σ={col.std():.2f}")
    save_fig(fig, out_dir / "channel_stats.png")

    return {"n_samples_analyzed": len(all_sig),
            "n_channel_samples": int(concat.shape[0]),
            "channels": stats}


# ============================================================
# 4. Signal plot per class
# ============================================================
def plot_signals_per_class(df: pd.DataFrame, class_names: list,
                            out_dir: Path, n_classes: int = 8,
                            max_len: int = 300):
    """Vẽ 1 sample signal per class (3 kênh x/y/z)."""
    selected = []
    for c in range(len(class_names)):
        cls_df = df[df["class_id"] == c]
        if len(cls_df) == 0:
            continue
        selected.append((c, cls_df.iloc[0]))
        if len(selected) >= n_classes:
            break

    if not selected:
        return

    n = len(selected)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n))
    if n == 1:
        axes = [axes]

    for ax, (cid, row) in zip(axes, selected):
        try:
            sig = np.load(row["imu_path"])
        except Exception:
            continue

        sig = sig[:max_len]
        t = np.arange(len(sig))
        ax.plot(t, sig[:, 0], label="ax", alpha=0.8)
        ax.plot(t, sig[:, 1], label="ay", alpha=0.8)
        ax.plot(t, sig[:, 2], label="az", alpha=0.8)
        ax.set_ylabel("Acc (m/s²)")
        ax.set_title(f"C{cid}: {class_names[cid]}", loc="left", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Sample index")
    save_fig(fig, out_dir / "signals_per_class.png")


# ============================================================
# 5. Sampling rate check
# ============================================================
def check_sampling_rate(df: pd.DataFrame, out_dir: Path,
                         n_samples: int = 200) -> dict:
    """Ước lượng sampling rate từ độ dài sequence (giả sử action ~9s)."""
    sample = df.sample(n=min(n_samples, len(df)), random_state=42)

    lengths = []
    for p in sample["imu_path"]:
        try:
            sig = np.load(p)
            lengths.append(sig.shape[0])
        except Exception:
            continue

    if not lengths:
        return {"error": "No data"}

    lengths = np.array(lengths)
    # Giả định clip ~9s
    durations = lengths / 50.0  # dataset resample về 50Hz

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(durations, bins=40, color="#55A868", edgecolor="black", alpha=0.8)
    ax.axvline(durations.mean(), color="red", linestyle="--",
               label=f"mean={durations.mean():.2f}s")
    ax.set_xlabel("Duration (giây, giả định 50 Hz)")
    ax.set_ylabel("Frequency")
    ax.set_title("IMU Clip Duration")
    ax.legend()
    save_fig(fig, out_dir / "duration_distribution.png")

    return {
        "assumed_hz": 50,
        "mean_sec": float(durations.mean()),
        "median_sec": float(np.median(durations)),
        "min_sec": float(durations.min()),
        "max_sec": float(durations.max()),
    }


# ============================================================
# 6. Quality check
# ============================================================
def quality_check(df: pd.DataFrame, n_samples: int = 300) -> dict:
    sample = df.sample(n=min(n_samples, len(df)), random_state=42)
    issues = {"missing_file": 0, "bad_shape": 0, "nan_found": 0,
              "too_short": 0, "ok": 0}

    for p in sample["imu_path"]:
        try:
            sig = np.load(p)
        except Exception:
            issues["missing_file"] += 1
            continue
        if sig.ndim != 2:
            issues["bad_shape"] += 1
            continue
        if np.isnan(sig).any():
            issues["nan_found"] += 1
            continue
        if sig.shape[0] < 10:
            issues["too_short"] += 1
            continue
        issues["ok"] += 1

    issues["n_checked"] = len(sample)
    issues["ok_pct"] = 100.0 * issues["ok"] / max(len(sample), 1)
    return issues


# ============================================================
# Orchestrator
# ============================================================
def run_imu_eda(manifest_csv: Path, class_names: list,
                out_dir: Path, log_fn=None) -> dict:
    def log(m):
        (log_fn or print)(m)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_csv)
    df = df[df["imu_path"].notna() & (df["imu_path"] != "")].copy()
    df["class_id"] = df["class_id"].astype(int)

    log(f"  Loaded {len(df)} IMU clips")

    stats = {"n_clips": len(df)}

    log("  [1/6] Class distribution...")
    stats["class_distribution"] = analyze_class_distribution(df, class_names, out_dir)

    log("  [2/6] Sequence length...")
    stats["sequence_length"] = analyze_sequence_length(df, out_dir)

    log("  [3/6] Channel stats...")
    stats["channel_stats"] = analyze_channel_stats(df, out_dir)

    log("  [4/6] Signal plot per class...")
    plot_signals_per_class(df, class_names, out_dir)

    log("  [5/6] Sampling rate...")
    stats["sampling_rate"] = check_sampling_rate(df, out_dir)

    log("  [6/6] Quality check...")
    stats["quality"] = quality_check(df)

    save_stats(stats, out_dir / "stats.json")
    log(f"  ✓ Saved → {out_dir}")
    return stats
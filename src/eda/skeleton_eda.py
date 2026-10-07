"""Skeleton EDA — MMAct COCO 17 keypoints."""
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eda.common import plot_class_distribution, save_fig, save_stats


# COCO 17 edges cho visualization
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

JOINT_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]


# ============================================================
# 1. Class distribution
# ============================================================
def analyze_class_distribution(df: pd.DataFrame, class_names: list,
                                out_dir: Path) -> dict:
    return plot_class_distribution(
        df, class_names,
        out_dir / "class_distribution.png",
        title="Skeleton — Class Distribution",
    )


# ============================================================
# 2. Sequence length
# ============================================================
def analyze_sequence_length(df: pd.DataFrame, out_dir: Path) -> dict:
    lengths = df["num_frames"].values
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Histogram
    axes[0].hist(lengths, bins=50, color="#4C72B0", edgecolor="black", alpha=0.8)
    axes[0].axvline(np.mean(lengths), color="red", linestyle="--",
                    label=f"mean={np.mean(lengths):.0f}")
    axes[0].axvline(np.median(lengths), color="orange", linestyle="--",
                    label=f"median={np.median(lengths):.0f}")
    axes[0].set_xlabel("Frames per clip")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Sequence Length Distribution")
    axes[0].legend()

    # By class (boxplot)
    classes = sorted(df["class_id"].unique())
    data = [df[df["class_id"] == c]["num_frames"].values for c in classes]
    tick_labels = [str(c) for c in classes]

    # ★ Fix matplotlib >= 3.9 (labels → tick_labels)
    try:
        axes[1].boxplot(data, tick_labels=tick_labels, showfliers=False)
    except TypeError:
        # Fallback cho matplotlib < 3.9
        axes[1].boxplot(data, labels=tick_labels, showfliers=False)

    axes[1].set_xlabel("Class ID")
    axes[1].set_ylabel("Frames")
    axes[1].set_title("Sequence Length per Class")
    plt.setp(axes[1].get_xticklabels(), fontsize=7)

    save_fig(fig, out_dir / "sequence_length.png")

    return {
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "std": float(lengths.std()),
        "p25": float(np.percentile(lengths, 25)),
        "p75": float(np.percentile(lengths, 75)),
    }


# ============================================================
# 3. Joint coordinate stats
# ============================================================
def analyze_joint_stats(df: pd.DataFrame, out_dir: Path,
                         n_samples: int = 300) -> dict:
    """Load n_samples NPY, tính mean/std cho từng joint."""
    sample = df.sample(n=min(n_samples, len(df)), random_state=42)

    all_kps = []
    for p in sample["skeleton_path"]:
        try:
            kps = np.load(p)  # (T, 17, 3)
            if kps.shape[1] == 17:
                all_kps.append(kps)
        except Exception:
            continue

    if not all_kps:
        return {"error": "Không load được skeleton nào"}

    # Concat theo time
    concat = np.concatenate(all_kps, axis=0)  # (N_total, 17, 3)

    # Stats per joint per axis
    mean_j = concat.mean(axis=0)  # (17, 3)
    std_j = concat.std(axis=0)    # (17, 3)

    # Heatmap mean
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, data, title in zip(axes, [mean_j, std_j],
                                ["Mean per joint/axis", "Std per joint/axis"]):
        im = ax.imshow(data, aspect="auto", cmap="viridis")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["x", "y", "conf"])
        ax.set_yticks(range(17))
        ax.set_yticklabels(JOINT_NAMES, fontsize=8)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    save_fig(fig, out_dir / "joint_stats_heatmap.png")

    return {
        "n_samples_analyzed": len(all_kps),
        "n_frames_total": int(concat.shape[0]),
        "coord_range": {
            "min": float(concat[:, :, :2].min()),
            "max": float(concat[:, :, :2].max()),
        },
        "conf_range": {
            "min": float(concat[:, :, 2].min()),
            "max": float(concat[:, :, 2].max()),
            "mean": float(concat[:, :, 2].mean()),
        },
        "per_joint_mean": {JOINT_NAMES[i]: mean_j[i].tolist()
                            for i in range(17)},
        "per_joint_std": {JOINT_NAMES[i]: std_j[i].tolist()
                           for i in range(17)},
    }


# ============================================================
# 4. Sample skeleton visualization
# ============================================================
def _draw_skeleton(ax, kps_2d, color="blue", alpha=1.0):
    """kps_2d: (17, 2)."""
    # Edges
    for i, j in COCO_EDGES:
        if i < len(kps_2d) and j < len(kps_2d):
            ax.plot([kps_2d[i, 0], kps_2d[j, 0]],
                    [kps_2d[i, 1], kps_2d[j, 1]],
                    color=color, alpha=alpha, linewidth=2)
    # Joints
    ax.scatter(kps_2d[:, 0], kps_2d[:, 1],
               c=color, s=30, alpha=alpha, zorder=5)


def plot_sample_skeletons(df: pd.DataFrame, class_names: list, out_dir: Path,
                           n_classes: int = 8, n_frames: int = 5):
    """Vẽ 1 skeleton sequence cho n_classes class, mỗi class n_frames."""
    # Chọn 1 sample cho mỗi class
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
    fig, axes = plt.subplots(n, n_frames, figsize=(3 * n_frames, 3 * n))

    if n == 1:
        axes = axes[np.newaxis, :]

    for row, (cid, row_df) in enumerate(selected):
        try:
            kps = np.load(row_df["skeleton_path"])
        except Exception:
            continue

        T = kps.shape[0]
        frame_idx = np.linspace(0, T - 1, n_frames).astype(int)

        for col, fi in enumerate(frame_idx):
            ax = axes[row, col]
            kps_2d = kps[fi, :, :2]  # (17, 2)
            _draw_skeleton(ax, kps_2d)
            ax.set_xlim(-3, 3)
            ax.set_ylim(3, -3)  # invert y (image coords)
            ax.set_aspect("equal")
            ax.axis("off")
            if col == 0:
                ax.set_title(f"C{cid}: {class_names[cid]}",
                             fontsize=10, loc="left")
            if col == n_frames // 2:
                ax.set_title(f"frame {fi}", fontsize=8)

    save_fig(fig, out_dir / "sample_skeletons.png")


# ============================================================
# 5. Motion energy per class
# ============================================================
def analyze_motion_per_class(df: pd.DataFrame, class_names: list,
                              out_dir: Path,
                              n_samples_per_class: int = 30) -> dict:
    """Motion energy = mean abs diff giữa 2 frames liên tiếp."""
    class_energy = defaultdict(list)

    for cid in sorted(df["class_id"].unique()):
        sub = df[df["class_id"] == cid].sample(
            n=min(n_samples_per_class, (df["class_id"] == cid).sum()),
            random_state=42)
        for p in sub["skeleton_path"]:
            try:
                kps = np.load(p)
                if kps.shape[0] < 2:
                    continue
                diff = np.diff(kps[:, :, :2], axis=0)
                energy = np.linalg.norm(diff, axis=-1).mean()
                class_energy[cid].append(energy)
            except Exception:
                continue

    if not class_energy:
        return {"error": "No data"}

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    cids = sorted(class_energy.keys())
    means = [np.mean(class_energy[c]) for c in cids]
    stds = [np.std(class_energy[c]) for c in cids]
    ax.bar(cids, means, yerr=stds, capsize=4,
           color=plt.cm.tab20(np.linspace(0, 1, len(cids))))

    ax.set_xticks(cids)
    ax.set_xticklabels([class_names[c] for c in cids],
                        rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Motion energy (mean |Δkps|)")
    ax.set_title("Motion Energy per Class")
    save_fig(fig, out_dir / "motion_energy_per_class.png")

    return {class_names[c]: {"mean": float(np.mean(class_energy[c])),
                              "std": float(np.std(class_energy[c])),
                              "n": len(class_energy[c])}
             for c in cids}


# ============================================================
# 6. Data quality check
# ============================================================
def quality_check(df: pd.DataFrame, n_samples: int = 200) -> dict:
    """Kiểm tra NaN, shape mismatch."""
    sample = df.sample(n=min(n_samples, len(df)), random_state=42)

    issues = {"missing_file": 0, "bad_shape": 0, "nan_found": 0,
              "all_zero": 0, "ok": 0}

    for p in sample["skeleton_path"]:
        try:
            kps = np.load(p)
        except Exception:
            issues["missing_file"] += 1
            continue

        if kps.ndim != 3 or kps.shape[1] != 17:
            issues["bad_shape"] += 1
            continue
        if np.isnan(kps).any():
            issues["nan_found"] += 1
            continue
        if np.abs(kps[:, :, :2]).sum() < 1e-6:
            issues["all_zero"] += 1
            continue

        issues["ok"] += 1

    issues["n_checked"] = len(sample)
    issues["ok_pct"] = 100.0 * issues["ok"] / max(len(sample), 1)
    return issues


# ============================================================
# Orchestrator
# ============================================================
def run_skeleton_eda(manifest_csv: Path, class_names: list,
                      out_dir: Path, n_samples_stats: int = 300,
                      log_fn=None) -> dict:
    def log(m):
        (log_fn or print)(m)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_csv)
    df = df[df["skeleton_path"].notna() & (df["skeleton_path"] != "")].copy()
    df["class_id"] = df["class_id"].astype(int)

    log(f"  Loaded {len(df)} skeleton clips")

    stats = {"n_clips": len(df)}

    log("  [1/6] Class distribution...")
    stats["class_distribution"] = analyze_class_distribution(df, class_names, out_dir)

    log("  [2/6] Sequence length...")
    stats["sequence_length"] = analyze_sequence_length(df, out_dir)

    log(f"  [3/6] Joint stats (sampling {n_samples_stats} clips)...")
    stats["joint_stats"] = analyze_joint_stats(df, out_dir, n_samples_stats)

    log("  [4/6] Sample skeletons visualization...")
    plot_sample_skeletons(df, class_names, out_dir, n_classes=8, n_frames=5)

    log("  [5/6] Motion energy per class...")
    stats["motion_energy"] = analyze_motion_per_class(df, class_names, out_dir)

    log("  [6/6] Quality check...")
    stats["quality"] = quality_check(df, n_samples=200)

    save_stats(stats, out_dir / "stats.json")
    log(f"  ✓ Saved → {out_dir}")
    return stats
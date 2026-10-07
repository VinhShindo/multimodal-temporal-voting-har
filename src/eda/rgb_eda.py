"""RGB EDA — MMAct trimmed video."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eda.common import plot_class_distribution, save_fig, save_stats

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ============================================================
# 1. Class distribution
# ============================================================
def analyze_class_distribution(df: pd.DataFrame, class_names: list,
                                out_dir: Path) -> dict:
    return plot_class_distribution(
        df, class_names,
        out_dir / "class_distribution.png",
        title="RGB — Class Distribution",
    )


# ============================================================
# 2. Video metadata
# ============================================================
def analyze_video_metadata(df: pd.DataFrame, out_dir: Path,
                            n_samples: int = 200) -> dict:
    if not HAS_CV2:
        return {"error": "cv2 not installed"}

    sample = df.sample(n=min(n_samples, len(df)), random_state=42)

    durations, fps_list, width_list, height_list, frame_counts = [], [], [], [], []
    missing = 0

    for p in sample["rgb_path"]:
        try:
            cap = cv2.VideoCapture(p)
            if not cap.isOpened():
                missing += 1
                continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if fps > 0 and n > 0:
                durations.append(n / fps)
                fps_list.append(fps)
                width_list.append(w)
                height_list.append(h)
                frame_counts.append(n)
        except Exception:
            missing += 1
            continue

    if not durations:
        return {"error": "No valid video", "missing": missing}

    durations = np.array(durations)
    fps_arr = np.array(fps_list)
    frame_arr = np.array(frame_counts)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Duration
    axes[0].hist(durations, bins=40, color="#4C72B0", alpha=0.8)
    axes[0].axvline(durations.mean(), color="red", linestyle="--",
                    label=f"mean={durations.mean():.2f}s")
    axes[0].set_xlabel("Duration (sec)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Video Duration")
    axes[0].legend()

    # FPS
    axes[1].hist(fps_arr, bins=20, color="#55A868", alpha=0.8)
    axes[1].axvline(fps_arr.mean(), color="red", linestyle="--",
                    label=f"mean={fps_arr.mean():.1f}")
    axes[1].set_xlabel("FPS")
    axes[1].set_title("Frame Rate")
    axes[1].legend()

    # Frame count
    axes[2].hist(frame_arr, bins=40, color="#C44E52", alpha=0.8)
    axes[2].axvline(frame_arr.mean(), color="red", linestyle="--",
                    label=f"mean={frame_arr.mean():.0f}")
    axes[2].set_xlabel("Frames")
    axes[2].set_title("Frame Count")
    axes[2].legend()

    save_fig(fig, out_dir / "video_metadata.png")

    return {
        "n_sampled": len(sample),
        "n_valid": len(durations),
        "n_missing": missing,
        "duration": {"min": float(durations.min()), "max": float(durations.max()),
                     "mean": float(durations.mean()), "median": float(np.median(durations))},
        "fps":      {"min": float(fps_arr.min()), "max": float(fps_arr.max()),
                     "mean": float(fps_arr.mean())},
        "resolution": {"widths": list(set(width_list))[:10],
                       "heights": list(set(height_list))[:10]},
        "frames":   {"min": int(frame_arr.min()), "max": int(frame_arr.max()),
                     "mean": float(frame_arr.mean())},
    }


# ============================================================
# 3. Sample frames per class
# ============================================================
def plot_sample_frames(df: pd.DataFrame, class_names: list, out_dir: Path,
                        n_classes: int = 8, n_frames: int = 5):
    if not HAS_CV2:
        return

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
    fig, axes = plt.subplots(n, n_frames, figsize=(3 * n_frames, 2.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, (cid, row_df) in enumerate(selected):
        try:
            cap = cv2.VideoCapture(row_df["rgb_path"])
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                continue
            idx = np.linspace(0, total - 1, n_frames).astype(int)

            for col, fi in enumerate(idx):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ret, frame = cap.read()
                ax = axes[row, col]
                ax.axis("off")
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ax.imshow(frame)
                if col == 0:
                    ax.set_title(f"C{cid}: {class_names[cid]}",
                                 fontsize=9, loc="left")
            cap.release()
        except Exception:
            continue

    save_fig(fig, out_dir / "sample_frames.png")


# ============================================================
# 4. Quality check
# ============================================================
def quality_check(df: pd.DataFrame, n_samples: int = 100) -> dict:
    if not HAS_CV2:
        return {"error": "cv2 not installed"}

    sample = df.sample(n=min(n_samples, len(df)), random_state=42)
    issues = {"missing_file": 0, "cant_open": 0, "empty": 0, "ok": 0}

    for p in sample["rgb_path"]:
        try:
            cap = cv2.VideoCapture(p)
            if not cap.isOpened():
                issues["cant_open"] += 1
                cap.release()
                continue
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if n <= 0:
                issues["empty"] += 1
            else:
                issues["ok"] += 1
        except Exception:
            issues["missing_file"] += 1

    issues["n_checked"] = len(sample)
    issues["ok_pct"] = 100.0 * issues["ok"] / max(len(sample), 1)
    return issues


# ============================================================
# Orchestrator
# ============================================================
def run_rgb_eda(manifest_csv: Path, class_names: list,
                out_dir: Path, log_fn=None) -> dict:
    def log(m):
        (log_fn or print)(m)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_csv)
    df = df[df["rgb_path"].notna() & (df["rgb_path"] != "")].copy()
    df["class_id"] = df["class_id"].astype(int)

    log(f"  Loaded {len(df)} RGB clips")

    stats = {"n_clips": len(df)}

    log("  [1/4] Class distribution...")
    stats["class_distribution"] = analyze_class_distribution(df, class_names, out_dir)

    log("  [2/4] Video metadata (sampling 200)...")
    stats["video_metadata"] = analyze_video_metadata(df, out_dir, n_samples=200)

    log("  [3/4] Sample frames...")
    plot_sample_frames(df, class_names, out_dir, n_classes=8, n_frames=5)

    log("  [4/4] Quality check...")
    stats["quality"] = quality_check(df, n_samples=100)

    save_stats(stats, out_dir / "stats.json")
    log(f"  ✓ Saved → {out_dir}")
    return stats
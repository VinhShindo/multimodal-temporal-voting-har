"""Chạy EDA cho cả 3 modality + combined summary."""
from pathlib import Path

from src.eda.common import ensure_dir, save_stats
from src.eda.skeleton_eda import run_skeleton_eda
from src.eda.imu_eda import run_imu_eda
from src.eda.rgb_eda import run_rgb_eda


def run_all_eda(manifest_dir: Path, class_names: list,
                output_root: Path, log_fn=None) -> dict:
    def log(m):
        (log_fn or print)(m)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results = {}

    # ─── Skeleton ───
    log("")
    log("=" * 70)
    log("  SKELETON EDA")
    log("=" * 70)
    try:
        results["skeleton"] = run_skeleton_eda(
            Path(manifest_dir) / "skeleton" / "train.csv",
            class_names,
            output_root / "skeleton",
            log_fn=log)
    except Exception as e:
        log(f"  ✗ Skeleton EDA failed: {e}")
        results["skeleton"] = {"error": str(e)}

    # ─── IMU ───
    log("")
    log("=" * 70)
    log("  IMU EDA")
    log("=" * 70)
    try:
        results["imu"] = run_imu_eda(
            Path(manifest_dir) / "imu" / "train.csv",
            class_names,
            output_root / "imu",
            log_fn=log)
    except Exception as e:
        log(f"  ✗ IMU EDA failed: {e}")
        results["imu"] = {"error": str(e)}

    # ─── RGB ───
    log("")
    log("=" * 70)
    log("  RGB EDA")
    log("=" * 70)
    try:
        results["rgb"] = run_rgb_eda(
            Path(manifest_dir) / "rgb" / "train.csv",
            class_names,
            output_root / "rgb",
            log_fn=log)
    except Exception as e:
        log(f"  ✗ RGB EDA failed: {e}")
        results["rgb"] = {"error": str(e)}

    # ─── Combined ───
    log("")
    log("=" * 70)
    log("  COMBINED SUMMARY")
    log("=" * 70)
    _combined_summary(results, class_names, output_root / "combined", log)

    # Save toàn bộ stats
    save_stats(results, output_root / "all_stats.json")

    return results


def _combined_summary(results: dict, class_names: list, out_dir: Path, log):
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bảng so sánh
    lines = []
    lines.append("=" * 70)
    lines.append("  MMAct — EDA COMBINED SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    for mod in ["skeleton", "imu", "rgb"]:
        r = results.get(mod, {})
        if "error" in r:
            lines.append(f"  {mod:10s}  ERROR: {r['error']}")
            continue
        n = r.get("n_clips", 0)
        cd = r.get("class_distribution", {})
        q = r.get("quality", {})
        lines.append(f"  {mod:10s}  clips={n:>6}  "
                     f"classes={cd.get('num_classes', 0):>2}  "
                     f"min/class={cd.get('min_per_class', 0):>4}  "
                     f"quality_ok={q.get('ok_pct', 0):.1f}%")
    lines.append("")

    # Class balance comparison — vẽ chung
    try:
        fig, ax = plt.subplots(figsize=(16, 6))
        x = np.arange(len(class_names))
        width = 0.27
        colors = {"skeleton": "#4C72B0", "imu": "#55A868", "rgb": "#C44E52"}

        for i, mod in enumerate(["skeleton", "imu", "rgb"]):
            r = results.get(mod, {})
            cd = r.get("class_distribution", {}).get("counts", {})
            counts = [cd.get(c, 0) for c in class_names]
            ax.bar(x + (i - 1) * width, counts, width,
                   label=mod, color=colors[mod], alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("Number of clips")
        ax.set_title("Class Distribution — 3 Modalities")
        ax.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "class_balance_3mod.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        lines.append("  Saved class_balance_3mod.png")
    except Exception as e:
        lines.append(f"  Không vẽ được combined chart: {e}")

    # Write summary
    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    # Log
    for ln in lines:
        log(ln)
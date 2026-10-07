"""Shared utilities cho EDA."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# ============================================================
# Setup
# ============================================================
def setup_plot_style():
    """Style đẹp cho báo cáo."""
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.autolayout": True,
    })


def human_size(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_class_names(cfg_path: str = "configs/classes_mmact.yaml") -> list:
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    id2name = {int(k): v for k, v in cfg["classes"].items()}
    n = int(cfg["num_classes"])
    return [id2name[i] for i in range(n)]


def save_stats(stats: dict, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def save_fig(fig, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Class balance
# ============================================================
def plot_class_distribution(df: pd.DataFrame, class_names: list,
                             out_path: Path, title: str = "Class Distribution",
                             log_scale: bool = False):
    counts = df["class_id"].value_counts().sort_index()
    # Fill missing
    all_ids = list(range(len(class_names)))
    counts = counts.reindex(all_ids, fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_ids)))
    bars = ax.bar(range(len(all_ids)), counts.values, color=colors)

    ax.set_xticks(range(len(all_ids)))
    ax.set_xticklabels(class_names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")

    # Annotate
    for bar, v in zip(bars, counts.values):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, v, str(int(v)),
                    ha="center", va="bottom", fontsize=7)

    ax.axhline(counts.mean(), color="red", linestyle="--",
               alpha=0.6, label=f"mean={counts.mean():.0f}")
    ax.legend()
    save_fig(fig, out_path)

    stats = {
        "total": int(counts.sum()),
        "num_classes": int((counts > 0).sum()),
        "min_per_class": int(counts[counts > 0].min()) if (counts > 0).any() else 0,
        "max_per_class": int(counts.max()),
        "mean_per_class": float(counts[counts > 0].mean()) if (counts > 0).any() else 0,
        "std_per_class": float(counts[counts > 0].std()) if (counts > 0).any() else 0,
        "counts": {class_names[i]: int(counts.iloc[i]) for i in range(len(all_ids))},
    }
    return stats
"""Base processor + shared utilities."""
import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Base class
# ============================================================
class BaseProcessor(ABC):
    """Abstract processor cho 1 modality."""

    name: str = "base"

    @abstractmethod
    def process_one(self, raw_path: Path, out_path: Path) -> dict:
        """Process 1 clip: raw → processed. Return dict metadata."""
        raise NotImplementedError

    def build_output_path(self, row: pd.Series, out_dir: Path,
                          split: str) -> Path:
        """Sinh output path từ manifest row."""
        clip_id = row["clip_id"]
        return out_dir / split / f"{clip_id}.npy"


# ============================================================
# IO helpers
# ============================================================
def safe_load_npy(path: Path) -> np.ndarray | None:
    try:
        return np.load(path)
    except Exception:
        return None


def save_npy(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.float32))


# ============================================================
# Stats aggregation
# ============================================================
class StatsCollector:
    """Thu thập thống kê trong quá trình process."""

    def __init__(self):
        self.ok = 0
        self.skip_missing = 0
        self.skip_bad = 0
        self.errors = []
        self.shapes = []
        self.issues = []

    def record_ok(self, shape: tuple):
        self.ok += 1
        self.shapes.append(shape)

    def record_skip(self, kind: str, msg: str):
        if kind == "missing":
            self.skip_missing += 1
        elif kind == "bad":
            self.skip_bad += 1
        self.issues.append({"kind": kind, "msg": msg})

    def record_error(self, msg: str):
        self.errors.append(msg)

    def summary(self) -> dict:
        out = {
            "ok": self.ok,
            "skip_missing": self.skip_missing,
            "skip_bad": self.skip_bad,
            "n_errors": len(self.errors),
        }
        shape_rows = [shape for shape in self.shapes if len(shape) > 1]
        if shape_rows:
            arr = np.asarray(shape_rows)
            out["shape_min"] = arr.min(axis=0).tolist()
            out["shape_max"] = arr.max(axis=0).tolist()
            out["shape_mean"] = arr.mean(axis=0).round(1).tolist()
        if self.errors:
            out["first_errors"] = self.errors[:5]
        return out


# ============================================================
# Manifest helpers
# ============================================================
def load_manifest(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["class_id"] = df["class_id"].astype(int)
    return df


def save_manifest(df: pd.DataFrame, csv_path: Path):
    df.to_csv(csv_path, index=False, encoding="utf-8")
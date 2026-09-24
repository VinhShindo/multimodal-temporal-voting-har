"""Đọc/ghi JSON, NPY, CSV chuẩn hóa."""
import json
from pathlib import Path
import numpy as np


def save_json(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_npy(arr: np.ndarray, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def load_npy(path: str | Path) -> np.ndarray:
    return np.load(path)

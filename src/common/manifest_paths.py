"""Resolve MMAct manifests from data configuration files."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_data_config(config_path: str | Path) -> dict:
    with Path(config_path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def resolve_manifest_path(
    config_path: str | Path,
    split: str,
    manifest_dir_override: str | Path | None = None,
) -> Path | None:
    """Resolve a configured train/val/test manifest.

    ``manifest_dir_override`` takes precedence for simple filenames such as
    ``train_cross_view.csv``.  Explicit paths such as
    ``data/.../test_cross_view.csv`` remain explicit.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported manifest split: {split}")
    cfg = load_data_config(config_path)
    paths = cfg.get("paths", {}) or {}
    configured_dir = paths.get("manifest_dir")
    base = Path(manifest_dir_override or configured_dir or ".")
    value = paths.get(f"{split}_manifest")
    if value is None or str(value).strip() == "":
        return None
    candidate = Path(str(value))
    if candidate.is_absolute() or candidate.is_file():
        return candidate
    return base / candidate


def resolve_train_val_manifests(
    config_path: str | Path,
    manifest_dir_override: str | Path | None = None,
) -> tuple[Path, Path]:
    train = resolve_manifest_path(config_path, "train", manifest_dir_override)
    val = resolve_manifest_path(config_path, "val", manifest_dir_override)
    if train is None or val is None:
        raise ValueError(
            f"{config_path} phải cấu hình paths.train_manifest và "
            "paths.val_manifest"
        )
    return train, val

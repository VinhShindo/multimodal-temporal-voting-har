"""Resolve modality-specific MMAct train/val manifests."""
from __future__ import annotations

from pathlib import Path

import yaml


MM_ACT_PROTOCOLS = (
    "x_scene", "x_view", "x_sub_scene", "x_sub_view",
    "x_session_scene", "x_session_view",
)


def load_data_config(config_path: str | Path) -> dict:
    with Path(config_path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def resolve_manifest_path(
    config_path: str | Path,
    split: str,
    manifest_dir_override: str | Path | None = None,
    protocol: str | None = None,
) -> Path | None:
    """Resolve train/val under ``<root>/<protocol>/<modality>/``."""
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported manifest split: {split}")
    cfg = load_data_config(config_path)
    paths = cfg.get("paths", {}) or {}
    selected_protocol = protocol or paths.get("protocol", "x_view")
    if selected_protocol not in MM_ACT_PROTOCOLS:
        raise ValueError(f"Unsupported MMAct protocol: {selected_protocol}")
    modality = cfg.get("modality")
    if not modality:
        raise ValueError(f"{config_path} must define a modality")
    if manifest_dir_override is not None:
        base = Path(manifest_dir_override)
    else:
        configured_dir = paths.get("manifest_dir")
        base = Path(configured_dir or ".") / selected_protocol / modality
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
    protocol: str | None = None,
) -> tuple[Path, Path]:
    train = resolve_manifest_path(
        config_path, "train", manifest_dir_override, protocol
    )
    val = resolve_manifest_path(
        config_path, "val", manifest_dir_override, protocol
    )
    if train is None or val is None:
        raise ValueError(
            f"{config_path} phải cấu hình paths.train_manifest và "
            "paths.val_manifest"
        )
    return train, val

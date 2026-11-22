"""Attach RGB video paths to the source scene/view manifests."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


IDENTITY_COLUMNS = ("subject", "cam", "scene", "session", "action")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir", default="data/processed/MMAct/source_manifests",
        help="Directory containing train/val_cross_{scene,view}.csv.",
    )
    parser.add_argument(
        "--raw-root", default="data/raw",
        help="Raw data root; nested data/raw/trimmed is detected automatically.",
    )
    parser.add_argument("--video-root", default=None,
                        help="Override the root containing extracted videos.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Resolve and report paths without modifying manifests.")
    return parser.parse_args()


def _raw_root(path: Path) -> Path:
    nested = path / "trimmed"
    return nested if (nested / "trimmed_splits").is_dir() else path


def _video_index(video_root: Path) -> dict[tuple[str, ...], str]:
    if not video_root.is_dir():
        print(f"[WARN] RGB video directory is not available yet: {video_root}")
        return {}
    index: dict[tuple[str, ...], str] = {}
    for path in sorted(
        candidate for candidate in video_root.rglob("*")
        if candidate.is_file() and candidate.suffix.casefold() == ".mp4"
    ):
        parts = {part.casefold() for part in path.parts}
        key_prefix = ("subject", "cam", "scene", "session")
        for part in parts:
            if part.startswith("subject") and part[7:].isdigit():
                subject = part
                break
        else:
            continue
        values = [subject]
        for prefix in key_prefix[1:]:
            match = next(
                (part for part in parts
                 if part.startswith(prefix) and part[len(prefix):].isdigit()),
                None,
            )
            if match is None:
                break
            values.append(match)
        else:
            key = (*values, path.stem.casefold())
            index.setdefault(key, str(path))
    if not index:
        print(f"[WARN] No indexable RGB videos found under {video_root}")
    return index


def _resolve(row: pd.Series, index: dict[tuple[str, ...], str]) -> str:
    key = tuple(str(row[column]).casefold() for column in IDENTITY_COLUMNS)
    return index.get(key, "")


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    raw_root = _raw_root(Path(args.raw_root))
    root_override = Path(args.video_root) if args.video_root else None
    for protocol in ("cross_scene", "cross_view"):
        video_root = root_override or (raw_root / f"{protocol}_video")
        index = _video_index(video_root)
        for subset in ("train", "val"):
            manifest_path = source_dir / f"{subset}_{protocol}.csv"
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            frame = pd.read_csv(manifest_path)
            missing = sorted(set(IDENTITY_COLUMNS) - set(frame.columns))
            if missing:
                raise ValueError(f"{manifest_path} is missing columns: {missing}")
            paths = [
                _resolve(row, index)
                for _, row in tqdm(
                    frame.iterrows(), total=len(frame),
                    desc=f"{protocol}/{subset}",
                )
            ]
            found = sum(bool(path) and Path(path).is_file() for path in paths)
            frame["rgb_path"] = paths
            print(f"{protocol}/{subset}: RGB {found}/{len(frame)}")
            if not args.verify_only:
                frame.to_csv(manifest_path, index=False)


if __name__ == "__main__":
    main()

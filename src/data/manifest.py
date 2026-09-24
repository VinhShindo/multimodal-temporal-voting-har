"""Build manifest tổng hợp từ tất cả session."""
from pathlib import Path
import pandas as pd

from src.data.annotation import load_annotation
from src.data.segmenter import build_isolated_manifest
from src.common.classes import load_classes


def build_full_manifest(annotation_dir: str | Path,
                        classes_path: str | Path = "configs/classes.yaml") -> pd.DataFrame:
    cmap = load_classes(classes_path)
    frames = []
    for csv_path in sorted(Path(annotation_dir).glob("*.csv")):
        subject = csv_path.stem.split("_")[0]
        session = csv_path.stem
        ann = load_annotation(csv_path)
        df = build_isolated_manifest(ann, subject, session)
        df["class_id"] = df["label"].map(cmap.name2id)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

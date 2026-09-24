"""Parse annotation gốc (CSV) → segment list."""
import pandas as pd
from pathlib import Path


def load_annotation(csv_path: str | Path) -> pd.DataFrame:
    """CSV cần có cột: start_time, end_time, label."""
    df = pd.read_csv(csv_path)
    required = {"start_time", "end_time", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Annotation thiếu cột: {required - set(df.columns)}")
    return df.sort_values("start_time").reset_index(drop=True)

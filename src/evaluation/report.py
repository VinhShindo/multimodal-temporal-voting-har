"""Xuất bảng kết quả CSV/LaTeX."""
import pandas as pd
from pathlib import Path


def save_metrics_csv(rows: list, path: str | Path):
    df = pd.DataFrame(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def to_latex(df: pd.DataFrame) -> str:
    return df.to_latex(index=False)

"""Cắt isolated sample từ continuous session theo annotation."""
import pandas as pd


def build_isolated_manifest(annotation: pd.DataFrame, subject: str, session: str) -> pd.DataFrame:
    rows = []
    for _, seg in annotation.iterrows():
        rows.append({
            "subject": subject,
            "session": session,
            "start": float(seg["start_time"]),
            "end": float(seg["end_time"]),
            "label": seg["label"],
        })
    return pd.DataFrame(rows)

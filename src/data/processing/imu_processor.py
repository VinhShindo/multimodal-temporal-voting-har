"""IMU processor — sensor CSV → resampled NPY @ 50 Hz.

Pipeline:
  1. Parse CSV (timestamp_str, ch1, ch2, ch3)
  2. Convert timestamp → epoch ms
  3. Remove invalid rows
  4. Linear-interpolation resample to target_hz
  5. Save float32 NPY (T, C)
"""
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.processing.base import (
    BaseProcessor, StatsCollector, save_npy, load_manifest, save_manifest,
)


class IMUProcessor(BaseProcessor):
    name = "imu"

    def __init__(self, target_hz: float = 50.0,
                 min_samples: int = 10,
                 sensor: str = "acc_watch_clip"):
        self.target_hz = target_hz
        self.min_samples = min_samples
        self.sensor = sensor

    # ---------- Core ----------
    def load_csv(self, path: Path):
        """Parse CSV → (ts_ms int64, sig float32 (T,C))."""
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 4:
                    continue
                try:
                    dt = datetime.strptime(parts[0], "%Y%m%d_%H:%M:%S.%f")
                    vals = [float(x) for x in parts[1:]]
                except (ValueError, IndexError):
                    continue
                rows.append((int(dt.timestamp() * 1000), vals))

        if not rows:
            return None, None

        ts = np.array([r[0] for r in rows], dtype=np.int64)
        sig = np.array([r[1] for r in rows], dtype=np.float32)
        return ts, sig

    def resample(self, ts: np.ndarray, sig: np.ndarray) -> np.ndarray:
        """Linear interp về target_hz."""
        t0, t1 = ts[0], ts[-1]
        dur_s = (t1 - t0) / 1000.0
        if dur_s <= 0:
            return sig

        n = int(round(dur_s * self.target_hz)) + 1
        t_new = t0 + np.arange(n) * (1000.0 / self.target_hz)
        t_new = np.clip(t_new, t0, t1)

        out = np.zeros((n, sig.shape[1]), dtype=np.float32)
        for c in range(sig.shape[1]):
            out[:, c] = np.interp(t_new, ts, sig[:, c])
        return out

    def process_one(self, raw_path: Path, out_path: Path) -> dict:
        ts, sig = self.load_csv(raw_path)
        if ts is None or len(ts) < self.min_samples:
            return {"ok": False, "reason": f"too_short ({0 if ts is None else len(ts)})"}

        sig_rs = self.resample(ts, sig)
        save_npy(sig_rs, out_path)
        return {"ok": True, "shape": sig_rs.shape}

    # ---------- Manifest ----------
    def process_manifest(self, manifest_csv: Path,
                         out_dir: Path,
                         split: str,
                         force: bool = False,
                         log_fn=None) -> tuple[pd.DataFrame, dict]:
        def log(m):
            (log_fn or print)(m)

        df = load_manifest(manifest_csv)
        log(f"  Loaded manifest: {len(df)} rows")

        stats = StatsCollector()
        new_paths = []

        for _, row in tqdm(df.iterrows(),
                            desc=f"  imu/{split}",
                            disable=log_fn is None):
            raw = self._build_raw_path(row)
            out = self.build_output_path(row, out_dir, split)

            if not force and out.exists() and out.stat().st_size > 0:
                new_paths.append(str(out))
                stats.record_ok((0,))
                continue

            if not raw.exists():
                new_paths.append("")
                stats.record_skip("missing", str(raw))
                continue

            try:
                res = self.process_one(raw, out)
                if res.get("ok"):
                    new_paths.append(str(out))
                    stats.record_ok(res["shape"])
                else:
                    new_paths.append("")
                    stats.record_skip("bad", f"{row['clip_id']}: {res['reason']}")
            except Exception as e:
                new_paths.append("")
                stats.record_error(f"{row['clip_id']}: {e}")

        df = df.copy()
        df["imu_path"] = new_paths
        save_manifest(df, manifest_csv)

        summary = stats.summary()
        log(f"  → OK={summary['ok']}  skip_missing={summary['skip_missing']}  "
            f"skip_bad={summary['skip_bad']}")
        return df, summary

    def _build_raw_path(self, row: pd.Series) -> Path:
        """IMU sensor KHÔNG có cam."""
        return Path(
            f"data/raw/trimmed_sensor/sensor/{self.sensor}/"
            f"{row['subject']}/{row['scene']}/{row['session']}/"
            f"{row['action']}.csv"
        )
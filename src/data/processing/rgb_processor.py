"""RGB processor — video verify + optional pre-extract frames.

Mode:
  - verify:  chỉ check video có mở được, đọc metadata
  - extract: sample T frames → resize → save uint8 NPY
"""
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.processing.base import (
    BaseProcessor, StatsCollector, load_manifest, save_manifest,
)


class RGBProcessor(BaseProcessor):
    name = "rgb"

    def __init__(self, mode: str = "verify",
                 n_frames: int = 16,
                 frame_size: int = 112):
        self.mode = mode
        self.n_frames = n_frames
        self.frame_size = frame_size

    # ---------- Core ----------
    def verify(self, path: Path) -> dict:
        import cv2
        try:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return {"ok": False, "reason": "cant_open"}
            fps = cap.get(cv2.CAP_PROP_FPS)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if n <= 0 or fps <= 0:
                return {"ok": False, "reason": "empty_video"}
            return {"ok": True, "fps": fps, "n_frames": n,
                    "w": w, "h": h, "duration": n / fps}
        except Exception as e:
            return {"ok": False, "reason": str(e)}

    def extract_frames(self, path: Path, out_path: Path) -> dict:
        """Sample T frames uniform, resize, save uint8 (T, H, W, 3)."""
        import cv2
        try:
            cap = cv2.VideoCapture(str(path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return {"ok": False, "reason": "empty"}

            idx = np.linspace(0, total - 1, self.n_frames).astype(int)
            frames = []
            for fi in idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ret, f = cap.read()
                if not ret:
                    if frames:
                        frames.append(frames[-1])
                    else:
                        frames.append(np.zeros((self.frame_size,
                                                 self.frame_size, 3), np.uint8))
                    continue
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                f = cv2.resize(f, (self.frame_size, self.frame_size),
                                interpolation=cv2.INTER_LINEAR)
                frames.append(f)
            cap.release()

            arr = np.stack(frames, axis=0).astype(np.uint8)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, arr)
            return {"ok": True, "shape": arr.shape}
        except Exception as e:
            return {"ok": False, "reason": str(e)}

    # ---------- Manifest ----------
    def process_manifest(self, manifest_csv: Path,
                         out_dir: Path,
                         split: str,
                         force: bool = False,
                         log_fn=None) -> tuple[pd.DataFrame, dict]:
        def log(m):
            (log_fn or print)(m)

        df = load_manifest(manifest_csv)
        log(f"  Loaded manifest: {len(df)} rows  mode={self.mode}")

        stats = StatsCollector()
        new_paths = []
        durations = []

        for _, row in tqdm(df.iterrows(),
                            desc=f"  rgb/{split}/{self.mode}",
                            disable=log_fn is None):
            raw = row.get("rgb_path", "")
            if not raw or pd.isna(raw):
                # Fallback build
                raw = self._build_raw_path(row)
            raw = Path(raw)

            if not raw.exists():
                new_paths.append("")
                stats.record_skip("missing", str(raw))
                continue

            try:
                if self.mode == "verify":
                    res = self.verify(raw)
                    if res.get("ok"):
                        new_paths.append(str(raw))
                        stats.record_ok((0,))
                        durations.append(res["duration"])
                    else:
                        new_paths.append("")
                        stats.record_skip("bad", f"{row['clip_id']}: {res['reason']}")

                elif self.mode == "extract":
                    out = out_dir / split / f"{row['clip_id']}.npy"
                    if not force and out.exists() and out.stat().st_size > 0:
                        new_paths.append(str(out))
                        stats.record_ok((0,))
                        continue
                    res = self.extract_frames(raw, out)
                    if res.get("ok"):
                        new_paths.append(str(out))
                        stats.record_ok(res["shape"])
                    else:
                        new_paths.append("")
                        stats.record_skip("bad", f"{row['clip_id']}: {res['reason']}")
                else:
                    raise ValueError(f"Unknown mode: {self.mode}")
            except Exception as e:
                new_paths.append("")
                stats.record_error(f"{row['clip_id']}: {e}")

        df = df.copy()
        df["rgb_path"] = new_paths
        save_manifest(df, manifest_csv)

        summary = stats.summary()
        if durations:
            summary["duration_mean_sec"] = float(np.mean(durations))
            summary["duration_min_sec"] = float(np.min(durations))
            summary["duration_max_sec"] = float(np.max(durations))

        log(f"  → OK={summary['ok']}  skip_missing={summary['skip_missing']}  "
            f"skip_bad={summary['skip_bad']}")
        return df, summary

    @staticmethod
    def _build_raw_path(row: pd.Series) -> str:
        subj = row["subject"]
        return (f"data/raw/cross_view_video/trimmed_x-view_{subj}/"
                f"video/cross_view/trainval/{subj}/"
                f"{row['cam']}/{row['scene']}/{row['session']}/"
                f"{row['action']}.mp4")
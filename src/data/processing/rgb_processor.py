"""RGB processor — video verify + optional pre-extract frames.

Mode:
  - verify:  chỉ check video có mở được, đọc metadata
  - extract: sample T frames → resize → save uint8 NPY
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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
                 frame_size: int = 112,
                 num_workers: int = 16):
        self.mode = mode
        self.n_frames = n_frames
        self.frame_size = frame_size
        self.num_workers = max(1, num_workers)

    # ============================================================
    # ABSTRACT — dispatch theo mode
    # ============================================================
    def process_one(self, raw_path: Path, out_path: Path) -> dict:
        """
        Dispatch theo self.mode.

        - verify:  chỉ check video, không ghi output
        - extract: sample frames → NPY
        """
        if self.mode == "verify":
            res = self.verify(raw_path)
            if res.get("ok"):
                return {"ok": True, "shape": (0,)}
            return {"ok": False, "reason": res.get("reason", "unknown")}

        elif self.mode == "extract":
            return self.extract_frames(raw_path, out_path)

        else:
            return {"ok": False, "reason": f"unknown_mode: {self.mode}"}

    # ============================================================
    # Core methods
    # ============================================================
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
        """Sample T frames, center-crop, resize, and save uint8 NPY."""
        try:
            try:
                from decord import VideoReader, cpu

                reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
                total = len(reader)
                if total <= 0:
                    return {"ok": False, "reason": "empty"}
                idx = np.linspace(0, total - 1, self.n_frames).astype(int)
                frames = reader.get_batch(idx).asnumpy()
            except Exception:
                import cv2

                cap = cv2.VideoCapture(str(path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    cap.release()
                    return {"ok": False, "reason": "empty"}
                idx = np.linspace(0, total - 1, self.n_frames).astype(int)
                frames = []
                for frame_idx in idx:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                    ok, frame = cap.read()
                    if ok:
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    elif frames:
                        frames.append(frames[-1])
                    else:
                        frames.append(np.zeros((self.frame_size,
                                                self.frame_size, 3), np.uint8))
                cap.release()
                frames = np.stack(frames, axis=0)

            arr = np.stack([self._center_crop_resize(frame) for frame in frames])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, arr)
            return {"ok": True, "shape": arr.shape}
        except Exception as e:
            return {"ok": False, "reason": str(e)}

    def _center_crop_resize(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        height, width = frame.shape[:2]
        scale = self.frame_size / 0.875
        if height < width:
            new_height = int(scale)
            new_width = int(width * scale / height)
        else:
            new_width = int(scale)
            new_height = int(height * scale / width)
        frame = cv2.resize(frame, (new_width, new_height),
                            interpolation=cv2.INTER_LINEAR)
        y0 = (new_height - self.frame_size) // 2
        x0 = (new_width - self.frame_size) // 2
        return frame[y0:y0 + self.frame_size,
                     x0:x0 + self.frame_size]

    # ============================================================
    # Manifest
    # ============================================================
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
        durations = []

        if self.mode == "extract":
            new_paths = [""] * len(df)
            jobs = {}
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for index, row in df.iterrows():
                    raw = row.get("rgb_path", "")
                    if not raw or pd.isna(raw):
                        raw = self._build_raw_path(row)
                    raw = Path(raw)
                    out = out_dir / split / f"{row['clip_id']}.npy"

                    if not raw.is_file():
                        stats.record_skip("missing", str(raw))
                    elif not force and out.is_file() and out.stat().st_size > 0:
                        new_paths[index] = str(out)
                        stats.record_ok((0,))
                    else:
                        future = executor.submit(self.process_one, raw, out)
                        jobs[future] = (index, row, out)

                for future in tqdm(as_completed(jobs), total=len(jobs),
                                   desc=f"  rgb/{split}/extract",
                                   disable=log_fn is None):
                    index, row, out = jobs[future]
                    try:
                        result = future.result()
                        if result.get("ok"):
                            new_paths[index] = str(out)
                            stats.record_ok(result["shape"])
                        else:
                            stats.record_skip(
                                "bad", f"{row['clip_id']}: {result['reason']}")
                    except Exception as error:
                        stats.record_error(f"{row['clip_id']}: {error}")
        else:
            new_paths = []
            for _, row in tqdm(df.iterrows(),
                               desc=f"  rgb/{split}/{self.mode}",
                               disable=log_fn is None):
                raw = row.get("rgb_path", "")
                if not raw or pd.isna(raw):
                    raw = self._build_raw_path(row)
                raw = Path(raw)

                if not raw.is_file():
                    new_paths.append("")
                    stats.record_skip("missing", str(raw))
                    continue

                try:
                    result = self.verify(raw)
                    if result.get("ok"):
                        new_paths.append(str(raw))
                        stats.record_ok((0,))
                        durations.append(result["duration"])
                    else:
                        new_paths.append("")
                        stats.record_skip(
                            "bad", f"{row['clip_id']}: {result['reason']}")
                except Exception as error:
                    new_paths.append("")
                    stats.record_error(f"{row['clip_id']}: {error}")

        df = df.copy()
        if self.mode == "extract":
            df["rgb_cache_path"] = new_paths
        else:
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
        path = Path(
            f"data/raw/cross_view_video/trimmed_x-view_{subj}/"
            f"video/cross_view/trainval/{subj}/"
            f"{row['cam']}/{row['scene']}/{row['session']}/"
            f"{row['action']}.mp4")
        if path.is_file():
            return str(path)

        if path.parent.is_dir():
            match = next((candidate for candidate in path.parent.iterdir()
                          if candidate.is_file()
                          and candidate.suffix.lower() == ".mp4"
                          and candidate.stem.casefold() == path.stem.casefold()), None)
            if match is not None:
                return str(match)
        return str(path)
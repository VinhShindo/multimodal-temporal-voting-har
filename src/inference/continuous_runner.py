"""Continuous pipeline — lazy video load + gap-aware postprocess."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.inference.sliding_window import generate_windows
from src.inference.predictor import ContinuousPredictor, fuse_probs
from src.inference.boundary import detect_boundaries, merge_nearby
from src.postprocess.smoothing import smooth_probs
from src.postprocess.pipeline import predictions_to_segments, run_postprocess


# ============================================================
# Lazy video reader
# ============================================================
class LazyVideoReader:
    def __init__(self, video_path: Path):
        self.path = str(video_path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Không mở được video: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read_range(self, start_idx: int, end_idx: int) -> np.ndarray:
        start_idx = max(0, start_idx)
        end_idx = min(self.total, end_idx)
        if end_idx <= start_idx:
            return np.zeros((1, self.h, self.w, 3), dtype=np.uint8)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        frames = []
        for _ in range(end_idx - start_idx):
            ret, f = self.cap.read()
            if not ret:
                break
            frames.append(f)
        if not frames:
            return np.zeros((1, self.h, self.w, 3), dtype=np.uint8)
        return np.stack(frames, axis=0)

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ============================================================
# Main
# ============================================================
def run_continuous(session_dir, predictor, class_names,
                    window_sec=4.0, step_sec=1.0,
                    fps=30.0, imu_hz=50.0,
                    fusion_weights=None, use_boundary=True,
                    smooth_kernel=3, t_avg=2.0, min_duration=1.0,
                    conf_threshold=0.15,
                    log_fn=None):
    def log(m): (log_fn or print)(m)
    session_dir = Path(session_dir)

    skel = np.load(session_dir / "skeleton.npy")
    imu = np.load(session_dir / "imu.npy")
    n_frames_csv = len(pd.read_csv(session_dir / "frames.csv"))
    duration = n_frames_csv / fps
    log(f"  Duration: {duration:.1f}s  (skel={skel.shape}, imu={imu.shape})")

    windows = generate_windows(duration, window_sec, step_sec)
    log(f"  Windows: {len(windows)}")

    # Lazy video
    video_reader = None
    if "rgb" in predictor.models:
        try:
            video_reader = LazyVideoReader(session_dir / "video.mp4")
            log(f"  Video: {video_reader.w}x{video_reader.h} @ "
                f"{video_reader.fps:.1f}fps, {video_reader.total} frames")
        except Exception as e:
            log(f"  Video open failed: {e}")
            video_reader = None

    log("  Predicting...")
    window_records = []

    try:
        for w in tqdm(windows, desc="  pred", disable=log_fn is None):
            s0 = max(0, min(int(w.start * fps), skel.shape[0] - 1))
            s1 = max(s0 + 1, min(int(w.end * fps), skel.shape[0]))
            skel_w = skel[s0:s1]

            i0 = max(0, min(int(w.start * imu_hz), imu.shape[0] - 1))
            i1 = max(i0 + 1, min(int(w.end * imu_hz), imu.shape[0]))
            imu_w = imu[i0:i1]

            rgb_w = None
            if video_reader is not None:
                f0 = max(0, min(int(w.start * fps), video_reader.total - 1))
                f1 = max(f0 + 1, min(int(w.end * fps), video_reader.total))
                try:
                    rgb_w = video_reader.read_range(f0, f1)
                except Exception:
                    rgb_w = None

            p_sk = predictor.predict_skeleton(skel_w) if skel_w.shape[0] >= 5 else None
            p_im = predictor.predict_imu(imu_w) if imu_w.shape[0] >= 5 else None
            p_rgb = predictor.predict_rgb(rgb_w) \
                    if rgb_w is not None and len(rgb_w) >= 2 else None

            fused = fuse_probs(
                {"skeleton": p_sk, "imu": p_im, "rgb": p_rgb},
                fusion_weights)
            if fused is None:
                continue

            window_records.append({
                "window_id": w.window_id,
                "start": w.start,
                "end": w.end,
                "probs": fused,
            })
    finally:
        if video_reader is not None:
            video_reader.close()

    log(f"  Predicted {len(window_records)} windows")
    if not window_records:
        return {"windows": [], "boundaries": [],
                "segments_raw": [], "segments_final": [],
                "duration": duration, "window_probs": None}

    # Smooth probs
    probs_all = np.stack([r["probs"] for r in window_records], 0)
    if smooth_kernel > 1:
        probs_all = smooth_probs(probs_all, kernel=smooth_kernel)

    preds = []
    for i, r in enumerate(window_records):
        cid = int(probs_all[i].argmax())
        preds.append({
            "window_id": r["window_id"],
            "start": r["start"], "end": r["end"],
            "class_id": cid,
            "confidence": float(probs_all[i, cid]),
        })

    segments_raw = predictions_to_segments(preds, class_names)

    # Boundary
    boundaries = []
    if use_boundary:
        log("  Detecting boundaries...")
        boundaries = detect_boundaries(skel, fps=fps, percentile=75.0)
        boundaries = merge_nearby(boundaries, max_gap_sec=0.3)
        log(f"  Boundaries: {len(boundaries)}")

    # ★ Postprocess với gap detection + refinement
    segments_final = run_postprocess(
        preds, class_names,
        duration=duration,
        fps=fps,
        num_classes=len(class_names),
        skel=skel,                    # ★ gap detection
        smooth_kernel=0,
        t_avg=t_avg,
        min_duration=min_duration,
        conf_threshold=conf_threshold,
        use_refinement=True,          # ★ segment-level voting
    )

    return {
        "windows": preds,
        "window_probs": probs_all,
        "boundaries": boundaries,
        "segments_raw": segments_raw,
        "segments_final": segments_final,
        "duration": duration,
    }
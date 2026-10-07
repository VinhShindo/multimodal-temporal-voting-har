"""
Simulate continuous video từ trimmed MMAct clips.

Ghép N clips + gap background → video dài liên tục.
Output: data/processed/MMAct/continuous/{session_id}/
  ├── video.mp4
  ├── frames.csv          (frame_idx, utc_offset_ms, clip_id)
  ├── skeleton.npy        (T, 17, 3) @ 30 Hz
  ├── imu.npy             (T, 3) @ 50 Hz
  ├── annotation.csv      (start_sec, end_sec, class_id, class_name, clip_id)
  └── metadata.json
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
import yaml


def load_class_names(cfg="configs/classes_mmact.yaml"):
    with open(cfg, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    id2name = {int(k): v for k, v in c["classes"].items()}
    return [id2name[i] for i in range(int(c["num_classes"]))]


def get_video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release(); return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (fps, n, w, h) if fps > 0 and n > 0 else None


def read_all_frames(path, w, h):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        if f.shape[1] != w or f.shape[0] != h:
            f = cv2.resize(f, (w, h))
        frames.append(f)
    cap.release()
    return np.stack(frames, 0) if frames else None


def resample_video(frames, T_new):
    """Nearest-neighbor cho video frames — KHÔNG dùng float math."""
    T_old = frames.shape[0]
    if T_old == T_new or T_old <= 1:
        return frames
    idx = np.round(np.linspace(0, T_old - 1, T_new)).astype(np.int64)
    return frames[idx]


def resample_signal(arr, T_new):
    """Linear interp cho skeleton/IMU — mượt, không tốn nhiều RAM."""
    T_old = arr.shape[0]
    if T_old == T_new or T_old <= 1:
        return arr.astype(np.float32)
    idx = np.linspace(0, T_old - 1, T_new)
    lo = np.floor(idx).astype(np.int64)
    hi = np.clip(lo + 1, 0, T_old - 1)
    w = (idx - lo).astype(np.float32).reshape(-1, *([1] * (arr.ndim - 1)))
    out = (arr[lo].astype(np.float32) * (1 - w)
           + arr[hi].astype(np.float32) * w)
    return out.astype(np.float32)


def simulate_one_session(df, class_names, out_dir, session_id,
                          n_clips, gap_sec, fps, imu_hz, seed):
    if seed is not None:
        random.seed(seed)

    # Sample stratified
    by_class = df.groupby("class_id")
    samples = []
    for cid, grp in by_class:
        if len(samples) >= n_clips: break
        samples.append(grp.sample(n=1, random_state=seed))
    if len(samples) < n_clips:
        pool = df.drop(pd.concat(samples).index)
        extra = pool.sample(n=n_clips - len(samples), random_state=seed)
        samples.append(extra)
    selected = pd.concat(samples).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"  Selected {len(selected)} clips")

    info0 = get_video_info(selected.iloc[0]["rgb_path"])
    if info0 is None: raise RuntimeError("Không đọc được video đầu")
    _, _, W, H = info0
    print(f"  Resolution {W}x{H}")

    all_frames, all_skel, all_imu = [], [], []
    annotations, frame_records = [], []
    gap_n = int(round(gap_sec * fps))
    gap_imu_n = int(round(gap_sec * imu_hz))
    t_cursor = 0.0

    for idx, row in selected.iterrows():
        clip_id = row["clip_id"]
        cid = int(row["class_id"])
        vpath = row["rgb_path"]
        spath = row.get("skeleton_path", "")
        ipath = row.get("imu_path", "")

        info = get_video_info(vpath)
        if info is None: continue
        vfps, _, _, _ = info

        frames = read_all_frames(vpath, W, H)
        if frames is None or len(frames) < 2: continue

        duration = len(frames) / vfps
        n_v_target = int(round(duration * fps))
        if n_v_target < 2: continue

        # ★ Video: nearest neighbor
        frames = resample_video(frames, n_v_target)

        # ★ Skeleton: linear interp
        skel = np.load(spath) if (spath and Path(spath).exists()) \
               else np.zeros((n_v_target, 17, 3), dtype=np.float32)
        skel = resample_signal(skel, n_v_target)

        # ★ IMU: linear interp
        n_i_target = int(round(duration * imu_hz))
        imu = np.load(ipath) if (ipath and Path(ipath).exists()) \
              else np.zeros((n_i_target, 3), dtype=np.float32)
        imu = resample_signal(imu, n_i_target)

        start_sec = t_cursor
        end_sec = t_cursor + n_v_target / fps

        for i, f in enumerate(frames):
            all_frames.append(f)
            frame_records.append((len(all_frames) - 1,
                                    int((start_sec + i / fps) * 1000),
                                    clip_id))
        all_skel.append(skel)
        all_imu.append(imu)

        annotations.append({
            "start_sec": start_sec, "end_sec": end_sec,
            "class_id": cid, "class_name": class_names[cid],
            "clip_id": clip_id,
        })
        t_cursor = end_sec

        # Gap
        if idx < len(selected) - 1:
            all_frames.extend([np.zeros((H, W, 3), dtype=np.uint8)] * gap_n)
            for i in range(gap_n):
                frame_records.append((len(all_frames) - 1,
                                        int((t_cursor + i / fps) * 1000), "gap"))
            all_skel.append(np.zeros((gap_n, 17, 3), dtype=np.float32))
            all_imu.append(np.zeros((gap_imu_n, 3), dtype=np.float32))
            t_cursor += gap_sec

    if not all_frames: raise RuntimeError("Không có frame")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Video
    writer = cv2.VideoWriter(str(out_dir / "video.mp4"),
                              cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (W, H))
    for f in all_frames: writer.write(f)
    writer.release()

    skel_out = np.concatenate(all_skel, 0)
    imu_out = np.concatenate(all_imu, 0)
    np.save(out_dir / "skeleton.npy", skel_out.astype(np.float32))
    np.save(out_dir / "imu.npy", imu_out.astype(np.float32))

    with open(out_dir / "frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "utc_offset_ms", "clip_id"])
        w.writerows(frame_records)

    with open(out_dir / "annotation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start_sec", "end_sec", "class_id", "class_name", "clip_id"])
        for a in annotations:
            w.writerow([a["start_sec"], a["end_sec"],
                        a["class_id"], a["class_name"], a["clip_id"]])

    meta = {
        "session_id": session_id, "n_clips": len(annotations),
        "gap_sec": gap_sec, "target_fps": fps, "imu_hz": imu_hz,
        "duration_sec": t_cursor, "n_frames": len(all_frames),
        "resolution": [W, H], "n_skeleton": int(skel_out.shape[0]),
        "n_imu": int(imu_out.shape[0]), "seed": seed,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  ✓ {t_cursor:.1f}s, {len(annotations)} actions, "
          f"skel={skel_out.shape}, imu={imu_out.shape}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default="data/processed/MMAct/manifests/val_cross_view.csv")
    ap.add_argument("--classes-cfg", default="configs/classes_mmact.yaml")
    ap.add_argument("--out-root", default="data/processed/MMAct/continuous")
    ap.add_argument("--n-sessions", type=int, default=20)
    ap.add_argument("--n-clips", type=int, default=25)
    ap.add_argument("--gap-sec", type=float, default=0.5)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--imu-hz", type=float, default=50.0)
    ap.add_argument("--start-id", type=int, default=1)
    args = ap.parse_args()

    class_names = load_class_names(args.classes_cfg)
    df = pd.read_csv(args.manifest)
    for col in ("rgb_path", "skeleton_path", "imu_path"):
        df = df[df[col].notna() & (df[col] != "")].copy()
    df["class_id"] = df["class_id"].astype(int)
    print(f"Loaded {len(df)} clips có đủ 3 modality")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for i in range(args.n_sessions):
        sid = f"sim_{args.start_id + i:03d}"
        out_dir = out_root / sid
        if (out_dir / "video.mp4").exists():
            print(f"\n[{sid}] đã tồn tại — skip")
            continue
        print(f"\n{'='*70}\n  SESSION {sid}  ({i+1}/{args.n_sessions})\n{'='*70}")
        try:
            simulate_one_session(df, class_names, out_dir, sid,
                                  args.n_clips, args.gap_sec,
                                  args.fps, args.imu_hz, seed=1000 + i)
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback; traceback.print_exc()

    print(f"\n✓ Done. Output: {out_root}")


if __name__ == "__main__":
    main()
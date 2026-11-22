"""
Process raw MMAct → processed NPY + manifest.

Usage:
    python scripts/01_process_data.py
    python scripts/01_process_data.py --modality skeleton
    python scripts/01_process_data.py --modality rgb --rgb-mode extract
    python scripts/01_process_data.py --force
    python scripts/01_process_data.py --tag exp01 --imu-hz 50

Output:
    data/processed/MMAct/{skeleton,imu,rgb}/{train,val}/*.npy
    data/processed/MMAct/source_manifests/{train,val}_cross_view.csv (updated)
    outputs/runs/process/{timestamp}/log.txt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.common.run_logger import RunLogger, Timer
from src.data.processing.pipeline import DataProcessingPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir",
                    default="data/processed/MMAct/source_manifests")
    ap.add_argument("--processed-root", default="data/processed/MMAct")
    ap.add_argument("--split", default="cross_view",
                    choices=["cross_view", "cross_scene"])
    ap.add_argument("--modality", default="all",
                    choices=["all", "skeleton", "imu", "rgb"])
    ap.add_argument("--force", action="store_true",
                    help="Re-process kể cả khi file đã tồn tại")
    ap.add_argument("--rgb-mode", default="verify",
                    choices=["verify", "extract"],
                    help="verify = chỉ check, extract = lưu frames NPY")
    ap.add_argument("--rgb-frames", type=int, default=16)
    ap.add_argument("--rgb-size", type=int, default=112)
    ap.add_argument("--rgb-workers", type=int, default=16)
    ap.add_argument("--imu-hz", type=float, default=50.0)
    ap.add_argument("--imu-sensor", default="acc_watch_clip",
                    choices=["acc_watch_clip", "acc_phone_clip",
                             "gyro_clip", "orientation_clip"])
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    # ── Run logger ──
    cfg = {
        "manifest_dir":  args.manifest_dir,
        "processed_root": args.processed_root,
        "split":         args.split,
        "modality":      args.modality,
        "force":         args.force,
        "rgb_mode":      args.rgb_mode,
        "rgb_workers":   args.rgb_workers,
        "imu_hz":        args.imu_hz,
        "imu_sensor":    args.imu_sensor,
    }
    run = RunLogger("process", config=cfg, tag=args.tag)

    run.info("=" * 70)
    run.info(f"  DATA PROCESSING — MMAct")
    run.info("=" * 70)
    run.info(f"  Manifest dir:   {args.manifest_dir}")
    run.info(f"  Processed root: {args.processed_root}")
    run.info(f"  Split:          {args.split}")
    run.info(f"  Modality:       {args.modality}")
    run.info(f"  RGB mode:       {args.rgb_mode}")
    run.info(f"  Force:          {args.force}")
    run.info(f"  IMU Hz:         {args.imu_hz}  ({args.imu_sensor})")
    run.info(f"  Log dir:        {run.path}")
    run.info("")

    # ── Pipeline ──
    with Timer() as t:
        pipeline = DataProcessingPipeline(
            processed_root=args.processed_root,
            split=args.split,
            imu_cfg={"target_hz": args.imu_hz, "sensor": args.imu_sensor},
            rgb_cfg={"mode": args.rgb_mode,
                     "n_frames": args.rgb_frames,
                     "frame_size": args.rgb_size,
                     "num_workers": args.rgb_workers},
            log_fn=run.info,
        )
        results = pipeline.run(
            modality=args.modality,
            force=args.force,
            manifest_dir=args.manifest_dir,
        )

    # ── Summary ──
    run.info("")
    run.info("=" * 70)
    run.info("  SUMMARY")
    run.info("=" * 70)
    for key, val in results.items():
        if isinstance(val, dict):
            ok = val.get("ok", "-")
            skm = val.get("skip_missing", "-")
            skb = val.get("skip_bad", "-")
            run.info(f"  {key:25s}  ok={ok}  missing={skm}  bad={skb}")
    run.info("")
    run.info(f"  Time: {t.elapsed_str}")
    run.info(f"  Log:  {run.path}")


if __name__ == "__main__":
    main()
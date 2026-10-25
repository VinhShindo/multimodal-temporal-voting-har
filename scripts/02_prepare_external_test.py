"""Build a labeled manifest from one extracted MMAct test shard.

MMAct Challenge Task 1 is vision-side at test time: sensor/IMU is used during
training but is not expected as an official test modality.  The release is
split into cross-view/cross-scene archive shards.  This script prepares the
currently extracted video shard and records which additional artifacts are
actually present locally; rerun it for each extracted shard.

Usage:
    python scripts/02_prepare_external_test.py
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--video-root",
        default="data/raw/test_set_cross-view_1/test_release/video/cross-view",
    )
    ap.add_argument(
        "--ground-truth",
        default="data/raw/annotation/trimmed_test_view_gt.json",
    )
    ap.add_argument(
        "--classes-config", default="configs/classes_mmact.yaml"
    )
    ap.add_argument("--pose-root", default="data/raw/trimmed_pose")
    ap.add_argument("--imu-root", default="data/raw/trimmed_sensor")
    ap.add_argument(
        "--output-manifest",
        default="data/processed/MMAct/manifests/test_cross_view.csv",
    )
    ap.add_argument("--output-audit", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    video_root = Path(args.video_root)
    gt_path = Path(args.ground_truth)
    classes_path = Path(args.classes_config)
    output_manifest = Path(args.output_manifest)
    output_audit = Path(args.output_audit or output_manifest.with_name(
        "test_cross_view_audit.json"))

    if not video_root.is_dir():
        raise FileNotFoundError(f"Missing video root: {video_root}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing ground truth: {gt_path}")

    classes_cfg = yaml.safe_load(classes_path.read_text(encoding="utf-8"))
    class_to_id = {
        str(name).strip().casefold(): int(class_id)
        for class_id, name in classes_cfg["classes"].items()
    }
    annotations = json.loads(gt_path.read_text(encoding="utf-8"))
    annotations = annotations.get("annotations", annotations)
    videos = sorted(video_root.glob("*.mp4"))
    if not videos:
        raise RuntimeError(f"No .mp4 files found under {video_root}")

    rows = []
    missing_gt = []
    unknown_labels = []
    empty_files = []
    for video in videos:
        clip_id = video.stem
        if video.stat().st_size <= 0:
            empty_files.append(str(video))
        values = annotations.get(clip_id)
        if not values or not isinstance(values, list):
            missing_gt.append(clip_id)
            continue
        raw_label = str(values[0].get("label", "")).strip()
        class_id = class_to_id.get(raw_label.casefold())
        if class_id is None:
            unknown_labels.append({"clip_id": clip_id, "label": raw_label})
            continue
        rows.append({
            "clip_id": clip_id,
            "split": "cross_view",
            "subset": "test",
            "subject": "external_test",
            "cam": "external_test",
            "scene": "external_test",
            "session": "external_test",
            "action": raw_label,
            "class_id": class_id,
            "num_frames": -1,
            "skeleton_path": "",
            "imu_path": "",
            "rgb_path": video.as_posix(),
            "rgb_cache_path": "",
        })

    if missing_gt or unknown_labels:
        raise RuntimeError(
            f"Test manifest validation failed: missing_gt={len(missing_gt)}, "
            f"unknown_labels={len(unknown_labels)}"
        )
    if len({row["clip_id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate clip_id in external test videos")

    frame = pd.DataFrame(rows)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_manifest, index=False, encoding="utf-8")
    external_ids = set(frame["clip_id"])
    pose_files = list(Path(args.pose_root).rglob("*.json")) \
        if Path(args.pose_root).is_dir() else []
    imu_files = list(Path(args.imu_root).rglob("*.csv")) \
        if Path(args.imu_root).is_dir() else []
    pose_stems = {path.stem for path in pose_files}
    imu_stems = {path.stem for path in imu_files}
    audit = {
        "status": "PASS",
        "source_video_root": str(video_root),
        "ground_truth": str(gt_path),
        "modality": "rgb_only",
        "protocol": "cross_view",
        "num_video_files": len(videos),
        "num_manifest_rows": len(frame),
        "num_classes": int(frame["class_id"].nunique()),
        "missing_ground_truth": missing_gt,
        "unknown_labels": unknown_labels,
        "empty_files": empty_files,
        "class_counts": {
            str(key): int(value)
            for key, value in Counter(frame["action"]).items()
        },
        "skeleton_available": False,
        "imu_available": False,
        "rgb_available": True,
        "raw_inventory": {
            "pose_files_total": len(pose_files),
            "imu_files_total": len(imu_files),
            "pose_files_matching_external_clip_ids": len(external_ids & pose_stems),
            "imu_files_matching_external_clip_ids": len(external_ids & imu_stems),
        },
    }
    output_audit.write_text(json.dumps(audit, indent=2) + "\n",
                            encoding="utf-8")
    print(f"Created {output_manifest}: {len(frame)} rows, "
          f"{frame['class_id'].nunique()} classes")
    print(f"Audit: {output_audit}")
    print("Modalities in this shard: RGB; inspect keypoints separately. IMU is not an official Task 1 test modality.")


if __name__ == "__main__":
    main()

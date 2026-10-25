"""Unified processing pipeline — chạy cả 3 modality."""
from pathlib import Path

import pandas as pd

from src.data.processing.skeleton_processor import SkeletonProcessor
from src.data.processing.imu_processor import IMUProcessor
from src.data.processing.rgb_processor import RGBProcessor
from src.data.processing.verification import verify_processed


class DataProcessingPipeline:

    def __init__(self,
                 processed_root: str = "data/processed/MMAct",
                 split: str = "cross_view",
                 skeleton_cfg: dict = None,
                 imu_cfg: dict = None,
                 rgb_cfg: dict = None,
                 log_fn=None):
        self.processed_root = Path(processed_root)
        self.split = split
        self.log_fn = log_fn or print

        self._skeleton_cfg = skeleton_cfg or {}
        self._imu_cfg = imu_cfg or {}
        self._rgb_cfg = rgb_cfg or {}
        self._skeleton = None
        self._imu = None
        self._rgb = None

    @property
    def skeleton(self):
        if self._skeleton is None:
            self._skeleton = SkeletonProcessor(**self._skeleton_cfg)
        return self._skeleton

    @property
    def imu(self):
        if self._imu is None:
            self._imu = IMUProcessor(**self._imu_cfg)
        return self._imu

    @property
    def rgb(self):
        if self._rgb is None:
            self._rgb = RGBProcessor(**self._rgb_cfg)
        return self._rgb
        
    def log(self, msg=""):
        self.log_fn(msg)

    # ============================================================
    # Run
    # ============================================================
    def run(self, modality: str = "all",
            force: bool = False,
            manifest_dir: str = "data/processed/MMAct/manifests") -> dict:

        manifest_dir = Path(manifest_dir)
        results = {}

        splits = [("train", manifest_dir / f"train_{self.split}.csv"),
                  ("val",   manifest_dir / f"val_{self.split}.csv")]

        for split, csv_path in splits:
            if not csv_path.exists():
                self.log(f"  ✗ Không tìm thấy manifest: {csv_path}")
                continue

            self.log("")
            self.log("=" * 70)
            self.log(f"  PROCESSING {split.upper()} — {csv_path.name}")
            self.log("=" * 70)
            self.log("")

            # ---------- Skeleton ----------
            if modality in ("all", "skeleton"):
                self.log("─" * 70)
                self.log("  [1/3] SKELETON")
                self.log("─" * 70)
                out_dir = self.processed_root / "skeleton" / self.split
                _, sumry = self.skeleton.process_manifest(
                    csv_path, out_dir, split, force=force, log_fn=self.log)
                results[f"{split}_skeleton"] = sumry

            # ---------- IMU ----------
            if modality in ("all", "imu"):
                self.log("")
                self.log("─" * 70)
                self.log("  [2/3] IMU")
                self.log("─" * 70)
                out_dir = self.processed_root / "imu" / self.split
                _, sumry = self.imu.process_manifest(
                    csv_path, out_dir, split, force=force, log_fn=self.log)
                results[f"{split}_imu"] = sumry

            # ---------- RGB ----------
            if modality in ("all", "rgb"):
                self.log("")
                self.log("─" * 70)
                self.log(f"  [3/3] RGB — mode={self.rgb.mode}")
                self.log("─" * 70)
                out_dir = self.processed_root / "rgb" / self.split
                _, sumry = self.rgb.process_manifest(
                    csv_path, out_dir, split, force=force, log_fn=self.log)
                results[f"{split}_rgb"] = sumry

        # ---------- Verification ----------
        self.log("")
        self.log("=" * 70)
        self.log("  VERIFICATION")
        self.log("=" * 70)
        for split, csv_path in splits:
            if not csv_path.exists():
                continue
            self.log("")
            self.log(f"  --- {split} ---")
            vres = verify_processed(csv_path, self.log, modalities=(
                ["skeleton", "imu", "rgb"] if modality == "all"
                else [modality]
            ))
            results[f"{split}_verify"] = vres

        return results
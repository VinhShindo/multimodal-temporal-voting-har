"""
Chạy EDA cho cả 3 modality.
Output: outputs/eda/{skeleton,imu,rgb,combined}/

Usage:
    python scripts/05_run_eda.py
    python scripts/05_run_eda.py --modality skeleton
    python scripts/05_run_eda.py --manifest-dir data/processed/MMAct/manifests
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.run_logger import RunLogger
from src.eda.common import setup_plot_style, load_class_names
from src.eda.orchestrator import run_all_eda
from src.eda.skeleton_eda import run_skeleton_eda
from src.eda.imu_eda import run_imu_eda
from src.eda.rgb_eda import run_rgb_eda


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", default="data/processed/MMAct/manifests")
    ap.add_argument("--classes-cfg", default="configs/classes_mmact.yaml")
    ap.add_argument("--output-root", default="outputs/eda")
    ap.add_argument("--modality", choices=["all", "skeleton", "imu", "rgb"],
                    default="all")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    setup_plot_style()

    # Run logger (log vào outputs/runs/eda/)
    run = RunLogger("eda", config={"modality": args.modality,
                                    "manifest_dir": args.manifest_dir},
                    tag=args.tag)

    run.info("=" * 70)
    run.info(f"  EDA — MMAct 3 modalities")
    run.info("=" * 70)
    run.info(f"  Manifest:    {args.manifest_dir}")
    run.info(f"  Output:      {args.output_root}")
    run.info(f"  Modality:    {args.modality}")
    run.info(f"  Log dir:     {run.path}")
    run.info("")

    class_names = load_class_names(args.classes_cfg)
    run.info(f"  Loaded {len(class_names)} class names")

    output_root = Path(args.output_root)
    manifest_dir = Path(args.manifest_dir)

    if args.modality == "all":
        run_all_eda(manifest_dir, class_names, output_root, log_fn=run.info)
    elif args.modality == "skeleton":
        run_skeleton_eda(manifest_dir / "train_cross_view.csv",
                         class_names, output_root / "skeleton",
                         log_fn=run.info)
    elif args.modality == "imu":
        run_imu_eda(manifest_dir / "train_cross_view.csv",
                    class_names, output_root / "imu",
                    log_fn=run.info)
    elif args.modality == "rgb":
        run_rgb_eda(manifest_dir / "train_cross_view.csv",
                    class_names, output_root / "rgb",
                    log_fn=run.info)

    run.info("")
    run.info("=" * 70)
    run.info("  DONE")
    run.info("=" * 70)
    run.info(f"  Output: {output_root}")
    run.info(f"  Log:    {run.path}")


if __name__ == "__main__":
    main()
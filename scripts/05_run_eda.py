"""
Chạy EDA cho cả 3 modality.
Output: outputs/eda/{skeleton,imu,rgb,combined}/

Usage:
    python scripts/05_run_eda.py
    python scripts/05_run_eda.py --modality skeleton
    python scripts/05_run_eda.py --protocol x_sub_scene --subset all \
        --manifest-dir data/processed/MMAct/manifests/protocols
"""
import argparse
import sys
from tempfile import TemporaryDirectory
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.common.run_logger import RunLogger
from src.eda.common import setup_plot_style, load_class_names
from src.eda.orchestrator import run_all_eda
from src.eda.skeleton_eda import run_skeleton_eda
from src.eda.imu_eda import run_imu_eda
from src.eda.rgb_eda import run_rgb_eda
from src.common.manifest_paths import MM_ACT_PROTOCOLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest-dir",
        default="data/processed/MMAct/manifests/protocols",
        help="Root containing the six protocol directories.",
    )
    ap.add_argument("--protocol", choices=MM_ACT_PROTOCOLS, default="x_view")
    ap.add_argument(
        "--subset",
        choices=["train", "val", "all"],
        default="train",
        help="Manifest subset to analyze; all combines train and val.",
    )
    ap.add_argument("--classes-cfg", default="configs/classes_mmact.yaml")
    ap.add_argument("--output-root", default="outputs/eda")
    ap.add_argument("--modality", choices=["all", "skeleton", "imu", "rgb"],
                    default="all")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    setup_plot_style()

    # Run logger (log vào outputs/runs/eda/)
    run = RunLogger("eda", config={"modality": args.modality,
                                    "manifest_dir": args.manifest_dir,
                                    "protocol": args.protocol,
                                    "subset": args.subset},
                    tag=args.tag)

    run.info("=" * 70)
    run.info(f"  EDA — MMAct 3 modalities")
    run.info("=" * 70)
    run.info(f"  Manifest:    {args.manifest_dir}")
    run.info(f"  Output:      {args.output_root}")
    run.info(f"  Modality:    {args.modality}")
    run.info(f"  Subset:      {args.subset}")
    run.info(f"  Log dir:     {run.path}")
    run.info("")

    class_names = load_class_names(args.classes_cfg)
    run.info(f"  Loaded {len(class_names)} class names")

    output_root = Path(args.output_root)
    manifest_dir = Path(args.manifest_dir) / args.protocol
    if args.subset == "train":
        _run_eda(args.modality, manifest_dir, class_names, output_root, run.info)
    else:
        with TemporaryDirectory(prefix="mmact_eda_") as temp_dir:
            selected_manifest_dir = Path(temp_dir)
            subsets = ("train", "val") if args.subset == "all" else ("val",)
            modalities = (
                ("skeleton", "imu", "rgb")
                if args.modality == "all" else (args.modality,)
            )
            for modality in modalities:
                frames = [
                    pd.read_csv(manifest_dir / modality / f"{subset}.csv")
                    for subset in subsets
                ]
                modality_dir = selected_manifest_dir / modality
                modality_dir.mkdir(parents=True, exist_ok=True)
                pd.concat(frames, ignore_index=True).to_csv(
                    modality_dir / "train.csv", index=False
                )
            _run_eda(
                args.modality, selected_manifest_dir,
                class_names, output_root, run.info,
            )

    run.info("")
    run.info("=" * 70)
    run.info("  DONE")
    run.info("=" * 70)
    run.info(f"  Output: {output_root}")
    run.info(f"  Log:    {run.path}")


def _run_eda(modality, manifest_dir, class_names, output_root, log_fn):
    if modality == "all":
        run_all_eda(manifest_dir, class_names, output_root, log_fn=log_fn)
    elif modality == "skeleton":
        run_skeleton_eda(manifest_dir / "skeleton" / "train.csv",
                         class_names, output_root / "skeleton",
                         log_fn=log_fn)
    elif modality == "imu":
        run_imu_eda(manifest_dir / "imu" / "train.csv",
                    class_names, output_root / "imu",
                    log_fn=log_fn)
    elif modality == "rgb":
        run_rgb_eda(manifest_dir / "rgb" / "train.csv",
                    class_names, output_root / "rgb",
                    log_fn=log_fn)


if __name__ == "__main__":
    main()
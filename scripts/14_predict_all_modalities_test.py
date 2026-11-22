"""Export train/test predictions for Skeleton, IMU and RGB in one run.

The selected checkpoints are never trained or modified.  For each modality
the script writes prediction artifacts into that modality's run directory:

    test_probs.npy, test_labels.npy, test_clip_ids.npy

With ``--export-train`` it also refreshes the train artifacts used by fusion.
All three modalities use the same configured train and test manifests, so the
result is ready for ``scripts/16_benchmark_fusion.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.manifest_paths import (
    MM_ACT_PROTOCOLS,
    resolve_train_val_manifests,
)
from src.data.imu.dataset import (
    IMUDataset,
    build_loaders as build_imu_loaders,
)
from src.data.rgb.dataset import RGBDataset
from src.data.skeleton.dataset import IN_CHANNELS, SkeletonDataset
from src.models.imu.architectures import build_imu_model
from src.models.rgb.video_backbone import build_video_model
from src.models.skeleton import build_model as build_skeleton_model


DATA_CONFIGS = {
    "skeleton": "configs/data/skeleton_mmact.yaml",
    "imu": "configs/data/imu_mmact.yaml",
    "rgb": "configs/data/rgb_mmact.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict an explicitly supplied external evaluation manifest"
    )
    parser.add_argument("--skeleton-run-dir", required=True)
    parser.add_argument("--imu-run-dir", required=True)
    parser.add_argument("--rgb-run-dir", required=True)
    parser.add_argument("--manifest-dir", default=None,
                        help="Optional protocol-manifest root override")
    parser.add_argument("--protocol", choices=MM_ACT_PROTOCOLS, default="x_view")
    parser.add_argument("--test-manifest", required=True,
                        help="External evaluation manifest; no internal test split is configured")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--skeleton-batch-size", type=int, default=256)
    parser.add_argument("--imu-batch-size", type=int, default=256)
    parser.add_argument("--rgb-batch-size", type=int, default=4)
    parser.add_argument("--rgb-cache-max-clips", type=int, default=4)
    parser.add_argument("--export-train", action="store_true",
                        help="Also refresh train_probs/labels/clip_ids")
    parser.add_argument("--summary-path", default=None,
                        help="Optional JSON summary output path")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def make_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    options = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": False,
    }
    if num_workers > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = 1
    return DataLoader(dataset, **options)


def build_model_and_datasets(
    modality: str,
    run_dir: Path,
    data_cfg_path: str,
    manifest_dir: str | None,
    protocol: str,
    test_manifest: Path,
    batch_size: int,
    num_workers: int,
    rgb_cache_max_clips: int,
):
    run_cfg = load_yaml(run_dir / "config.yaml")
    protocol = str(run_cfg.get("protocol", protocol))
    modality_manifest_dir = (
        Path(manifest_dir) / protocol / modality if manifest_dir else None
    )
    train_manifest, _ = resolve_train_val_manifests(
        data_cfg_path, modality_manifest_dir, protocol)
    train_cfg = run_cfg.get("train", {}) or {}
    model_cfg = run_cfg.get("model", {}) or {}

    if modality == "skeleton":
        data_cfg = load_yaml(Path(data_cfg_path))
        stream = str(model_cfg.get("stream", "rich"))
        target_len = int(data_cfg.get("sequence", {}).get("target_len", 64))
        common = dict(
            arch=str(model_cfg.get("arch", "ctrgcn_head")),
            num_classes=int(model_cfg.get("num_classes", 35)),
            num_joints=int(model_cfg.get("num_joints", 17)),
            in_channels=IN_CHANNELS[stream],
            target_len=target_len,
            dropout=float(model_cfg.get("dropout", 0.4)),
            part_drop=float(model_cfg.get("part_drop", 0.1)),
            drop_path=float(model_cfg.get("drop_path", 0.1)),
            tf_depth=int(model_cfg.get("tf_depth", 2)),
            tf_heads=int(model_cfg.get("tf_heads", 4)),
        )
        model = build_skeleton_model(**common)
        train_ds = SkeletonDataset(
            str(train_manifest), data_cfg_path, augment=False, stream=stream,
            aug_level=str(train_cfg.get("aug_level", "light")), cache=True,
        )
        test_ds = SkeletonDataset(
            str(test_manifest), data_cfg_path, augment=False, stream=stream,
            aug_level=str(train_cfg.get("aug_level", "light")), cache=True,
        )

    elif modality == "imu":
        feature_set = str(run_cfg.get("feature_set", "raw"))
        # Compute normalization only from the selected protocol's train data.
        train_loader, _ = build_imu_loaders(
            modality_manifest_dir, data_cfg_path, batch_size, num_workers=0,
            protocol=protocol, feature_set=feature_set,
        )
        stats = train_loader.dataset.stats
        train_ds = IMUDataset(
            str(train_manifest), data_cfg_path, stats=stats, training=False,
            feature_set=feature_set,
        )
        test_ds = IMUDataset(
            str(test_manifest), data_cfg_path, stats=stats, training=False,
            feature_set=feature_set,
        )
        model = build_imu_model(
            arch=str(model_cfg.get("arch", "tcn")),
            in_channels=int(model_cfg.get("in_channels", 3)),
            num_classes=int(model_cfg.get("num_classes", 35)),
            dropout=float(model_cfg.get("dropout", 0.3)),
        )

    elif modality == "rgb":
        model = build_video_model(
            arch=str(model_cfg.get("arch", "mc3_18")),
            num_classes=int(model_cfg.get("num_classes", 35)),
            # The checkpoint supplies all learned weights; do not download or
            # initialize a second pretrained model during prediction.
            pretrained=False,
            dropout=float(model_cfg.get("dropout", 0.3)),
        )
        train_ds = RGBDataset(
            str(train_manifest), data_cfg_path, augment=False,
            cache_in_memory=False, cache_max_clips=rgb_cache_max_clips,
        )
        test_ds = RGBDataset(
            str(test_manifest), data_cfg_path, augment=False,
            cache_in_memory=False, cache_max_clips=rgb_cache_max_clips,
        )
    else:
        raise ValueError(f"Unknown modality: {modality}")

    train_loader = make_loader(train_ds, batch_size, num_workers)
    test_loader = make_loader(test_ds, batch_size, num_workers)
    return model, train_ds, train_loader, test_ds, test_loader, run_cfg


def collect_predictions(model, loader, modality: str, device: torch.device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.inference_mode():
        for x, y in tqdm(loader, desc=f"predict/{modality}"):
            x = x.to(device, non_blocking=True)
            if modality == "imu" and x.ndim == 4:
                batch_size, crops = x.shape[:2]
                logits = model(x.flatten(0, 1)).reshape(batch_size, crops, -1)
                probs = torch.softmax(logits.float(), dim=-1).mean(dim=1)
            else:
                probs = torch.softmax(model(x).float(), dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(np.asarray(y))
    return np.concatenate(all_probs), np.concatenate(all_labels).astype(np.int64)


def save_split(run_dir: Path, split: str, dataset, probs, labels,
               manifest: Path) -> dict:
    clip_ids = dataset.df["clip_id"].astype(str).to_numpy()
    if len(clip_ids) != len(probs):
        raise RuntimeError(
            f"{run_dir} {split}: clip_ids={len(clip_ids)} probs={len(probs)}")

    preds = probs.argmax(axis=-1)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(
            labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(
            labels, preds, average="weighted", zero_division=0)),
        "num_samples": int(len(labels)),
        "source": "predict-only (from best.pt)",
        "manifest": str(manifest),
    }
    np.save(run_dir / f"{split}_probs.npy", probs.astype(np.float32))
    np.save(run_dir / f"{split}_labels.npy", labels.astype(np.int64))
    np.save(run_dir / f"{split}_clip_ids.npy", clip_ids)

    metrics_path = run_dir / "metrics.json"
    all_metrics = {}
    if metrics_path.exists():
        try:
            all_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            all_metrics = {}
    all_metrics[split] = metrics
    metrics_path.write_text(
        json.dumps(all_metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    run_dirs = {
        "skeleton": Path(args.skeleton_run_dir),
        "imu": Path(args.imu_run_dir),
        "rgb": Path(args.rgb_run_dir),
    }
    batch_sizes = {
        "skeleton": args.skeleton_batch_size,
        "imu": args.imu_batch_size,
        "rgb": args.rgb_batch_size,
    }

    test_manifest = Path(args.test_manifest)
    if not test_manifest.is_file():
        raise FileNotFoundError(f"Missing test manifest: {test_manifest}")

    summary = {
        "test_manifest": str(test_manifest),
        "protocol": args.protocol,
        "export_train": bool(args.export_train),
        "device": str(device),
        "modalities": {},
    }

    for modality, run_dir in run_dirs.items():
        ckpt_path = run_dir / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        data_cfg = DATA_CONFIGS[modality]
        print(f"\n{'=' * 70}\n{modality.upper()}\n{'=' * 70}")
        model, train_ds, train_loader, test_ds, test_loader, run_cfg = \
            build_model_and_datasets(
                modality, run_dir, data_cfg, args.manifest_dir, args.protocol,
                test_manifest, batch_sizes[modality], args.num_workers,
                args.rgb_cache_max_clips,
            )
        protocol = str(run_cfg.get("protocol", args.protocol))
        checkpoint = torch.load(
            ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)

        test_probs, test_labels = collect_predictions(
            model, test_loader, modality, device)
        test_metrics = save_split(
            run_dir, "test", test_ds, test_probs, test_labels, test_manifest)
        print(f"[{modality}] test n={test_metrics['num_samples']} "
              f"acc={test_metrics['accuracy']:.4f} "
              f"f1_macro={test_metrics['f1_macro']:.4f}")

        item = {
            "run_dir": str(run_dir),
            "test": test_metrics,
        }
        if args.export_train:
            train_manifest, _ = resolve_train_val_manifests(
                data_cfg,
                (Path(args.manifest_dir) / protocol / modality
                 if args.manifest_dir else None),
                protocol,
            )
            train_probs, train_labels = collect_predictions(
                model, train_loader, modality, device)
            item["train"] = save_split(
                run_dir, "train", train_ds, train_probs, train_labels,
                train_manifest,
            )
            print(f"[{modality}] train n={item['train']['num_samples']} "
                  f"f1_macro={item['train']['f1_macro']:.4f}")
        summary["modalities"][modality] = item
        del model, train_loader, test_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_path = Path(args.summary_path) if args.summary_path else (
        Path("outputs/runs/fusion") /
        f"predict_test_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved combined summary: {summary_path}")


if __name__ == "__main__":
    main()

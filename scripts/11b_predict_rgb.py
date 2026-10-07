"""
Predict-only RGB — sinh val_probs.npy + val_labels.npy từ best.pt.

Usage:
    python scripts/11b_predict_rgb.py
    python scripts/11b_predict_rgb.py --run-dir outputs/runs/rgb/20260927_235612_mc3_18
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

from src.data.rgb.dataset import build_loaders
from src.models.rgb.video_backbone import build_video_model, SUPPORTED_ARCHS


def collect_predictions(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="  predict"):
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(np.asarray(y))
    return np.concatenate(all_probs, 0), np.concatenate(all_labels, 0)


def find_latest_rgb_run(runs_root: Path) -> Path:
    rgb_root = runs_root / "rgb"
    runs = [d for d in rgb_root.iterdir() if d.is_dir()]
    runs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    for r in runs:
        if (r / "best.pt").exists():
            return r
    raise FileNotFoundError(f"Không tìm thấy run RGB nào có best.pt trong {rgb_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None,
                    help="Folder chứa best.pt (mặc định: latest)")
    ap.add_argument("--manifest-dir", default="data/processed/MMAct/manifests")
    ap.add_argument("--data-cfg", default="configs/data/rgb_mmact.yaml")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    # ── Xác định run folder ──
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_latest_rgb_run(Path("outputs/runs"))

    if not (run_dir / "best.pt").exists():
        print(f"✗ Không có best.pt trong {run_dir}")
        return

    print("=" * 70)
    print(f"  RGB PREDICT-ONLY")
    print("=" * 70)
    print(f"  Run dir:  {run_dir}")
    print(f"  Device:   {args.device}")
    print()

    # ── Load checkpoint ──
    ckpt = torch.load(run_dir / "best.pt", map_location=args.device,
                       weights_only=False)
    arch = ckpt.get("arch", "mc3_18")
    cfg = ckpt.get("config", {})
    best_f1_ckpt = ckpt.get("best_f1", None)
    print(f"  Checkpoint epoch:    {ckpt.get('epoch')}")
    print(f"  Checkpoint best_f1:  {best_f1_ckpt}")
    print(f"  Arch:                {arch}")
    print()

    # ── Load data ──
    print("[Data] Loading val set...")
    _, val_loader = build_loaders(
        args.manifest_dir, args.data_cfg, args.batch_size,
        num_workers=args.num_workers, max_samples=None)

    # ── Build model ──
    num_classes = cfg.get("model", {}).get("num_classes", 32)
    dropout = cfg.get("model", {}).get("dropout", 0.3)

    model = build_video_model(
        arch=arch,
        num_classes=num_classes,
        pretrained=False,   # load weights từ checkpoint
        dropout=dropout,
    ).to(args.device)

    model.load_state_dict(ckpt["model_state"])
    print(f"[Model] Loaded {arch}  ({sum(p.numel() for p in model.parameters())/1e6:.2f}M)")

    # ── Predict ──
    print()
    print("─" * 70)
    print("  PREDICTING VAL SET")
    print("─" * 70)

    probs, labels = collect_predictions(model, val_loader, args.device)
    preds = probs.argmax(-1)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)

    print()
    print(f"  Val accuracy:     {acc:.4f}")
    print(f"  Val F1 macro:     {f1_macro:.4f}")
    print(f"  Val F1 weighted:  {f1_weighted:.4f}")
    print(f"  Samples:          {len(labels)}")
    print(f"  Probs shape:      {probs.shape}")

    # ── Save ──
    val_probs_path = run_dir / "val_probs.npy"
    val_labels_path = run_dir / "val_labels.npy"
    np.save(val_probs_path, probs.astype(np.float32))
    np.save(val_labels_path, labels.astype(np.int64))
    print()
    print(f"  ✓ Saved: {val_probs_path.relative_to(Path.cwd())}")
    print(f"  ✓ Saved: {val_labels_path.relative_to(Path.cwd())}")

    # ── Update metrics.json nếu có ──
    metrics_path = run_dir / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}
    if "val" not in metrics:
        metrics["val"] = {}
    metrics["val"]["accuracy"] = float(acc)
    metrics["val"]["f1_macro"] = float(f1_macro)
    metrics["val"]["f1_weighted"] = float(f1_weighted)
    metrics["val"]["num_samples"] = int(len(labels))
    metrics["val"]["source"] = "predict-only (from best.pt)"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    print(f"  ✓ Updated: {metrics_path.relative_to(Path.cwd())}")
    print()
    print("✓ Done.")


if __name__ == "__main__":
    main()
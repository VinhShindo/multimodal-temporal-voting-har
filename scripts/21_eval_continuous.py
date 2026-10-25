"""Đánh giá continuous (bước 21)."""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.common.run_logger import RunLogger
from src.evaluation.continuous_metrics import (
    frame_wise_accuracy, match_segments, boundary_metrics, load_gt_segments,
)
from src.postprocess.linking import Segment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sessions-root",
                    default="data/processed/MMAct/continuous")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--boundary-tol", type=float, default=0.3)
    ap.add_argument("--tag", default="eval")
    args = ap.parse_args()

    run = RunLogger("continuous_eval", tag=args.tag)
    run.info("=" * 70)
    run.info(f"  EVAL — {run.run_name}")
    run.info("=" * 70)

    run_dir = Path(args.run_dir)
    sessions_root = Path(args.sessions_root)

    rows = []
    session_dirs = sorted([d for d in run_dir.iterdir()
                            if d.is_dir() and (d / "segments.csv").exists()])
    run.info(f"Sessions to eval: {len(session_dirs)}\n")

    for sdir in session_dirs:
        sid = sdir.name
        gt_dir = sessions_root / sid
        if not (gt_dir / "annotation.csv").exists():
            continue

        preds = []
        with open(sdir / "segments.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                preds.append(Segment(
                    start=float(r["start"]), end=float(r["end"]),
                    class_id=int(r["class_id"]),
                    class_name=r["class_name"],
                    confidence=float(r.get("confidence", 1.0))))

        gt = load_gt_segments(gt_dir / "annotation.csv")
        meta = json.loads((gt_dir / "metadata.json").read_text(encoding="utf-8"))
        duration = meta["duration_sec"]

        fw = frame_wise_accuracy(preds, gt, duration, fps=args.fps)
        sm = match_segments(preds, gt, iou_threshold=args.iou_threshold)
        gt_b = [(s.start, s.end) for s in gt]
        pr_b = [(s.start, s.end) for s in preds]
        bm = boundary_metrics(pr_b, gt_b, tolerance_sec=args.boundary_tol)

        row = {"session_id": sid, "duration": duration,
                "n_gt": len(gt), "n_pred": len(preds),
                "frame_acc": fw, "seg_f1": sm["f1"],
                "seg_prec": sm["precision"], "seg_rec": sm["recall"],
                "mean_iou": sm["mean_iou"],
                "bound_f1": bm["f1"], "bound_prec": bm["precision"],
                "bound_rec": bm["recall"]}
        rows.append(row)
        run.info(f"  [{sid}] dur={duration:.1f}s gt={len(gt)} pred={len(preds)}  "
                 f"frame_acc={fw:.4f}  seg_f1={sm['f1']:.4f}  "
                 f"iou={sm['mean_iou']:.4f}  bound_f1={bm['f1']:.4f}")

    if not rows:
        run.info("  ✗ Không có session nào"); return

    def mean(k): return float(np.mean([r[k] for r in rows]))
    summary = {
        "n_sessions": len(rows),
        "mean_frame_acc":  mean("frame_acc"),
        "mean_seg_f1":     mean("seg_f1"),
        "mean_seg_prec":   mean("seg_prec"),
        "mean_seg_rec":    mean("seg_rec"),
        "mean_iou":        mean("mean_iou"),
        "mean_bound_f1":   mean("bound_f1"),
        "mean_bound_prec": mean("bound_prec"),
        "mean_bound_rec":  mean("bound_rec"),
    }

    with open(run.path / "per_session.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    (run.path / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    run.info("")
    run.info("=" * 70)
    run.info("  SUMMARY")
    run.info("=" * 70)
    run.info(f"  Sessions:           {summary['n_sessions']}")
    run.info(f"  Frame-wise acc:     {summary['mean_frame_acc']:.4f}")
    run.info(f"  Segment F1:         {summary['mean_seg_f1']:.4f}")
    run.info(f"  Segment Precision:  {summary['mean_seg_prec']:.4f}")
    run.info(f"  Segment Recall:     {summary['mean_seg_rec']:.4f}")
    run.info(f"  Mean temporal IoU:  {summary['mean_iou']:.4f}")
    run.info(f"  Boundary F1:        {summary['mean_bound_f1']:.4f}")
    run.info(f"  Boundary Precision: {summary['mean_bound_prec']:.4f}")
    run.info(f"  Boundary Recall:    {summary['mean_bound_rec']:.4f}")
    run.info("")
    run.info(f"  Run dir: {run.path}")


if __name__ == "__main__":
    main()
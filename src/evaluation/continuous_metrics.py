"""Metrics cho continuous recognition."""
import numpy as np


def frame_wise_accuracy(pred_segs, gt_segs, duration, fps=30.0):
    n = int(duration * fps)
    if n <= 0: return 0.0
    pl = np.full(n, -1, dtype=np.int32)
    gl = np.full(n, -1, dtype=np.int32)
    for s in pred_segs:
        i0 = max(0, int(s.start * fps))
        i1 = min(n, int(s.end * fps))
        pl[i0:i1] = s.class_id
    for s in gt_segs:
        i0 = max(0, int(s.start * fps))
        i1 = min(n, int(s.end * fps))
        gl[i0:i1] = s.class_id
    return float((pl == gl).mean())


def temporal_iou(pred, gt):
    inter = max(0.0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return inter / union if union > 0 else 0.0


def match_segments(pred_segs, gt_segs, iou_threshold=0.5):
    matched_gt, matched_pred, ious = set(), set(), []
    for pi, p in enumerate(pred_segs):
        best_iou, best_gi = 0.0, -1
        for gi, g in enumerate(gt_segs):
            if gi in matched_gt: continue
            if p.class_id != g.class_id: continue
            iou = temporal_iou((p.start, p.end), (g.start, g.end))
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= iou_threshold and best_gi >= 0:
            matched_gt.add(best_gi); matched_pred.add(pi); ious.append(best_iou)

    tp = len(matched_pred); fp = len(pred_segs) - tp; fn = len(gt_segs) - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
            "n_gt": len(gt_segs), "n_pred": len(pred_segs)}


def boundary_metrics(pred_bounds, gt_bounds, tolerance_sec=0.3):
    gt_b = [x for g in gt_bounds for x in (g[0], g[1])]
    pr_b = [x for g in pred_bounds for x in (g[0], g[1])]
    if not gt_b or not pr_b:
        return {"matched": 0, "n_gt": len(gt_b), "n_pred": len(pr_b),
                "precision": 0.0, "recall": 0.0, "f1": 0.0}

    matched_gt, matched_pred = set(), set()
    for pi, pb in enumerate(pr_b):
        best_dt, best_gi = tolerance_sec + 1, -1
        for gi, gb in enumerate(gt_b):
            if gi in matched_gt: continue
            dt = abs(pb - gb)
            if dt < best_dt:
                best_dt, best_gi = dt, gi
        if best_dt <= tolerance_sec and best_gi >= 0:
            matched_gt.add(best_gi); matched_pred.add(pi)
    tp = len(matched_pred); fp = len(pr_b) - tp; fn = len(gt_b) - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
            "n_gt": len(gt_b), "n_pred": len(pr_b)}


def load_gt_segments(annotation_csv):
    import pandas as pd
    from src.postprocess.linking import Segment
    df = pd.read_csv(annotation_csv)
    return [Segment(start=float(r["start_sec"]), end=float(r["end_sec"]),
                    class_id=int(r["class_id"]),
                    class_name=r["class_name"]) for _, r in df.iterrows()]
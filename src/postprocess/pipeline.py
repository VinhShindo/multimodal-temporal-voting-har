"""Postprocess pipeline — Gap detection + Frame-level voting + Segment refinement."""
import numpy as np

from src.postprocess.linking import Segment, link_segments
from src.postprocess.removing import remove_short


# Sentinel class — đánh dấu frame không phải action
NO_ACTION_ID = -1


# ============================================================
# Gap detection
# ============================================================
def detect_gap_frames(skel, fps=30.0):
    """
    Trả về bool mask (T,) — True nếu frame là gap (không có người).

    Gap = skeleton all-zero (energy < ngưỡng) + không có chuyển động.
    """
    n = skel.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    # Tổng |x| + |y| của tất cả joints
    energy = np.abs(skel[:, :, :2]).sum(axis=(1, 2))
    return energy < 1e-3


# ============================================================
# Frame-level majority voting (with gap override + confidence)
# ============================================================
def window_predictions_to_frame_labels(preds, duration_sec, fps=30.0,
                                        num_classes=32,
                                        gap_mask=None,
                                        conf_threshold=0.0):
    """
    Mỗi frame nhận class từ đa số window phủ lên nó.

    Args:
        preds: list[dict{start, end, class_id, confidence}]
        duration_sec: total duration
        fps: frame rate
        num_classes: số class
        gap_mask: (n_frames,) bool — True tại gap frames (override → NO_ACTION)
        conf_threshold: nếu max vote confidence < threshold → NO_ACTION

    Returns:
        frame_labels: (n_frames,) int32 — NO_ACTION_ID tại gap/uncertain
    """
    n_frames = int(round(duration_sec * fps))
    if n_frames <= 0:
        return np.zeros(0, dtype=np.int32)

    votes = np.zeros((n_frames, num_classes), dtype=np.float32)

    for p in preds:
        f0 = max(0, int(round(p["start"] * fps)))
        f1 = min(n_frames, int(round(p["end"] * fps)))
        if f1 <= f0:
            continue
        conf = float(p.get("confidence", 1.0))
        votes[f0:f1, p["class_id"]] += conf

    # Argmax + confidence threshold
    max_votes = votes.max(axis=1)
    labels = votes.argmax(axis=1).astype(np.int32)

    if conf_threshold > 0:
        uncertain = max_votes < conf_threshold
        labels[uncertain] = NO_ACTION_ID

    # Override gap frames
    if gap_mask is not None and len(gap_mask) == n_frames:
        labels[gap_mask] = NO_ACTION_ID

    return labels


# ============================================================
# Group frame labels → segments (bỏ NO_ACTION)
# ============================================================
def frame_labels_to_segments(frame_labels, fps=30.0, class_names=None,
                              min_duration_sec=0.3):
    """Group consecutive frames cùng label → list Segment. Bỏ NO_ACTION."""
    if len(frame_labels) == 0:
        return []
    min_frames = max(1, int(round(min_duration_sec * fps)))

    segs = []
    start = 0
    cur = int(frame_labels[0])

    for i in range(1, len(frame_labels) + 1):
        # End of run
        is_end = (i == len(frame_labels))
        is_change = (not is_end) and (int(frame_labels[i]) != cur)

        if is_end or is_change:
            duration_frames = i - start
            if cur != NO_ACTION_ID and duration_frames >= min_frames:
                segs.append(Segment(
                    start=start / fps,
                    end=i / fps,
                    class_id=cur,
                    class_name=(class_names[cur]
                                if class_names and 0 <= cur < len(class_names)
                                else str(cur)),
                    confidence=1.0))
            if not is_end:
                start = i
                cur = int(frame_labels[i])

    return segs


# ============================================================
# Segment-level refinement — vote lại từ windows trong mỗi segment
# ============================================================
def refine_segments_by_voting(segments, preds, fps=30.0,
                               num_classes=32, class_names=None):
    """
    Với mỗi segment, vote class từ các windows có overlap > 50%.
    """
    if not segments or not preds:
        return segments

    refined = []
    for s in segments:
        s_dur = s.end - s.start
        if s_dur <= 0:
            refined.append(s)
            continue

        votes = np.zeros(num_classes, dtype=np.float32)
        total_w = 0.0

        for p in preds:
            # Overlap
            ov_start = max(p["start"], s.start)
            ov_end = min(p["end"], s.end)
            ov = max(0.0, ov_end - ov_start)
            if ov <= 0:
                continue
            # ★ Chỉ đếm nếu > 50% window nằm trong segment
            p_dur = p["end"] - p["start"]
            if p_dur <= 0:
                continue
            ratio = ov / p_dur
            if ratio < 0.5:
                continue

            w = float(p.get("confidence", 1.0)) * ratio
            votes[p["class_id"]] += w
            total_w += w

        if total_w > 0:
            best_cid = int(votes.argmax())
            best_conf = float(votes[best_cid] / total_w)
        else:
            best_cid = s.class_id
            best_conf = s.confidence

        refined.append(Segment(
            start=s.start, end=s.end,
            class_id=best_cid,
            class_name=(class_names[best_cid]
                        if class_names and 0 <= best_cid < len(class_names)
                        else str(best_cid)),
            confidence=best_conf,
        ))
    return refined


# ============================================================
# Legacy
# ============================================================
def predictions_to_segments(preds, class_names=None):
    """DEPRECATED."""
    if not preds:
        return []
    segs = []
    cur = None
    for p in preds:
        cid = p["class_id"]
        if cur is None or cur.class_id != cid:
            if cur is not None:
                segs.append(cur)
            cur = Segment(
                start=p["start"], end=p["end"], class_id=cid,
                class_name=class_names[cid] if class_names else str(cid),
                confidence=p.get("confidence", 1.0))
        else:
            cur.end = p["end"]
            cur.confidence = max(cur.confidence, p.get("confidence", 0))
    if cur is not None:
        segs.append(cur)
    return segs


# ============================================================
# Main API — Full pipeline
# ============================================================
def run_postprocess(preds, class_names, duration,
                     fps=30.0, num_classes=32,
                     skel=None, gap_mask=None,
                     smooth_kernel=0, t_avg=2.0, min_duration=1.0,
                     conf_threshold=0.05,
                     use_refinement=True):
    """
    Full pipeline:
      1. Detect gap frames (auto từ skeleton nếu chưa có)
      2. Frame-level majority voting (với confidence threshold)
      3. Group → segments
      4. Segment-level voting refinement
      5. Linking (merge gap nhỏ)
      6. Removing (loại segment ngắn)

    Args:
        preds: list[dict{start, end, class_id, confidence}]
        class_names: list[str]
        duration: video duration (s)
        fps: frame rate
        num_classes: số class
        skel: (T, 17, 3) — optional, dùng để auto gap detection
        gap_mask: (n_frames,) bool — optional, override
        conf_threshold: ngưỡng confidence tối thiểu để gán class
        use_refinement: bật segment-level voting
    """
    if not preds:
        return []

    # 1. Gap detection
    if gap_mask is None and skel is not None:
        gap_mask = detect_gap_frames(skel, fps=fps)

    # 2. Frame-level voting
    frame_labels = window_predictions_to_frame_labels(
        preds, duration, fps=fps,
        num_classes=num_classes,
        gap_mask=gap_mask,
        conf_threshold=conf_threshold,
    )

    # 3. Group → segments
    segs = frame_labels_to_segments(
        frame_labels, fps=fps, class_names=class_names,
        min_duration_sec=min_duration)

    # 4. Segment-level refinement
    if use_refinement and segs:
        segs = refine_segments_by_voting(
            segs, preds, fps=fps,
            num_classes=num_classes, class_names=class_names)

    # 5. Linking
    segs = link_segments(segs, t_avg=min(t_avg, 0.5))

    # 6. Remove short
    segs = remove_short(segs, min_duration=min_duration)

    return segs
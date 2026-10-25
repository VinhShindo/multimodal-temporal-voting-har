"""
Grid search weight cho weighted probability voting.
Caller quyết định fit split; benchmark không được truyền eval labels vào đây
để tìm trọng số nếu muốn đánh giá leakage-safe.
"""
import numpy as np
from sklearn.metrics import f1_score

from src.fusion.voting import weighted_probability


def grid_search_weights(probs_dict: dict, labels: np.ndarray,
                        step: float = 0.1, metric: str = "f1_macro",
                        log_fn=None) -> dict:
    """
    Return {"weights": {...}, "score": float, "all_results": [...]}.
    ``score`` là score trên chính dữ liệu được truyền vào và chỉ dùng để
    chọn weights trong fit split.
    """
    def log(m):
        if log_fn:
            log_fn(m)

    mods = list(probs_dict.keys())
    M = len(mods)

    if M == 1:
        w = {mods[0]: 1.0}
        fused = weighted_probability(probs_dict, w)
        preds = fused.argmax(-1)
        score = _score(labels, preds, metric)
        return {"weights": w, "score": score, "all_results": [(w, score)]}

    all_results = []

    if M == 2:
        grid = np.arange(0.0, 1.0 + 1e-9, step)
        for w1 in grid:
            w2 = 1.0 - w1
            if w2 < -1e-9:
                continue
            w = {mods[0]: float(w1), mods[1]: float(max(0.0, w2))}
            score = _eval(probs_dict, labels, w, metric)
            all_results.append((w, score))

    elif M == 3:
        grid = np.arange(0.0, 1.0 + 1e-9, step)
        for w1 in grid:
            for w2 in grid:
                w3 = 1.0 - w1 - w2
                if w3 < -1e-9:
                    continue
                w = {mods[0]: float(w1), mods[1]: float(w2),
                     mods[2]: float(max(0.0, w3))}
                score = _eval(probs_dict, labels, w, metric)
                all_results.append((w, score))
    else:
        raise ValueError(f"Grid search hỗ trợ 2-3 modalities, không hỗ trợ {M}")

    # Best
    all_results.sort(key=lambda x: -x[1])
    best_w, best_score = all_results[0]
    log(f"  Grid search: {len(all_results)} điểm, best score={best_score:.4f}")
    log(f"  Best weights: {best_w}")

    return {"weights": best_w, "score": best_score, "all_results": all_results}


def _eval(probs_dict, labels, w, metric):
    fused = weighted_probability(probs_dict, w)
    preds = fused.argmax(-1)
    return _score(labels, preds, metric)


def _score(labels, preds, metric):
    if metric == "f1_macro":
        return float(f1_score(labels, preds, average="macro", zero_division=0))
    if metric == "accuracy":
        return float((labels == preds).mean())
    raise ValueError(f"Unknown metric: {metric}")

"""Grid search weight trên validation set."""
import itertools
import numpy as np


def grid_search_weights(val_records, val_labels, cmap, grid_config, method="weighted"):
    """Trả về weight tốt nhất theo accuracy."""
    from src.fusion.voting import fuse
    best_w, best_acc = None, -1.0
    for w in grid_config:
        weights = {"rgb": w[0], "skeleton": w[1], "imu": w[2]}
        correct = 0
        for rec, gt in zip(val_records, val_labels):
            pred = fuse(rec, method, cmap, weights)
            if pred.class_id == gt:
                correct += 1
        acc = correct / max(len(val_records), 1)
        if acc > best_acc:
            best_acc, best_w = acc, weights
    return best_w, best_acc

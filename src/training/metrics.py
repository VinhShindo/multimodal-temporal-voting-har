"""Metrics cho isolated classification."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def compute_isolated_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

"""Chạy inference trên isolated test set → trả list Prediction."""
import torch

from src.common.schema import Prediction
from src.common.classes import load_classes


@torch.no_grad()
def predict_isolated(model, loader, device="cuda", classes_path="configs/classes.yaml"):
    cmap = load_classes(classes_path)
    model.eval().to(device)
    preds = []
    for x, y, meta in loader:
        x = x.to(device)
        probs = model.predict_proba(x).cpu().numpy()
        for i in range(len(probs)):
            cid = int(probs[i].argmax())
            preds.append(Prediction(
                window_id=i,
                start=0.0, end=0.0,
                class_id=cid,
                class_name=cmap.to_name(cid),
                confidence=float(probs[i, cid]),
                probabilities=probs[i].tolist(),
            ))
    return preds

"""Wrapper: model + preprocess + postprocess → Prediction."""
import numpy as np
import torch

from src.common.schema import Prediction


class Predictor:
    def __init__(self, model, preprocess_fn, class_names, device="cuda"):
        self.model = model.eval().to(device)
        self.preprocess = preprocess_fn
        self.class_names = class_names
        self.device = device

    @torch.no_grad()
    def predict(self, raw_input, window_id: int, start: float, end: float) -> Prediction:
        x = self.preprocess(raw_input)
        x = torch.as_tensor(x[None]).to(self.device)
        probs = self.model.predict_proba(x)[0].cpu().numpy()
        cid = int(probs.argmax())
        return Prediction(
            window_id=window_id, start=start, end=end,
            class_id=cid, class_name=self.class_names[cid],
            confidence=float(probs[cid]),
            probabilities=probs.tolist(),
        )

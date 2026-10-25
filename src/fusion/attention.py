"""Learned attention-based fusion for modality prediction vectors.

The isolated modality models in this project expose class probabilities rather
than a shared feature space.  This module therefore treats each ``(N, C)``
probability vector as one modality token and learns a small meta-fuser on the
training predictions.  It is intentionally independent from the backbone
models, so it can compare attention fusion with the existing voting fusion.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ATTENTION_METHODS = ("self", "cross", "adaptive", "gate_cross")


class AttentionFusion(nn.Module):
    """Fuse modality probability vectors with one of four attention modes.

    Args:
        num_modalities: Number of modality tokens, normally 2 or 3.
        num_classes: Number of classes in each probability vector.
        method: ``self``, ``cross``, ``adaptive`` or ``gate_cross``.
        hidden_dim: Token embedding size.
        num_heads: Number of attention heads.
    """

    def __init__(self, num_modalities: int, num_classes: int,
                 method: str = "self", hidden_dim: int = 64,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        method = method.lower()
        if method not in ATTENTION_METHODS:
            raise ValueError(f"Unknown attention method: {method}")
        if num_modalities < 1 or num_classes < 1:
            raise ValueError("num_modalities and num_classes must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.num_modalities = int(num_modalities)
        self.num_classes = int(num_classes)
        self.method = method
        self.hidden_dim = int(hidden_dim)

        self.input_norm = nn.LayerNorm(num_classes)
        self.input_proj = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.modality_embedding = nn.Parameter(
            torch.zeros(1, num_modalities, hidden_dim))
        nn.init.normal_(self.modality_embedding, std=0.02)

        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.query_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.query_token, std=0.02)
        self.adaptive_score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _encode(self, probs: torch.Tensor) -> torch.Tensor:
        if probs.ndim != 3:
            raise ValueError(
                f"Expected probabilities with shape (B, M, C), got {tuple(probs.shape)}")
        if probs.shape[1] != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modalities, got {probs.shape[1]}")
        if probs.shape[2] != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} classes, got {probs.shape[2]}")

        # Log probabilities make confidence differences visible to the small
        # meta-model while keeping the input bounded and numerically stable.
        x = torch.log(probs.clamp_min(1e-6))
        x = self.input_norm(x)
        return self.input_proj(x) + self.modality_embedding

    def forward(self, probs: torch.Tensor,
                return_attention: bool = False):
        tokens = self._encode(probs)
        info = {}

        if self.method == "self":
            attended, weights = self.self_attn(tokens, tokens, tokens,
                                               need_weights=return_attention)
            tokens = self.norm1(tokens + attended)
            tokens = self.norm2(tokens + self.ffn(tokens))
            fused = tokens.mean(dim=1)
            if return_attention:
                info["attention"] = weights

        elif self.method == "cross":
            query = self.query_token.expand(tokens.shape[0], -1, -1)
            attended, weights = self.cross_attn(query, tokens, tokens,
                                                need_weights=return_attention)
            fused = self.norm1(query + attended).squeeze(1)
            fused = self.norm2(fused + self.ffn(fused))
            if return_attention:
                info["attention"] = weights

        elif self.method == "adaptive":
            scores = self.adaptive_score(tokens).squeeze(-1)
            weights = scores.softmax(dim=1)
            fused = (tokens * weights.unsqueeze(-1)).sum(dim=1)
            if return_attention:
                info["attention"] = weights

        else:  # gate_cross
            if self.num_modalities == 1:
                # There is no other token to attend to; this keeps the class
                # safe for E1/E2/E3 while the benchmark reports single as the
                # meaningful baseline for those experiments.
                context = tokens
                weights = None
            else:
                # Prevent a modality from attending to itself.  This is a
                # genuine cross-modal attention operation, not self-attention.
                mask = torch.eye(self.num_modalities, device=tokens.device,
                                 dtype=torch.bool)
                context, weights = self.cross_attn(
                    tokens, tokens, tokens, attn_mask=mask,
                    need_weights=return_attention)
                context = self.norm1(tokens + context)
            gates = self.gate(torch.cat([tokens, context], dim=-1))
            gated = gates * context + (1.0 - gates) * tokens
            fused = self.norm2(gated + self.ffn(gated)).mean(dim=1)
            if return_attention:
                info["attention"] = weights
                info["gates"] = gates.mean(dim=-1)

        logits = self.classifier(fused)
        return (logits, info) if return_attention else logits


def _predict(model: nn.Module, probs: np.ndarray, device: torch.device,
             batch_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    x = torch.as_tensor(probs, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    all_probs, all_preds = [], []
    with torch.no_grad():
        for (batch,) in loader:
            logits = model(batch.to(device))
            p = logits.softmax(dim=-1)
            all_probs.append(p.cpu().numpy())
            all_preds.append(p.argmax(dim=-1).cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_preds)


def fit_attention(probs_train: np.ndarray, labels_train: np.ndarray,
                  probs_eval: np.ndarray, method: str,
                  hidden_dim: int = 64, num_heads: int = 4,
                  dropout: float = 0.1, epochs: int = 60,
                  batch_size: int = 512, lr: float = 2e-3,
                  weight_decay: float = 1e-4, patience: int = 10,
                  seed: int = 42, device: str = "cpu") -> dict:
    """Train one attention meta-fuser on train predictions.

    A stratified slice of the training predictions is used only for early
    stopping.  ``probs_eval`` remains untouched until the final prediction.
    """
    probs_train = np.asarray(probs_train, dtype=np.float32)
    probs_eval = np.asarray(probs_eval, dtype=np.float32)
    labels_train = np.asarray(labels_train, dtype=np.int64)
    if probs_train.ndim != 3 or probs_eval.ndim != 3:
        raise ValueError("Attention inputs must have shape (N, M, C)")
    if len(probs_train) != len(labels_train):
        raise ValueError("Training probabilities and labels have different lengths")
    if probs_train.shape[1:] != probs_eval.shape[1:]:
        raise ValueError("Train/eval modality or class dimensions do not match")

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    model = AttentionFusion(
        probs_train.shape[1], probs_train.shape[2], method=method,
        hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout).to(dev)

    # Keep early stopping independent from the final validation set.
    if len(labels_train) >= 20 and len(np.unique(labels_train)) > 1:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15,
                                          random_state=seed)
        fit_idx, stop_idx = next(splitter.split(probs_train, labels_train))
    else:
        fit_idx = np.arange(len(labels_train))
        stop_idx = fit_idx

    x_fit = torch.as_tensor(probs_train[fit_idx], dtype=torch.float32)
    y_fit = torch.as_tensor(labels_train[fit_idx], dtype=torch.long)
    x_stop = torch.as_tensor(probs_train[stop_idx], dtype=torch.float32)
    y_stop = torch.as_tensor(labels_train[stop_idx], dtype=torch.long)
    loader = DataLoader(TensorDataset(x_fit, y_fit), batch_size=batch_size,
                        shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_score = -float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch.to(dev)), y_batch.to(dev))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            stop_logits = []
            for start in range(0, len(x_stop), batch_size):
                stop_logits.append(model(x_stop[start:start + batch_size].to(dev)))
            stop_preds = torch.cat(stop_logits).argmax(dim=-1).cpu().numpy()
        from sklearn.metrics import f1_score
        score = f1_score(labels_train[stop_idx], stop_preds,
                         average="macro", zero_division=0)
        if score > best_score + 1e-7:
            best_score = float(score)
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    eval_probs, eval_preds = _predict(model, probs_eval, dev,
                                      batch_size=batch_size)
    return {
        "model": model,
        "probs": eval_probs,
        "preds": eval_preds,
        "best_epoch": best_epoch,
        "stop_f1_macro": float(best_score),
    }

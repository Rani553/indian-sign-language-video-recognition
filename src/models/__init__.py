"""Model factory shared by training, evaluation, the benchmark and the demo.

``build_model(cfg, num_classes, input_dim)`` reads ``cfg["model"]`` and returns an
``nn.Module`` with ``forward(x) -> logits (B, num_classes)``.
Models are imported lazily, so each pipeline only needs its own dependencies.
"""
from __future__ import annotations

import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Number of parameters (optionally only the ones that are being trained)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def build_model(cfg: dict, num_classes: int, input_dim: int | None = None) -> nn.Module:
    """Create the model named in ``cfg["model"]["name"]``.

    ``input_dim`` is the keypoint feature size (pose models only; ``None`` for frames).
    """
    m = dict(cfg["model"])
    name = m.pop("name")
    # Keys that belong to the training script, not to the model constructor.
    m.pop("unfreeze_epoch", None)
    if name == "cnn_lstm":
        from src.models.cnn_lstm import CNNLSTM
        return CNNLSTM(num_classes=num_classes, **m)
    # The two pose configs inherit from each other (pose_transformer.yaml -> pose_lstm.yaml), so
    # the merged ``model`` section can contain the other model's own keys. Drop exactly those
    # (a typo in any other key still raises a TypeError).
    if name == "pose_lstm":
        from src.models.pose_lstm import PoseLSTM
        for k in ("d_model", "nhead", "dim_feedforward", "max_len"):
            m.pop(k, None)
        return PoseLSTM(num_classes=num_classes, input_dim=input_dim, **m)
    if name == "pose_transformer":
        from src.models.pose_transformer import PoseTransformer
        for k in ("hidden_size", "bidirectional", "pooling"):
            m.pop(k, None)
        return PoseTransformer(num_classes=num_classes, input_dim=input_dim, **m)
    raise ValueError(f"Unknown model name: {name}")

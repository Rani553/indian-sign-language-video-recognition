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
    # Member 3 adds "pose_lstm" and "pose_transformer" here.
    raise ValueError(f"Unknown model name: {name}")

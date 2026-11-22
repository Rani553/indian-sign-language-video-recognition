"""Load a trained checkpoint (``best.pt``) back into a ready-to-use model.

Every checkpoint written by ``Trainer`` is a dict with (see src/training/trainer.py, common.py):
    model_state, config, label_map, num_classes, input_dim, epoch, best_val_acc

``load_checkpoint`` rebuilds the model from the stored config, so evaluation, the speed
benchmark, the predictor and the demo all start from the same code.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn

from src.models import build_model

REQUIRED_KEYS = ("model_state", "config", "label_map", "num_classes")


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict, dict]:
    """Return ``(model, cfg, ckpt)``: the model in eval mode on ``device``, its config and the raw checkpoint.

    ``weights_only=False`` is needed because the checkpoint holds a config and a label map, not only
    tensors. Pickle can run code, so only load checkpoints that come from the team.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    missing = [k for k in REQUIRED_KEYS if k not in ckpt]
    if missing:
        raise ValueError(f"{path} is not a checkpoint of this project (missing keys: {missing})")
    cfg = copy.deepcopy(ckpt["config"])
    # The weights come from the checkpoint, so do not download ImageNet weights first.
    if "pretrained" in cfg.get("model", {}):
        cfg["model"]["pretrained"] = False
    model = build_model(cfg, ckpt["num_classes"], ckpt.get("input_dim"))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, cfg, ckpt


def class_names_from(label_map: dict[str, int]) -> list[str]:
    """Class names in index order: ``names[i]`` is the word for label ``i``."""
    return [c for c, _ in sorted(label_map.items(), key=lambda kv: kv[1])]


def experiment_name(path: str | Path, cfg: dict) -> str:
    """Name of the experiment: its results folder (``results/<experiment>/best.pt``)."""
    return Path(path).resolve().parent.name or str(cfg.get("experiment", "experiment"))

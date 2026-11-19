"""Shared ``run_training`` used by the model-specific training scripts."""
from __future__ import annotations

from typing import Callable

from src.data import build_dataloaders
from src.models import build_model, count_parameters
from src.training.trainer import Trainer
from src.utils.config import experiment_dir, save_config
from src.utils.logger import get_logger
from src.utils.seed import get_device, set_seed


def run_training(cfg: dict, epoch_start_hook: Callable | None = None) -> dict:
    set_seed(int(cfg.get("seed", 42)))
    out = experiment_dir(cfg)
    log = get_logger("run", out / "train.log")
    save_config(cfg, out / "config.yaml")

    loaders, info = build_dataloaders(cfg, splits=("train", "val"))
    model = build_model(cfg, info["num_classes"], info["input_dim"])
    log.info(f"Model {cfg['model']['name']}: {count_parameters(model) / 1e6:.2f}M params "
             f"({count_parameters(model, trainable_only=True) / 1e6:.2f}M trainable) | "
             f"classes={info['num_classes']} | input_dim={info['input_dim']} | "
             f"train={len(loaders['train'].dataset)} val={len(loaders['val'].dataset) if 'val' in loaders else 0}")

    trainer = Trainer(model, loaders, cfg, out, get_device(cfg.get("device", "auto")),
                      extra_ckpt={"label_map": info["label_map"], "input_dim": info["input_dim"],
                                  "num_classes": info["num_classes"]},
                      epoch_start_hook=epoch_start_hook)
    result = trainer.fit()
    result["out_dir"] = str(out)
    return result

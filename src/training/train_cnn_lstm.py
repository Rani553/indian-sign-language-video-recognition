"""Train the Pipeline A model (CNN + LSTM).

    python -m src.training.train_cnn_lstm --config configs/cnn_lstm.yaml
    python -m src.training.train_cnn_lstm --config configs/cnn_lstm.yaml --tiny
    python -m src.training.train_cnn_lstm --config configs/cnn_lstm.yaml \
        --set model.backbone=resnet18 train.epochs=20 experiment=cnn_lstm_resnet18

The CNN starts frozen and is unfrozen at ``model.unfreeze_epoch`` (freeze -> fine-tune schedule).
"""
from __future__ import annotations

import argparse

from src.training.common import run_training
from src.utils.config import add_common_args, config_from_args


def make_unfreeze_hook(cfg: dict):
    """Epoch-start hook: unfreeze the CNN once ``model.unfreeze_epoch`` is reached.

    Called at the start of every epoch, also after a resume, so a run restarted from
    last.pt after the unfreeze epoch unfreezes the CNN again straight away.
    """
    unfreeze_epoch = cfg["model"].get("unfreeze_epoch")

    def hook(epoch: int, trainer) -> None:
        model = trainer.model
        if unfreeze_epoch is not None and epoch >= int(unfreeze_epoch) and model.backbone_frozen:
            model.unfreeze_backbone()
            trainer.log.info(f"ep {epoch}: CNN backbone unfrozen (fine-tuning)")

    return hook


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                     formatter_class=argparse.RawDescriptionHelpFormatter),
                             "configs/cnn_lstm.yaml")
    cfg = config_from_args(parser.parse_args())
    run_training(cfg, epoch_start_hook=make_unfreeze_hook(cfg))


if __name__ == "__main__":
    main()

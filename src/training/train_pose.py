"""Train a Pipeline B model (Pose-LSTM or Pose-Transformer) on MediaPipe keypoints.

    python -m src.training.train_pose --config configs/pose_lstm.yaml
    python -m src.training.train_pose --config configs/pose_transformer.yaml
    python -m src.training.train_pose --config configs/pose_lstm.yaml --tiny
    python -m src.training.train_pose --config configs/pose_lstm.yaml \
        --set keypoints.parts=[left_hand,right_hand] experiment=pose_lstm_hands_only

The model is chosen by the config (``model.name``). Every ablation is a ``--set`` override;
see scripts/run_ablations_pose.sh.
"""
from __future__ import annotations

import argparse

from src.training.common import run_training
from src.utils.config import add_common_args, config_from_args


def main() -> None:
    # --config, --set and --tiny come from the shared helper, so this script is used exactly
    # like the Pipeline A one. The default config is the LSTM; pass the Transformer one to switch.
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                     formatter_class=argparse.RawDescriptionHelpFormatter),
                             "configs/pose_lstm.yaml")
    cfg = config_from_args(parser.parse_args())
    # Guard against a common slip: pointing this script at a frames config (cnn_lstm.yaml). That
    # would fail later with a confusing "file not found" for keypoints, so stop early and say why.
    if cfg.get("input_type") != "keypoints":
        raise SystemExit(f"{cfg.get('experiment')}: input_type must be 'keypoints' for the pose models")
    # The shared training loop does everything else (data, model, optimiser, checkpoints), so no
    # pose-specific training code is needed here; the model and data are chosen by the config.
    run_training(cfg)


if __name__ == "__main__":
    main()

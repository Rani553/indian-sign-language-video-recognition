"""Shared test fixtures: a tiny synthetic dataset and two tiny trained models (built once per test run).

Nothing here needs the real INCLUDE data, a GPU, pretrained weights or MediaPipe.
"""
from __future__ import annotations

import pytest

from scripts.make_dummy_data import make_dummy_data, make_fake_keypoints
from src.utils.config import apply_overrides, load_config

# Settings that make a run tiny: 4 classes of 64x64 frames, 2 epochs, no workers, no mixed precision.
TINY = [
    "frames.stored_size=64", "frames.image_size=64", "frames.num_frames=4",
    "keypoints.num_frames=8",
    "train.epochs=2", "train.batch_size=4", "train.num_workers=0", "train.amp=false",
    "train.warmup_epochs=0", "train.early_stopping_patience=0",
]


@pytest.fixture(scope="session")
def dummy_root(tmp_path_factory):
    """Synthetic data in the same layout as the real ``processed/`` folder (frames and keypoints)."""
    root = tmp_path_factory.mktemp("isl_dummy")
    make_dummy_data(root, num_classes=4, videos_per_class=8, frames=24, size=64)
    make_fake_keypoints(root, frames=24)
    return root


@pytest.fixture(scope="session")
def make_cfg(dummy_root, tmp_path_factory):
    """``make_cfg("configs/pose_lstm.yaml", "experiment=x", ...)`` -> a tiny config for the dummy data."""
    results = tmp_path_factory.mktemp("isl_results")

    def build(config: str, *extra: str) -> dict:
        overrides = [f"paths.data_root={dummy_root}", f"paths.results_root={results}", *TINY, *extra]
        return apply_overrides(load_config(config), overrides)

    return build


@pytest.fixture(scope="session")
def trained(make_cfg):
    """Train one tiny model per pipeline once. Returns ``{"cnn": best.pt path, "pose": best.pt path}``."""
    from src.training.common import run_training
    from src.training.train_cnn_lstm import make_unfreeze_hook
    from pathlib import Path

    cnn = make_cfg("configs/cnn_lstm.yaml", "experiment=tiny_cnn", "model.backbone=mobilenet_v2",
                   "model.pretrained=false", "model.unfreeze_epoch=1")
    pose = make_cfg("configs/pose_lstm.yaml", "experiment=tiny_pose", "model.hidden_size=16", "model.num_layers=1")
    out_cnn = run_training(cnn, epoch_start_hook=make_unfreeze_hook(cnn))["out_dir"]
    out_pose = run_training(pose)["out_dir"]
    return {"cnn": Path(out_cnn) / "best.pt", "pose": Path(out_pose) / "best.pt",
            "cnn_cfg": cnn, "pose_cfg": pose}

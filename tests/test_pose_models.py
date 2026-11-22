"""Pipeline B tests: dataset, models and a short train -> resume loop (CPU, synthetic keypoints)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_dummy_data import make_dummy_data, make_fake_keypoints  # noqa: E402

from src.data import build_dataloaders  # noqa: E402
from src.features.keypoint_features import feature_dim  # noqa: E402
from src.models import build_model, count_parameters  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402
from src.utils.config import apply_overrides, load_config  # noqa: E402

NUM_CLASSES, T = 4, 16


def pose_cfg(config="configs/pose_lstm.yaml", *overrides) -> dict:
    return apply_overrides(load_config(config), list(overrides))


@pytest.fixture(scope="module")
def data_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("isl_pose")
    make_dummy_data(root, num_classes=NUM_CLASSES, videos_per_class=10, frames=24, size=16)
    make_fake_keypoints(root, frames=24)
    return root


def cfg_for(data_root: Path, config="configs/pose_lstm.yaml", *overrides) -> dict:
    c = pose_cfg(config, *overrides)
    c["paths"].update(data_root=str(data_root), results_root=str(data_root / "results"))
    c["keypoints"]["num_frames"] = T
    c["train"].update(batch_size=8, num_workers=0, epochs=2, amp=False, early_stopping_patience=0)
    return c


# ----------------------------------------------------------------- configs
def test_pose_configs_inherit_and_select_the_right_model():
    lstm, tr = pose_cfg(), pose_cfg("configs/pose_transformer.yaml")
    assert lstm["model"]["name"] == "pose_lstm" and tr["model"]["name"] == "pose_transformer"
    assert lstm["input_type"] == tr["input_type"] == "keypoints"
    assert tr["train"]["lr"] < lstm["train"]["lr"]            # transformer override applied
    assert tr["augment"] == lstm["augment"]                   # inherited unchanged


def test_both_configs_build_models_despite_inheriting_each_others_keys():
    F = feature_dim(pose_cfg())
    for path in ("configs/pose_lstm.yaml", "configs/pose_transformer.yaml"):
        model = build_model(pose_cfg(path), NUM_CLASSES, F)
        assert model(torch.randn(2, 32, F)).shape == (2, NUM_CLASSES)


def test_typo_in_model_key_is_not_swallowed():
    with pytest.raises(TypeError):
        build_model(pose_cfg("configs/pose_lstm.yaml", "model.hiden_size=8"), NUM_CLASSES, 100)


# ------------------------------------------------------------------ models
@pytest.mark.parametrize("pooling", ["attention", "mean", "max", "last"])
@pytest.mark.parametrize("bidirectional", [True, False])
def test_pose_lstm_shapes(pooling, bidirectional):
    c = pose_cfg("configs/pose_lstm.yaml", f"model.pooling={pooling}",
                 f"model.bidirectional={str(bidirectional).lower()}", "model.hidden_size=16")
    assert build_model(c, NUM_CLASSES, 50)(torch.randn(3, T, 50)).shape == (3, NUM_CLASSES)


def test_pose_lstm_attention_weights_sum_to_one():
    model = build_model(pose_cfg("configs/pose_lstm.yaml", "model.hidden_size=16"), NUM_CLASSES, 50)
    h = torch.randn(2, T, 32)
    w = torch.softmax(model.attn.score(h).squeeze(-1), dim=1)
    assert torch.allclose(w.sum(1), torch.ones(2), atol=1e-5)


def test_pose_transformer_shape_and_length_limit():
    c = pose_cfg("configs/pose_transformer.yaml", "model.d_model=32", "model.dim_feedforward=64", "model.max_len=20")
    model = build_model(c, NUM_CLASSES, 50)
    assert model(torch.randn(2, T, 50)).shape == (2, NUM_CLASSES)
    with pytest.raises(ValueError):
        model(torch.randn(2, 21, 50))


def test_invalid_model_arguments_are_rejected():
    with pytest.raises(ValueError):
        build_model(pose_cfg("configs/pose_lstm.yaml", "model.pooling=median"), NUM_CLASSES, 50)
    with pytest.raises(ValueError):
        build_model(pose_cfg("configs/pose_lstm.yaml"), NUM_CLASSES, None)         # needs input_dim
    with pytest.raises(ValueError):
        build_model(pose_cfg("configs/pose_transformer.yaml", "model.nhead=5"), NUM_CLASSES, 50)


def test_models_are_small_and_gradients_reach_every_layer():
    for path in ("configs/pose_lstm.yaml", "configs/pose_transformer.yaml"):
        model = build_model(pose_cfg(path), NUM_CLASSES, feature_dim(pose_cfg()))
        assert count_parameters(model) < 2_000_000
        model(torch.randn(2, 32, feature_dim(pose_cfg()))).sum().backward()
        assert all(p.grad is not None for p in model.parameters())


# ----------------------------------------------------------------- dataset
def test_dataloader_batch_contract(data_root):
    c = cfg_for(data_root)
    loaders, info = build_dataloaders(c)
    x, y = next(iter(loaders["train"]))
    assert x.shape == (8, T, feature_dim(c)) and x.dtype == torch.float32
    assert y.shape == (8,) and y.dtype == torch.long
    assert info["num_classes"] == NUM_CLASSES and info["input_dim"] == feature_dim(c)
    assert torch.isfinite(x).all()


@pytest.mark.parametrize("overrides", [["keypoints.parts=[left_hand,right_hand]"], ["keypoints.parts=[pose]"],
                                       ["keypoints.use_velocity=false"], ["augment.enabled=false"],
                                       ["augment.mirror=0.5"]])
def test_every_ablation_config_loads(data_root, overrides):
    c = cfg_for(data_root, "configs/pose_lstm.yaml", *overrides)
    loaders, info = build_dataloaders(c, splits=("train",))
    x, _ = next(iter(loaders["train"]))
    assert x.shape[-1] == info["input_dim"] == feature_dim(c)


def test_train_split_is_augmented_and_eval_split_is_not(data_root):
    c = cfg_for(data_root)
    loaders, _ = build_dataloaders(c)
    train_ds, val_ds = loaders["train"].dataset, loaders["val"].dataset
    assert not torch.equal(train_ds[0][0], train_ds[0][0])      # random augmentation
    assert torch.equal(val_ds[0][0], val_ds[0][0])              # deterministic


# --------------------------------------------------- train + resume loop
@pytest.mark.parametrize("config", ["configs/pose_lstm.yaml", "configs/pose_transformer.yaml"])
def test_train_learns_and_resumes(data_root, tmp_path, config):
    c = cfg_for(data_root, config, "model.hidden_size=32", "model.d_model=32", "model.dim_feedforward=64")
    loaders, info = build_dataloaders(c, splits=("train", "val"))
    out = tmp_path / "run"

    def trainer(epochs: int) -> Trainer:
        cc = {**c, "train": {**c["train"], "epochs": epochs, "lr": 3e-3}}
        return Trainer(build_model(cc, info["num_classes"], info["input_dim"]), loaders, cc, out,
                       torch.device("cpu"))

    first = trainer(1)
    first.fit()
    assert (out / "best.pt").exists() and (out / "last.pt").exists()
    t2 = trainer(8)                      # same folder -> must resume at epoch 1, not restart
    t2.fit()
    assert t2.start_epoch == 1
    rows = t2.csv.read()
    assert len(rows) == 8
    assert float(rows[-1]["train_loss"]) < float(rows[0]["train_loss"])   # it actually learns

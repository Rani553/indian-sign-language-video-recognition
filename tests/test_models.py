"""Pipeline A tests: model, transforms, dataset and a short train -> resume loop (CPU, synthetic data)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_dummy_data import make_dummy_data  # noqa: E402

from src.data import build_dataloaders  # noqa: E402
from src.data.transforms import ClipTransform, spatial_augment  # noqa: E402
from src.models import build_model, count_parameters  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402
from src.utils.config import load_config  # noqa: E402

NUM_CLASSES, T, SIZE = 4, 4, 32


def model_cfg(**kw) -> dict:
    return {"model": {"name": "cnn_lstm", "backbone": "mobilenet_v2", "pretrained": False,
                      "hidden_size": 16, **kw}}


@pytest.fixture(scope="module")
def cfg(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("isl")
    make_dummy_data(root, num_classes=NUM_CLASSES, videos_per_class=6, frames=8, size=SIZE)
    c = load_config("configs/data.yaml")
    c["paths"].update(data_root=str(root), results_root=str(root / "results"))
    c["frames"].update(num_frames=T, image_size=SIZE)
    c["train"] = {"batch_size": 4, "num_workers": 0, "lr": 1e-3, "epochs": 2, "amp": False}
    return c


# ----------------------------------------------------------------- model
@pytest.mark.parametrize("backbone", ["mobilenet_v2", "resnet18", "efficientnet_b0"])
def test_backbones_output_shape(backbone):
    model = build_model(model_cfg(backbone=backbone), NUM_CLASSES)
    assert model(torch.randn(2, T, 3, SIZE, SIZE)).shape == (2, NUM_CLASSES)


@pytest.mark.parametrize("pooling", ["last", "mean", "max"])
@pytest.mark.parametrize("bidirectional", [True, False])
def test_pooling_and_direction(pooling, bidirectional):
    model = build_model(model_cfg(pooling=pooling, bidirectional=bidirectional, num_layers=2), NUM_CLASSES)
    assert model(torch.randn(2, T, 3, SIZE, SIZE)).shape == (2, NUM_CLASSES)


def test_unknown_names_are_rejected():
    with pytest.raises(ValueError):
        build_model({"model": {"name": "nope"}}, NUM_CLASSES)
    with pytest.raises(ValueError):
        build_model(model_cfg(backbone="vgg"), NUM_CLASSES)
    with pytest.raises(ValueError):
        build_model(model_cfg(pooling="median"), NUM_CLASSES)


def test_freeze_unfreeze_schedule():
    model = build_model(model_cfg(freeze_backbone=True), NUM_CLASSES)
    frozen = count_parameters(model, trainable_only=True)
    model.train()
    assert not model.cnn.training                     # frozen CNN stays in eval (BatchNorm fixed)
    assert frozen < count_parameters(model)
    model.unfreeze_backbone()
    assert count_parameters(model, trainable_only=True) == count_parameters(model)
    assert model.cnn.training


def test_frozen_backbone_gets_no_gradient_but_head_does():
    model = build_model(model_cfg(freeze_backbone=True), NUM_CLASSES)
    model(torch.randn(2, T, 3, SIZE, SIZE)).sum().backward()
    assert all(p.grad is None for p in model.cnn.parameters())
    assert model.head[1].weight.grad is not None


def test_param_groups_give_backbone_smaller_lr():
    groups = build_model(model_cfg(), NUM_CLASSES).param_groups(1e-3, None)
    assert groups[0]["lr"] < groups[1]["lr"]
    assert groups[0]["lr"] == pytest.approx(1e-4)


# ------------------------------------------------------------ transforms
def test_spatial_augment_is_identical_for_every_frame():
    clip = torch.rand(1, 3, 40, 40).repeat(6, 1, 1, 1)   # six copies of one frame
    out = spatial_augment(clip, 32, rotation=15.0)
    assert out.shape == (6, 3, 32, 32)
    assert all(torch.allclose(out[0], out[i]) for i in range(1, 6))


def test_clip_transform_eval_is_deterministic_and_train_is_not():
    clip = torch.randint(0, 255, (T, 48, 48, 3), dtype=torch.uint8).numpy()
    ev = ClipTransform(SIZE, train=False)
    assert torch.equal(ev(clip), ev(clip))
    tr = ClipTransform(SIZE, train=True)
    assert tr(clip).shape == (T, 3, SIZE, SIZE)
    assert not torch.equal(tr(clip), tr(clip))


# --------------------------------------------------------------- dataset
def test_dataloader_batch_contract(cfg):
    loaders, info = build_dataloaders(cfg)
    x, y = next(iter(loaders["train"]))
    assert x.shape == (4, T, 3, SIZE, SIZE) and x.dtype == torch.float32
    assert y.shape == (4,) and y.dtype == torch.long
    assert info["num_classes"] == NUM_CLASSES


# ---------------------------------------------------- train + resume loop
def test_train_eval_and_resume(cfg, tmp_path):
    loaders, info = build_dataloaders(cfg, splits=("train", "val"))
    out = tmp_path / "run"

    def trainer(epochs: int) -> Trainer:
        c = {**cfg, "train": {**cfg["train"], "epochs": epochs}}
        model = build_model({**c, **model_cfg(freeze_backbone=True)}, info["num_classes"])
        return Trainer(model, loaders, c, out, torch.device("cpu"))

    result = trainer(1).fit()
    assert 0.0 <= result["best_val_acc"] <= 1.0
    assert (out / "best.pt").exists() and (out / "last.pt").exists() and (out / "history.csv").exists()

    t2 = trainer(2)                       # same folder, one more epoch -> must resume, not restart
    t2.fit()
    assert t2.start_epoch == 1
    assert len(t2.csv.read()) == 2

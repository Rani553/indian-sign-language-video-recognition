"""Data side: splits, the batch contract between the data and the models, and the EDA helpers."""
import pandas as pd
import pytest
import torch

from src.data import build_dataloaders, build_dataset
from src.data.eda import class_balance, leak_check, mvi_number
from src.data.split import stratified_split
from src.utils.config import resolve_paths


def _splits(cfg) -> dict[str, pd.DataFrame]:
    d = resolve_paths(cfg)["splits_dir"]
    return {s: pd.read_csv(d / f"{s}.csv") for s in ("train", "val", "test")}


def test_splits_are_disjoint_and_cover_every_class(make_cfg):
    s = _splits(make_cfg("configs/cnn_lstm.yaml"))
    ids = [set(df["video_id"]) for df in s.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    for df in s.values():
        assert df["label_idx"].nunique() == 4          # stratified: every class in every split


def test_stratified_split_is_reproducible():
    meta = pd.DataFrame({"video_id": [f"v{i}" for i in range(40)], "label": ["a", "b"] * 20,
                         "label_idx": [0, 1] * 20})
    a = stratified_split(meta, 0.2, 0.2, seed=1)
    b = stratified_split(meta, 0.2, 0.2, seed=1)
    assert all(a[k]["video_id"].tolist() == b[k]["video_id"].tolist() for k in a)


def test_frame_batch_contract(make_cfg):
    cfg = make_cfg("configs/cnn_lstm.yaml")
    loaders, info = build_dataloaders(cfg)
    x, y = next(iter(loaders["train"]))
    assert x.shape == (4, 4, 3, 64, 64) and x.dtype == torch.float32      # (B, T, 3, H, W)
    assert y.shape == (4,) and y.dtype == torch.long
    assert info["num_classes"] == 4 and info["input_dim"] is None


def test_keypoint_batch_contract(make_cfg):
    cfg = make_cfg("configs/pose_lstm.yaml")
    loaders, info = build_dataloaders(cfg)
    x, y = next(iter(loaders["train"]))
    assert x.shape[0] == 4 and x.shape[1] == 8 and x.shape[2] == info["input_dim"]   # (B, T, F)
    assert y.dtype == torch.long


@pytest.mark.parametrize("config", ["configs/cnn_lstm.yaml", "configs/pose_lstm.yaml"])
def test_evaluation_data_is_deterministic(make_cfg, config):
    """Evaluation must give the same input every time (no augmentation, fixed frame sampling)."""
    ds = build_dataset(make_cfg(config), "test", train=False)
    a, b = ds[0][0], ds[0][0]
    assert torch.equal(a, b)


def test_mvi_number():
    assert mvi_number("adjectives_dry_mvi_5167") == 5167
    assert mvi_number("no_number_here") is None


def test_class_balance_counts():
    df = pd.DataFrame({"video_id": list("abcdef"), "label": ["x", "x", "x", "y", "y", "y"],
                       "split": ["train", "train", "test", "train", "val", "test"]})
    t = class_balance(df).set_index("label")
    assert t.loc["x", "train"] == 2 and t.loc["y", "val"] == 1 and t.loc["x", "total"] == 3


def test_leak_check_finds_neighbouring_takes():
    df = pd.DataFrame({"video_id": ["c_mvi_100", "c_mvi_101", "c_mvi_500"], "label": ["c"] * 3,
                       "split": ["train", "test", "test"]})
    r = leak_check(df, gap=2)
    assert r["test_videos"] == 2 and r["with_near_duplicate_in_train"] == 1   # 101 has 100; 500 has none

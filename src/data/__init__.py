"""Data loading entry point shared by all training / evaluation scripts.

``build_dataloaders(cfg)`` returns ``(loaders, info)`` where
``info = {"num_classes", "input_dim", "label_map", "class_names"}``.

Batch contract (agreed by the team):
    frames    -> x: (B, T, 3, H, W) float, y: (B,) long
    keypoints -> x: (B, T, F) float,       y: (B,) long
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.data.split import load_label_map, read_split
from src.utils.config import resolve_paths
from src.utils.seed import seed_worker


def build_dataset(cfg: dict, split: str, train: bool | None = None):
    paths = resolve_paths(cfg)
    train = (split == "train") if train is None else train
    df = read_split(paths["splits_dir"], split, cfg)
    input_type = cfg.get("input_type", "frames")
    aug = cfg.get("augment", {})
    if input_type == "frames":
        from src.data.frame_dataset import FrameDataset
        f = cfg["frames"]
        return FrameDataset(df, paths["frames_dir"], f["num_frames"], f["image_size"], train, aug)
    if input_type == "keypoints":
        from src.data.keypoint_dataset import KeypointDataset
        return KeypointDataset(df, paths["keypoints_dir"], cfg["keypoints"], train, aug)
    raise ValueError(f"Unknown input_type: {input_type}")


def build_dataloaders(cfg: dict, splits=("train", "val", "test")):
    paths = resolve_paths(cfg)
    label_map = load_label_map(paths["splits_dir"])
    tcfg = cfg.get("train", {})
    nw = int(tcfg.get("num_workers", 2))
    g = torch.Generator()
    g.manual_seed(int(cfg.get("seed", 42)))
    loaders, input_dim = {}, None
    for split in splits:
        ds = build_dataset(cfg, split)
        if len(ds) == 0:
            continue
        if input_dim is None and hasattr(ds, "feature_dim"):
            input_dim = ds.feature_dim
        loaders[split] = DataLoader(
            ds, batch_size=int(tcfg.get("batch_size", 8)), shuffle=(split == "train"),
            num_workers=nw, pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train" and len(ds) > int(tcfg.get("batch_size", 8))),
            worker_init_fn=seed_worker, generator=g, persistent_workers=nw > 0)
    class_names = [c for c, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    info = {"num_classes": len(label_map), "input_dim": input_dim,
            "label_map": label_map, "class_names": class_names}
    return loaders, info

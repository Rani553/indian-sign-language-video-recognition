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
    """Create the Dataset for one split ("train", "val" or "test").

    ``cfg["input_type"]`` decides which pipeline's dataset is used:
    "frames" (Pipeline A, Member 2) or "keypoints" (Pipeline B, Member 3).
    ``train`` turns augmentation on; by default only the train split is augmented.
    """
    paths = resolve_paths(cfg)
    train = (split == "train") if train is None else train
    df = read_split(paths["splits_dir"], split, cfg)  # table of videos in this split
    input_type = cfg.get("input_type", "frames")
    aug = cfg.get("augment", {})
    # The dataset classes are imported here (not at the top of the file), so this
    # file works even when only one of the two pipelines is installed.
    if input_type == "frames":
        from src.data.frame_dataset import FrameDataset
        f = cfg["frames"]
        return FrameDataset(df, paths["frames_dir"], f["num_frames"], f["image_size"], train, aug)
    if input_type == "keypoints":
        from src.data.keypoint_dataset import KeypointDataset
        return KeypointDataset(df, paths["keypoints_dir"], cfg["keypoints"], train, aug)
    raise ValueError(f"Unknown input_type: {input_type}")


def build_dataloaders(cfg: dict, splits=("train", "val", "test")):
    """Create one DataLoader per split, plus ``info`` about the classes and input size."""
    paths = resolve_paths(cfg)
    label_map = load_label_map(paths["splits_dir"])
    tcfg = cfg.get("train", {})
    nw = int(tcfg.get("num_workers", 2))  # parallel processes that load data
    # Fixed random generator -> the training data is shuffled the same way every run.
    g = torch.Generator()
    g.manual_seed(int(cfg.get("seed", 42)))
    loaders, input_dim = {}, None
    for split in splits:
        ds = build_dataset(cfg, split)
        if len(ds) == 0:
            continue  # e.g. no val videos in tiny mode -> skip this split
        # Keypoint datasets know their feature size F; the model needs it.
        if input_dim is None and hasattr(ds, "feature_dim"):
            input_dim = ds.feature_dim
        loaders[split] = DataLoader(
            ds, batch_size=int(tcfg.get("batch_size", 8)), shuffle=(split == "train"),
            num_workers=nw, pin_memory=torch.cuda.is_available(),  # pin_memory = faster copy to GPU
            # Drop the last small batch in training (a batch of 1 can upset BatchNorm),
            # but only when there is more than one batch.
            drop_last=(split == "train" and len(ds) > int(tcfg.get("batch_size", 8))),
            worker_init_fn=seed_worker, generator=g, persistent_workers=nw > 0)
    # Class names in index order: class_names[i] is the word for label i.
    class_names = [c for c, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    info = {"num_classes": len(label_map), "input_dim": input_dim,
            "label_map": label_map, "class_names": class_names}
    return loaders, info

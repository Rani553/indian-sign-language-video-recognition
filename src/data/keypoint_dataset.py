"""Pipeline B dataset: cached keypoints of one video -> (features, label).

``extract_keypoints.py`` stores raw landmarks as ``<keypoints_dir>/<video_id>.npz``;
``keypoint_features.py`` turns them into the feature matrix.

    __getitem__ -> x: (T, F) float32 with T = keypoints.num_frames, F = feature_dim(cfg); y: long scalar

The slow, deterministic part (gap filling + normalisation) is done once per video and kept in
memory; only the random augmentation and the cheap feature step run on every access.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features.keypoint_features import features_from_prepared, feature_dim, prepare_clip
from src.features.keypoint_io import load_keypoints


class KeypointDataset(Dataset):
    def __init__(self, df: pd.DataFrame, keypoints_dir: str | Path, kp_cfg: dict,
                 train: bool = False, aug: dict | None = None):
        self.df = df.reset_index(drop=True)
        self.keypoints_dir = Path(keypoints_dir)
        self.train = train
        # Augmentation settings come from the top-level ``augment:`` section (like Pipeline A).
        self.cfg = {"keypoints": {**kp_cfg, "augment": {**(kp_cfg.get("augment") or {}), **(aug or {})}}}
        self.feature_dim = feature_dim(self.cfg)   # the model needs it; see build_dataloaders
        self._cache: dict[str, dict] = {}
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.df)

    def _prepared(self, video_id: str) -> dict:
        if video_id not in self._cache:
            path = self.keypoints_dir / f"{video_id}.npz"
            if not path.exists():
                raise FileNotFoundError(f"{path} not found. Run `python -m src.features.extract_keypoints` first.")
            self._cache[video_id] = prepare_clip(load_keypoints(path), self.cfg)
        return self._cache[video_id]

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        if self.train and self._rng is None:
            # torch.initial_seed() differs per worker but is fixed by the DataLoader generator,
            # so augmentation is reproducible for a given config.seed.
            self._rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))
        x = features_from_prepared(self._prepared(row["video_id"]), self.cfg, self.train, self._rng)
        return torch.from_numpy(x), torch.tensor(int(row["label_idx"]))

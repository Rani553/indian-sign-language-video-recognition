"""Pipeline A dataset: cached frames of one video -> (clip, label).

``extract_frames.py`` stores every video once as ``<frames_dir>/<video_id>.npy`` with shape
``(stored_frames, S, S, 3)`` uint8. Training never touches the raw videos.

    __getitem__ -> x: (T, 3, H, W) float (ImageNet-normalised), y: long scalar
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.transforms import ClipTransform
from src.utils.video import random_segment_indices, uniform_indices


class FrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, frames_dir: str | Path, num_frames: int, image_size: int,
                 train: bool = False, aug: dict | None = None):
        self.df = df.reset_index(drop=True)
        self.frames_dir = Path(frames_dir)
        self.num_frames = int(num_frames)
        self.train = train
        self.transform = ClipTransform(image_size, train, aug)

    def __len__(self) -> int:
        return len(self.df)

    def _sample_indices(self, available: int) -> np.ndarray:
        """Training: one random frame per time segment (temporal augmentation).
        Evaluation: evenly spaced frames, so results are repeatable."""
        if self.train:
            return random_segment_indices(available, self.num_frames)
        return uniform_indices(available, self.num_frames)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        path = self.frames_dir / f"{row['video_id']}.npy"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `python -m src.features.extract_frames` first.")
        frames = np.load(path, mmap_mode="r")           # read only the frames we pick
        clip = frames[self._sample_indices(len(frames))]
        return self.transform(np.ascontiguousarray(clip)), torch.tensor(int(row["label_idx"]))

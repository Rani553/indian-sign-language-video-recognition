"""Raw video -> top-k signs, for the demo and the end-to-end speed benchmark.

    p = Predictor("results/cnn_lstm_mobilenet_v2/best.pt")
    top = p.predict("clip.mov", k=5)        # [("thankyou", 0.91), ("good", 0.03), ...]
    top, timing = p.predict("clip.mov", k=5, return_timing=True)

A video goes through EXACTLY the preprocessing of training, so the model sees the kind of input
it was trained on:

    1. ``sample_and_resize``: ``frames.stored_frames`` evenly spaced frames, letterboxed to
       ``frames.stored_size`` (the same as src/features/extract_frames.py, the cache of the training data)
    2. frames models:   pick ``frames.num_frames`` of them evenly + ImageNet normalisation (no augmentation)
       keypoint models: MediaPipe Holistic -> ``keypoint_features`` (needs ``requirements-pose.txt``)
    3. the model, then softmax

``timing`` splits the time into the three stages, so the benchmark can show where it goes.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from src.evaluation.checkpoint import class_names_from, experiment_name, load_checkpoint
from src.utils.seed import get_device
from src.utils.video import sample_and_resize, uniform_indices


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


class Predictor:
    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        self.device = get_device(device)
        self.model, self.cfg, self.ckpt = load_checkpoint(checkpoint, self.device)
        self.name = experiment_name(checkpoint, self.cfg)
        self.input_type = self.cfg.get("input_type", "frames")
        self.class_names = class_names_from(self.ckpt["label_map"])
        f = self.cfg["frames"]
        self.stored_frames, self.stored_size = int(f["stored_frames"]), int(f["stored_size"])
        self.resize_mode = f.get("resize_mode", "pad")
        self._extractor = None   # MediaPipe is loaded on first use (slow, and only the pose models need it)
        self._transform = None
        if self.input_type == "frames":
            from src.data.transforms import ClipTransform
            self._transform = ClipTransform(int(f["image_size"]), train=False)

    # ------------------------------------------------------------------ #
    def _keypoint_extractor(self):
        if self._extractor is None:
            try:
                from src.features.extract_keypoints import KeypointExtractor
                self._extractor = KeypointExtractor(self.cfg)
            except (ImportError, AttributeError) as e:
                raise RuntimeError(
                    "The pose models need MediaPipe 0.10.21 to read a raw video "
                    "(pip install -r requirements-pose.txt, Python 3.12 recommended). "
                    f"Original error: {e}") from e
        return self._extractor

    def read_frames(self, video_path: str | Path) -> np.ndarray:
        """Stage 1: decode and resize. Returns ``(stored_frames, S, S, 3)`` uint8."""
        return sample_and_resize(video_path, self.stored_frames, self.stored_size, self.resize_mode)

    def features(self, frames: np.ndarray) -> torch.Tensor:
        """Stage 2: model input for one clip, shape ``(1, T, 3, H, W)`` or ``(1, T, F)``."""
        if self.input_type == "frames":
            idx = uniform_indices(len(frames), int(self.cfg["frames"]["num_frames"]))
            return self._transform(np.ascontiguousarray(frames[idx])).unsqueeze(0)
        from src.features.keypoint_features import keypoint_features
        raw = self._keypoint_extractor().from_frames(frames)
        return torch.from_numpy(keypoint_features(raw, self.cfg)).unsqueeze(0)

    @torch.no_grad()
    def predict_tensor(self, x: torch.Tensor) -> np.ndarray:
        """Stage 3: model + softmax. ``x`` has a batch dimension; returns ``(B, C)`` probabilities."""
        return torch.softmax(self.model(x.to(self.device)).float(), dim=1).cpu().numpy()

    # ------------------------------------------------------------------ #
    def predict(self, video_path: str | Path, k: int = 5, return_timing: bool = False):
        """Top-``k`` predictions for one video file: list of ``(class name, probability)``."""
        t0 = time.perf_counter()
        frames = self.read_frames(video_path)
        t1 = time.perf_counter()
        x = self.features(frames)
        _sync(self.device)
        t2 = time.perf_counter()
        probs = self.predict_tensor(x)[0]
        _sync(self.device)
        t3 = time.perf_counter()
        order = np.argsort(-probs)[:k]
        top = [(self.class_names[i], float(probs[i])) for i in order]
        if not return_timing:
            return top
        return top, {"decode_resize_ms": (t1 - t0) * 1e3, "features_ms": (t2 - t1) * 1e3,
                     "model_ms": (t3 - t2) * 1e3, "total_ms": (t3 - t0) * 1e3}

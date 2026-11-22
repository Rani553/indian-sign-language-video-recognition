"""The on-disk format of the keypoint cache, plus a synthetic generator for tests.

One file per video: ``<keypoints_dir>/<video_id>.npz`` with RAW MediaPipe Holistic landmarks
(no normalisation, no gap filling - that is done later in ``keypoint_features.py``, so the
expensive extraction never has to be repeated when a feature idea changes).

Arrays (T = number of cached frames, 24 for INCLUDE-50), float32:
    pose        (T, 33, 4)   x, y, z, visibility   (x, y in [0, 1] of the image; y grows downwards)
    left_hand   (T, 21, 3)   x, y, z               (the signer's left hand)
    right_hand  (T, 21, 3)   x, y, z
A body part that MediaPipe did not find in a frame is filled with NaN (never 0, because
0 is a valid coordinate). Use ``part_present`` to get a boolean (T,) mask.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# The single source of truth for the format: part name -> (number of landmarks, values per landmark).
# Pose has a 4th value (visibility); the hands do not. Writer, reader and validator all use this
# dict, so the format cannot silently drift apart between them.
PARTS = {"pose": (33, 4), "left_hand": (21, 3), "right_hand": (21, 3)}


def save_keypoints(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write one video's keypoints atomically (a disconnect never leaves a half-written file)."""
    # Check the shapes before writing: a wrong shape found here is a clear error message,
    # the same mistake found later would be a confusing crash in the middle of training.
    for name, shape in PARTS.items():
        a = arrays[name]
        if a.ndim != 3 or a.shape[1:] != shape:
            raise ValueError(f"{name} must have shape (T, {shape[0]}, {shape[1]}), got {a.shape}")
    # Write to a temporary name first and rename at the end. A rename is atomic, so a Colab
    # disconnect in the middle of np.savez never leaves a broken .npz that looks finished
    # (the extraction script skips existing files, so a broken one would stay broken).
    tmp = path.with_name(path.stem + ".tmp.npz")
    # float32 is what MediaPipe itself returns, and it halves the file size compared with float64.
    np.savez_compressed(tmp, **{k: np.asarray(arrays[k], dtype=np.float32) for k in PARTS})
    tmp.replace(path)


def load_keypoints(path: Path) -> dict[str, np.ndarray]:
    """Read one video's keypoints: dict part name -> float32 array (NaN = not detected)."""
    # The ``with`` closes the file at once. np.load is lazy, so reading every array inside
    # the block also makes the result safe to use after the file is closed.
    with np.load(path) as z:
        return {k: z[k] for k in PARTS}


def part_present(a: np.ndarray) -> np.ndarray:
    """Boolean (T,) mask: True where the part was detected (its coordinates are not NaN)."""
    # A part counts as "found" only if ALL its numbers are real. MediaPipe returns either a whole
    # hand or nothing, so any NaN means the whole frame of that part is missing.
    return ~np.isnan(a.reshape(a.shape[0], -1)).any(axis=1)


def fake_keypoints(class_idx: int, num_classes: int, frames: int, rng: np.random.Generator,
                   hand_detect_prob: float = 0.75) -> dict[str, np.ndarray]:
    """Synthetic keypoints so the Pipeline B code runs without MediaPipe or the real data.

    A fixed skeleton stands still; the hands move along a class-specific circle, so a model can
    really learn the class. Each hand is randomly "not detected" (NaN) in about a quarter of the
    frames, which mimics what MediaPipe does on the real frames and exercises gap filling.
    """
    t = np.linspace(0.0, 1.0, frames)[:, None]   # time from 0 to 1, shape (frames, 1)
    # Body: 33 points spread from top to bottom of the image, with a little noise.
    # Visibility (last value) is 1 = "clearly visible".
    pose = np.zeros((frames, 33, 4), np.float32)
    pose[:, :, 0] = 0.5 + rng.normal(0, 0.01, (frames, 33)) + np.linspace(-0.12, 0.12, 33)[None] * 0.5
    pose[:, :, 1] = np.linspace(0.15, 0.9, 33)[None] + rng.normal(0, 0.01, (frames, 33))
    pose[:, :, 2] = rng.normal(0, 0.02, (frames, 33))
    pose[:, :, 3] = 1.0

    # The class decides the circle: its start angle (phase) and its size (radius). Different
    # classes therefore have different hand paths, which is exactly the signal a model can learn.
    phase = 2 * np.pi * class_idx / max(1, num_classes)
    radius = 0.08 + 0.12 * (class_idx % 4) / 3
    out = {"pose": pose}
    for side, sign in (("left_hand", -1.0), ("right_hand", 1.0)):
        # sign puts the left hand left of the body and the right hand right of it, and makes
        # the two hands turn in opposite directions (like two hands moving as a pair).
        cx = 0.5 + sign * 0.15 + radius * np.cos(2 * np.pi * t + phase * (1 if sign > 0 else -1))
        cy = 0.45 + radius * np.sin(2 * np.pi * t + phase)
        # A fixed per-clip hand shape (21 points on a slightly tilted line) plus tiny jitter.
        shape = rng.normal(0, 0.02, (1, 21, 3)) + np.linspace(0, 0.06, 21)[None, :, None]
        hand = np.concatenate([cx, cy, np.zeros_like(cx)], axis=1)[:, None, :] + shape
        hand = hand + rng.normal(0, 0.003, hand.shape)
        # Drop random frames to NaN, like MediaPipe missing a hand. Without this the tests
        # would never exercise the gap-filling code.
        hand[rng.random(frames) > hand_detect_prob] = np.nan
        out[side] = hand.astype(np.float32)
    return out

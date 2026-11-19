"""Video reading and frame-sampling utilities shared by both pipelines."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".webm", ".m4v"}


def video_info(path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    info["duration_s"] = info["frames"] / info["fps"] if info["fps"] > 0 else 0.0
    return info


def read_video(path: str | Path, stride: int = 1, max_side: int | None = None) -> list[np.ndarray]:
    """Read all frames (RGB, uint8). ``stride`` keeps every n-th frame.
    ``max_side`` downsizes large frames early to save memory (aspect kept)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            if max_side is not None:
                h, w = frame.shape[:2]
                scale = max_side / max(h, w)
                if scale < 1.0:
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                       interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    if not frames:
        raise IOError(f"No frames decoded from video: {path}")
    return frames


def uniform_indices(num_available: int, num_samples: int) -> np.ndarray:
    """Evenly spaced indices; repeats frames when the clip is too short."""
    if num_available <= 0:
        raise ValueError("Video has no frames")
    return np.linspace(0, num_available - 1, num_samples).round().astype(int)


def random_segment_indices(num_available: int, num_samples: int,
                           rng: np.random.Generator | None = None) -> np.ndarray:
    """TSN-style sampling: split the clip into ``num_samples`` equal segments and
    pick one random frame from each (temporal augmentation for training)."""
    rng = rng or np.random.default_rng()
    if num_available < num_samples:
        return uniform_indices(num_available, num_samples)
    edges = np.linspace(0, num_available, num_samples + 1)
    starts, ends = edges[:-1].astype(int), np.maximum(edges[1:].astype(int), edges[:-1].astype(int) + 1)
    return np.array([rng.integers(s, e) for s, e in zip(starts, ends)])


def resize_frame(frame: np.ndarray, size: int, mode: str = "pad") -> np.ndarray:
    """Make a square ``size x size`` frame.

    ``pad``: letterbox (keeps the signer's arms; recommended for INCLUDE's 16:9 videos).
    ``center_crop``: crop the central square (higher resolution on the signer).
    """
    h, w = frame.shape[:2]
    if mode == "center_crop":
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        frame = frame[y0:y0 + side, x0:x0 + side]
        return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    if mode == "pad":
        scale = size / max(h, w)
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((size, size, 3), dtype=frame.dtype)
        y0, x0 = (size - nh) // 2, (size - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas
    if mode == "stretch":
        return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    raise ValueError(f"Unknown resize mode: {mode}")


def sample_and_resize(path: str | Path, num_frames: int, size: int, mode: str = "pad") -> np.ndarray:
    """Read a video and return ``(num_frames, size, size, 3)`` uint8."""
    frames = read_video(path)
    idx = uniform_indices(len(frames), num_frames)
    return np.stack([resize_frame(frames[i], size, mode) for i in idx])

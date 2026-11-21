"""Clip-consistent augmentation for Pipeline A.

Every frame of a clip gets the SAME random crop, rotation and colour change. Augmenting
frames independently would make the signer jitter from frame to frame, which destroys the
motion the LSTM has to read. Horizontal flip is off by default: flipping swaps the dominant
hand and can change the meaning of a sign.

Input:  clip (T, 3, H, W) float in [0, 1].   Output: (T, 3, out_size, out_size).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Defaults; any key can be overridden by the ``augment:`` section of the config.
DEFAULT_AUG = {
    "enabled": True,
    "scale": [0.8, 1.0],   # visible fraction of the frame side (random crop + resize back)
    "rotation": 10.0,      # max rotation in degrees (+/-)
    "brightness": 0.2,     # factor drawn from [1-b, 1+b]
    "contrast": 0.2,
    "saturation": 0.2,
    "hflip": False,
}


def normalize(clip: torch.Tensor) -> torch.Tensor:
    """ImageNet mean/std normalisation (what the pretrained CNN expects)."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=clip.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=clip.dtype).view(1, 3, 1, 1)
    return (clip - mean) / std


def _uniform(lo: float, hi: float) -> float:
    return float(torch.empty(1).uniform_(lo, hi))


def spatial_augment(clip: torch.Tensor, out_size: int, scale=(0.8, 1.0), rotation: float = 10.0,
                    hflip: bool = False) -> torch.Tensor:
    """One random crop + rotation (+ optional flip) applied to all frames, output ``out_size``.

    Done with a single affine grid, so crop, rotate and resize cost one interpolation.
    """
    s = _uniform(float(scale[0]), float(scale[1]))          # side of the crop, relative to the frame
    angle = math.radians(_uniform(-rotation, rotation))
    # The crop may move inside the frame by (1 - s) in each direction (grid coords span [-1, 1]).
    tx, ty = _uniform(-(1 - s), 1 - s), _uniform(-(1 - s), 1 - s)
    cos, sin = math.cos(angle) * s, math.sin(angle) * s
    flip = -1.0 if hflip else 1.0
    theta = torch.tensor([[cos * flip, -sin, tx], [sin * flip, cos, ty]], dtype=clip.dtype)
    theta = theta.unsqueeze(0).expand(clip.shape[0], -1, -1)  # same transform for every frame
    grid = F.affine_grid(theta, (clip.shape[0], 3, out_size, out_size), align_corners=False)
    return F.grid_sample(clip, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def color_jitter(clip: torch.Tensor, brightness: float = 0.2, contrast: float = 0.2,
                 saturation: float = 0.2) -> torch.Tensor:
    """Brightness, contrast and saturation change with one random factor per clip."""
    if brightness:
        clip = clip * _uniform(1 - brightness, 1 + brightness)
    if contrast:
        mean = clip.mean(dim=(1, 2, 3), keepdim=True)         # per-frame mean brightness
        clip = (clip - mean) * _uniform(1 - contrast, 1 + contrast) + mean
    if saturation:
        gray = (0.299 * clip[:, 0] + 0.587 * clip[:, 1] + 0.114 * clip[:, 2]).unsqueeze(1)
        clip = (clip - gray) * _uniform(1 - saturation, 1 + saturation) + gray
    return clip.clamp(0.0, 1.0)


class ClipTransform:
    """Turns a uint8 clip (T, H, W, 3) into a normalised float tensor (T, 3, size, size).

    ``train=True`` adds the random augmentation from ``aug``; otherwise only resize + normalise.
    """

    def __init__(self, image_size: int, train: bool = False, aug: dict | None = None):
        self.size = int(image_size)
        self.train = train
        self.aug = {**DEFAULT_AUG, **(aug or {})}

    def __call__(self, clip) -> torch.Tensor:
        x = torch.as_tensor(clip).permute(0, 3, 1, 2).float() / 255.0   # (T, 3, H, W) in [0, 1]
        if self.train and self.aug["enabled"]:
            x = spatial_augment(x, self.size, self.aug["scale"], float(self.aug["rotation"]),
                                bool(self.aug["hflip"]))
            x = color_jitter(x, float(self.aug["brightness"]), float(self.aug["contrast"]),
                             float(self.aug["saturation"]))
        elif x.shape[-1] != self.size or x.shape[-2] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return normalize(x)

"""Pipeline B model 2: Pose-Transformer = keypoint features -> Transformer encoder with a CLS token -> classifier.

Interface (agreed by the team):
    forward(x)  x: (B, T, F) float  ->  logits (B, num_classes)      F = feature_dim(cfg)

A learnable CLS token is put in front of the frame sequence; after the encoder its output
summarises the whole clip and goes through a linear classifier (as in BERT / ViT).
Learned position embeddings tell the encoder the order of the frames.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PoseTransformer(nn.Module):
    def __init__(self, num_classes: int, input_dim: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 3, dim_feedforward: int = 256, dropout: float = 0.2, max_len: int = 64):
        super().__init__()
        if input_dim is None or input_dim <= 0:
            raise ValueError("PoseTransformer needs input_dim (the keypoint feature size, from the dataset)")
        # Each attention head gets d_model / nhead numbers, so this must divide exactly.
        # Checking here gives a readable message instead of a shape error inside PyTorch.
        if d_model % nhead:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.max_len = max_len
        # LayerNorm first: the features have very different scales (coordinates, velocities, flags).
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, d_model)     # bring every frame to the model's width
        # The CLS token is a learned "summary slot". It has no frame of its own, but through
        # attention it can look at all frames, so its final value describes the whole clip.
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        # A Transformer has no sense of order by itself (attention ignores position), so a learned
        # position vector is added to each frame. Learned (not sine) is a simple choice that suits
        # our short clips of fixed length. The +1 is for the CLS token.
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, d_model))     # CLS + up to max_len frames
        # Small random start values (std 0.02, the usual choice in BERT / ViT), so the first
        # training steps are not dominated by a large random token or position.
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(dropout)
        # norm_first=True ("pre-norm"): normalise before attention, which tends to train more
        # stably than the original post-norm design, especially without a long warm-up or much data.
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        # enable_nested_tensor=False: we never pad clips, and with norm_first=True PyTorch cannot
        # use that speed-up anyway (it would only print a warning).
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d_model)    # pre-norm blocks leave the last output un-normalised
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        # The position table only has max_len entries; a longer clip would index outside it.
        if T > self.max_len:
            raise ValueError(f"{T} frames given but max_len is {self.max_len}; raise model.max_len")
        h = self.proj(self.norm(x))                                          # (B, T, d_model)
        # Put the CLS token in front (one copy per clip in the batch), then add the positions.
        # self.pos[:, :T + 1] uses only as many positions as this clip has frames.
        h = torch.cat([self.cls.expand(B, -1, -1), h], dim=1) + self.pos[:, :T + 1]
        h = self.encoder(self.drop(h))
        # Classify from position 0, the CLS token: it has gathered information from every frame.
        return self.head(self.out_norm(h[:, 0]))                             # CLS output

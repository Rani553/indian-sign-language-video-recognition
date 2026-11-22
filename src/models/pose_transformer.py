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
        if d_model % nhead:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.max_len = max_len
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, d_model))     # CLS + up to max_len frames
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        if T > self.max_len:
            raise ValueError(f"{T} frames given but max_len is {self.max_len}; raise model.max_len")
        h = self.proj(self.norm(x))                                          # (B, T, d_model)
        h = torch.cat([self.cls.expand(B, -1, -1), h], dim=1) + self.pos[:, :T + 1]
        h = self.encoder(self.drop(h))
        return self.head(self.out_norm(h[:, 0]))                             # CLS output

"""Pipeline B model 1: Pose-LSTM = keypoint features -> (Bi)LSTM over time -> attention pooling -> classifier.

Interface (agreed by the team):
    forward(x)  x: (B, T, F) float  ->  logits (B, num_classes)      F = feature_dim(cfg)

Attention pooling lets the model decide which frames carry the sign (the hands are often at rest
at the start and the end of a clip) instead of averaging all frames equally.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# The ways to turn "one vector per frame" into "one vector per clip". "attention" is the default;
# the others exist so the pooling can be compared in an ablation with a single config line.
POOLING = ("attention", "mean", "max", "last")


class AttentionPooling(nn.Module):
    """Weighted average over time: weight_t = softmax_t(v . tanh(W h_t))."""

    def __init__(self, dim: int, attn_dim: int = 64):
        super().__init__()
        # A tiny network that gives every frame one score ("how important is this frame?").
        # The small hidden size (64) keeps it cheap; one score per frame is all that is needed.
        self.score = nn.Sequential(nn.Linear(dim, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:           # h: (B, T, D)
        # softmax over time turns the scores into weights that add up to 1, so the result is a
        # weighted average: frames with the sign get a large weight, resting frames a small one.
        w = torch.softmax(self.score(h).squeeze(-1), dim=1)       # (B, T), sums to 1 over time
        return (h * w.unsqueeze(-1)).sum(dim=1)                   # (B, D)


class PoseLSTM(nn.Module):
    def __init__(self, num_classes: int, input_dim: int, hidden_size: int = 128, num_layers: int = 2,
                 bidirectional: bool = True, dropout: float = 0.3, pooling: str = "attention"):
        super().__init__()
        # Check the arguments at construction time: a clear error now is better than a crash
        # (or a silently wrong model) in the middle of a long training run.
        if pooling not in POOLING:
            raise ValueError(f"pooling must be one of {POOLING}, got '{pooling}'")
        if input_dim is None or input_dim <= 0:
            raise ValueError("PoseLSTM needs input_dim (the keypoint feature size, from the dataset)")
        self.pooling = pooling
        # The features mix coordinates (~1), velocities (~0.1) and 0/1 flags: normalise them first.
        # LayerNorm works on each frame by itself, so it does not depend on the batch size
        # (BatchNorm would behave differently with a batch of 1 at prediction time).
        self.norm = nn.LayerNorm(input_dim)
        # Shrink the ~500 input features to the hidden size before the LSTM. This keeps the LSTM
        # (the biggest part) small, which matters with only 684 training videos.
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden_size), nn.GELU(), nn.Dropout(dropout))
        # Bidirectional: we classify a finished clip, so the future frames are available, and a
        # sign is often understood from what comes both before and after a frame.
        # LSTM dropout acts between layers, so it needs at least 2 layers (PyTorch warns otherwise).
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        out_dim = hidden_size * (2 if bidirectional else 1)   # two directions are concatenated
        self.attn = AttentionPooling(out_dim) if pooling == "attention" else None
        # Dropout before the last layer: with so little data the classifier overfits quickly.
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(self.proj(self.norm(x)))                 # (B, T, D)
        # Pool over time: (B, T, D) -> (B, D), one vector that describes the whole clip.
        if self.pooling == "attention":
            z = self.attn(h)
        elif self.pooling == "mean":
            z = h.mean(dim=1)
        elif self.pooling == "max":
            z = h.max(dim=1).values
        else:                                                      # last time step
            z = h[:, -1]
        return self.head(z)    # raw scores (logits); the loss applies softmax itself

"""Pipeline A model: LRCN = pretrained CNN on every frame -> (Bi)LSTM over time -> classifier.

Interface (agreed by the team):
    forward(x)  x: (B, T, 3, H, W) float, ImageNet-normalised  ->  logits (B, num_classes)

The CNN turns each frame into a feature vector, the LSTM reads the vectors in time order,
and the pooled LSTM output goes through a linear classifier (Donahue et al., 2015).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tvm

POOLING = ("last", "mean", "max")


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Return (feature extractor, feature size). The extractor maps (N,3,H,W) -> (N,D)."""
    if name == "mobilenet_v2":
        net = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        dim = net.last_channel
        net.classifier = nn.Identity()  # drop the ImageNet classifier, keep pooled features
    elif name == "resnet18":
        net = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        dim = net.fc.in_features
        net.fc = nn.Identity()
    elif name == "efficientnet_b0":
        net = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        dim = net.classifier[1].in_features
        net.classifier = nn.Identity()
    else:
        raise ValueError(f"Unknown backbone '{name}'. Use mobilenet_v2, resnet18 or efficientnet_b0.")
    return net, dim


class CNNLSTM(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "mobilenet_v2", pretrained: bool = True,
                 hidden_size: int = 256, num_layers: int = 1, bidirectional: bool = True,
                 dropout: float = 0.3, pooling: str = "mean", freeze_backbone: bool = False):
        super().__init__()
        if pooling not in POOLING:
            raise ValueError(f"pooling must be one of {POOLING}, got '{pooling}'")
        self.pooling = pooling
        self.cnn, feat_dim = _build_backbone(backbone, pretrained)
        self.lstm = nn.LSTM(feat_dim, hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0.0)  # dropout needs 2+ layers
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, num_classes))
        self.backbone_frozen = False
        if freeze_backbone:
            self.freeze_backbone()

    # ------------------------------------------------------------------ #
    # Freeze -> fine-tune schedule
    # ------------------------------------------------------------------ #
    def freeze_backbone(self) -> None:
        """Stop training the CNN (only LSTM + classifier learn)."""
        for p in self.cnn.parameters():
            p.requires_grad = False
        self.backbone_frozen = True
        self.cnn.eval()  # also keeps BatchNorm statistics fixed

    def unfreeze_backbone(self) -> None:
        """Let the CNN learn too (fine-tuning)."""
        for p in self.cnn.parameters():
            p.requires_grad = True
        self.backbone_frozen = False
        self.cnn.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_frozen:
            self.cnn.eval()  # Trainer calls model.train() every epoch; keep the frozen CNN in eval
        return self

    def param_groups(self, lr: float, backbone_lr: float | None = None) -> list[dict]:
        """Optimizer groups: the pretrained CNN gets a smaller learning rate than the new layers.

        Frozen CNN parameters are included too (with requires_grad=False they get no gradient),
        so unfreezing later needs no new optimizer.
        """
        new_params = list(self.lstm.parameters()) + list(self.head.parameters())
        backbone_lr = lr * 0.1 if backbone_lr is None else float(backbone_lr)
        return [{"params": list(self.cnn.parameters()), "lr": backbone_lr},
                {"params": new_params, "lr": lr}]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape[:2]
        frames = x.flatten(0, 1)  # (B*T, 3, H, W): all frames go through the CNN as one big batch
        if self.backbone_frozen:
            with torch.no_grad():  # no gradients through a frozen CNN: saves memory and time
                feats = self.cnn(frames)
        else:
            feats = self.cnn(frames)
        out, _ = self.lstm(feats.view(b, t, -1))  # (B, T, hidden * directions)
        if self.pooling == "last":
            # For a BiLSTM the last step only saw the backward direction's first frame, so
            # take the forward state at t=-1 and the backward state at t=0.
            if self.lstm.bidirectional:
                h = out.size(-1) // 2
                pooled = torch.cat([out[:, -1, :h], out[:, 0, h:]], dim=1)
            else:
                pooled = out[:, -1]
        elif self.pooling == "mean":
            pooled = out.mean(dim=1)
        else:
            pooled = out.max(dim=1).values
        return self.head(pooled)

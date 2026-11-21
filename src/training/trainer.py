"""Generic training loop used by every model.

Features: AdamW + warmup/cosine LR, label smoothing, mixed precision (AMP),
gradient accumulation, gradient clipping, early stopping, and checkpointing
every epoch so a disconnected Colab/Kaggle session can resume.

Files written to the experiment folder:
    last.pt       full state (model, optimizer, scheduler, epoch) -> resume
    best.pt       best model by validation accuracy (+ config, label map)
    history.csv   one row of metrics per epoch
    curves.png    loss / accuracy curves
    train.log     console log
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from src.utils.logger import CSVLogger, TensorBoardLogger, get_logger


def topk_correct(logits: torch.Tensor, y: torch.Tensor, k: int) -> int:
    """Count samples whose true class is among the model's k highest scores."""
    k = min(k, logits.shape[1])  # cannot take top-5 if there are fewer than 5 classes
    return (logits.topk(k, dim=1).indices == y[:, None]).any(dim=1).sum().item()


def warmup_cosine(total_steps: int, warmup_steps: int, min_factor: float = 0.01) -> Callable:
    """Learning-rate schedule: rise linearly during warmup, then fall along a cosine curve.

    Returns a function step -> factor; the real learning rate is lr * factor.
    The factor never goes below ``min_factor`` (1% of lr by default).
    """
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)  # warmup: small -> 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)  # 0.0 -> 1.0
        return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return fn


class Trainer:
    """Trains any model that follows the team interface: ``model(x) -> logits (B, num_classes)``."""

    def __init__(self, model: nn.Module, loaders: dict, cfg: dict, out_dir: str | Path,
                 device: torch.device, extra_ckpt: dict | None = None,
                 epoch_start_hook: Callable[[int, "Trainer"], None] | None = None):
        self.model = model.to(device)
        self.loaders = loaders
        self.cfg = cfg
        self.t = cfg["train"]  # shortcut to the "train:" section of the config
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.extra_ckpt = extra_ckpt or {}  # extra info saved in every checkpoint
        self.epoch_start_hook = epoch_start_hook
        # Loggers: screen + train.log, history.csv, and (optional) TensorBoard.
        self.log = get_logger(f"train.{self.out.name}", self.out / "train.log")
        self.csv = CSVLogger(self.out / "history.csv")
        self.tb = TensorBoardLogger(self.out / "tensorboard", self.t.get("tensorboard", False))

        # Optimizer. A model can define param_groups() to give its pretrained
        # backbone a smaller learning rate than the new layers.
        lr = float(self.t["lr"])
        if hasattr(model, "param_groups"):
            groups = model.param_groups(lr, self.t.get("backbone_lr"))
        else:
            groups = [{"params": list(model.parameters()), "lr": lr}]
        self.optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=float(self.t.get("weight_decay", 1e-4)))
        # Gradient accumulation: add up gradients over `accum` batches before one
        # optimizer step. Acts like a bigger batch without needing more GPU memory.
        self.accum = max(1, int(self.t.get("grad_accum_steps", 1)))
        steps_per_epoch = max(1, math.ceil(len(loaders["train"]) / self.accum))
        epochs = int(self.t["epochs"])
        # The scheduler is stepped once per optimizer step (not per epoch).
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, warmup_cosine(steps_per_epoch * epochs,
                                          steps_per_epoch * int(self.t.get("warmup_epochs", 0))))
        # Mixed precision (AMP): faster and uses less memory, but only on a GPU.
        self.use_amp = bool(self.t.get("amp", True)) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        # Label smoothing makes the model a little less over-confident.
        self.criterion = nn.CrossEntropyLoss(label_smoothing=float(self.t.get("label_smoothing", 0.0)))

        # Training state (overwritten by try_resume() if a last.pt exists).
        self.start_epoch = 0
        self.best_acc = -1.0
        self.bad_epochs = 0  # epochs in a row without a better val accuracy (for early stopping)

    # ------------------------------------------------------------------ #
    def _save(self, path: Path, full: bool, epoch: int) -> None:
        """Save a checkpoint. ``full=True`` also saves optimizer/scheduler state (needed to resume)."""
        state = {"model_state": self.model.state_dict(), "config": self.cfg, "epoch": epoch,
                 "best_val_acc": self.best_acc, **self.extra_ckpt}
        if full:
            state.update(optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(),
                         scaler=self.scaler.state_dict(), bad_epochs=self.bad_epochs)
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(path)   # atomic: a disconnect never leaves a broken checkpoint

    def try_resume(self) -> None:
        """If last.pt exists (and resume is on), continue training from where it stopped."""
        last = self.out / "last.pt"
        if not (self.t.get("resume", True) and last.exists()):
            return
        ckpt = torch.load(last, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch = ckpt["epoch"] + 1  # the saved epoch is finished, start with the next one
        self.best_acc = ckpt["best_val_acc"]
        self.bad_epochs = ckpt.get("bad_epochs", 0)
        self.log.info(f"Resumed from epoch {ckpt['epoch']} (best val acc {self.best_acc:.4f})")

    # ------------------------------------------------------------------ #
    def train_one_epoch(self) -> dict:
        """One pass over the training data. Returns average loss and accuracy."""
        self.model.train()  # training mode: dropout and BatchNorm updates are on
        loss_sum, correct, n = 0.0, 0, 0
        self.optimizer.zero_grad(set_to_none=True)
        loader = self.loaders["train"]
        for step, (x, y) in enumerate(loader):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            # Forward pass (in half precision when AMP is on).
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            # Backward pass. The loss is divided by `accum` so the accumulated
            # gradient is an average, not a sum.
            self.scaler.scale(loss / self.accum).backward()
            # Update the weights every `accum` batches (and after the last batch).
            if (step + 1) % self.accum == 0 or step + 1 == len(loader):
                if self.t.get("grad_clip"):
                    # Limit the gradient size to avoid unstable, exploding updates.
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), float(self.t["grad_clip"]))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
            # Running totals for the epoch averages.
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        return {"train_loss": loss_sum / max(1, n), "train_acc": correct / max(1, n)}

    @torch.no_grad()  # no gradients needed for evaluation -> faster, less memory
    def evaluate(self, split: str = "val") -> dict:
        """Compute loss, top-1 accuracy and top-5 accuracy on one split."""
        self.model.eval()  # evaluation mode: dropout off, BatchNorm uses saved statistics
        loss_sum, c1, c5, n = 0.0, 0, 0, 0
        for x, y in self.loaders[split]:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            loss_sum += loss.item() * len(y)
            c1 += topk_correct(logits, y, 1)
            c5 += topk_correct(logits, y, 5)
            n += len(y)
        return {f"{split}_loss": loss_sum / max(1, n), f"{split}_acc": c1 / max(1, n),
                f"{split}_top5": c5 / max(1, n)}

    # ------------------------------------------------------------------ #
    def fit(self) -> dict:
        """Full training: resume if possible, train epoch by epoch, save checkpoints, stop early."""
        self.try_resume()
        epochs = int(self.t["epochs"])
        patience = int(self.t.get("early_stopping_patience", 0) or 0)  # 0 = no early stopping
        # If there is no validation split (e.g. tiny mode), judge on the train split.
        val_split = "val" if "val" in self.loaders else "train"
        self.log.info(f"Training on {self.device} | AMP={self.use_amp} | epochs={epochs} | "
                      f"train batches={len(self.loaders['train'])} | out={self.out}")
        for epoch in range(self.start_epoch, epochs):
            if self.epoch_start_hook:
                self.epoch_start_hook(epoch, self)
            t0 = time.time()
            metrics = {"epoch": epoch, **self.train_one_epoch(), **self.evaluate(val_split)}
            metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            metrics["epoch_time_s"] = round(time.time() - t0, 2)
            val_acc = metrics[f"{val_split}_acc"]
            improved = val_acc > self.best_acc
            if improved:
                # New best model -> save it as best.pt and reset the early-stopping counter.
                self.best_acc, self.bad_epochs = val_acc, 0
                self._save(self.out / "best.pt", full=False, epoch=epoch)
            else:
                self.bad_epochs += 1
            # Always save last.pt, so training can resume after a disconnect.
            self._save(self.out / "last.pt", full=True, epoch=epoch)
            self.csv.log({k: (round(v, 5) if isinstance(v, float) else v) for k, v in metrics.items()})
            self.tb.log(metrics, epoch)
            self.log.info(
                f"ep {epoch:3d} | train loss {metrics['train_loss']:.3f} acc {metrics['train_acc']:.3f} | "
                f"{val_split} loss {metrics[f'{val_split}_loss']:.3f} acc {val_acc:.3f} "
                f"top5 {metrics[f'{val_split}_top5']:.3f} | {metrics['epoch_time_s']:.0f}s"
                + ("  *best*" if improved else ""))
            # Early stopping: stop when val accuracy has not improved for `patience` epochs.
            if patience and self.bad_epochs >= patience:
                self.log.info(f"Early stopping after {patience} epochs without improvement")
                break
        self.tb.close()
        self.plot_curves()
        self.log.info(f"Best {val_split} accuracy: {self.best_acc:.4f} -> {self.out / 'best.pt'}")
        return {"best_val_acc": self.best_acc}

    def plot_curves(self) -> None:
        """Draw loss and accuracy per epoch (train vs val) from history.csv into curves.png."""
        rows = self.csv.read()
        if not rows:
            return
        import matplotlib
        matplotlib.use("Agg")  # draw to a file, no screen needed (works on Colab/servers)
        import matplotlib.pyplot as plt
        ep = [int(r["epoch"]) for r in rows]
        val_key = "val" if "val_loss" in rows[0] else "train"
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))  # left: loss, right: accuracy
        for key, ax in (("loss", axes[0]), ("acc", axes[1])):
            ax.plot(ep, [float(r[f"train_{key}"]) for r in rows], label="train")
            ax.plot(ep, [float(r[f"{val_key}_{key}"]) for r in rows], label=val_key)
            ax.set_xlabel("epoch"), ax.set_title(key), ax.legend(), ax.grid(alpha=0.3)
        fig.suptitle(self.out.name)
        fig.tight_layout()
        fig.savefig(self.out / "curves.png", dpi=120)
        plt.close(fig)

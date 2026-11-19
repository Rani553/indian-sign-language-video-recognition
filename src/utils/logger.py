"""Console + file logging and a small CSV metrics logger."""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path


def get_logger(name: str = "isl", log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file is not None:
        log_file = str(log_file)
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(Path(log_file).resolve())
                   for h in logger.handlers):
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    logger.propagate = False
    return logger


class CSVLogger:
    """Append one row of metrics per epoch to a CSV file (survives Colab restarts)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def log(self, row: dict) -> None:
        exists = self.path.exists()
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, newline="") as f:
            return list(csv.DictReader(f))


class TensorBoardLogger:
    """Optional TensorBoard logging; silently does nothing if not installed."""

    def __init__(self, log_dir: str | Path, enabled: bool = True):
        self.writer = None
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(log_dir))
            except Exception:  # tensorboard not installed
                self.writer = None

    def log(self, metrics: dict, step: int) -> None:
        if self.writer is None:
            return
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()

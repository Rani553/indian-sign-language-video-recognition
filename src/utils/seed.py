"""Reproducibility helpers."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Fix all random number generators so a run gives the same result when repeated."""
    random.seed(seed)                 # Python's random module
    np.random.seed(seed)              # NumPy
    torch.manual_seed(seed)           # PyTorch on CPU
    torch.cuda.manual_seed_all(seed)  # PyTorch on every GPU
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # Exactly repeatable GPU results, but slower.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Let cuDNN pick the fastest algorithms (tiny differences between runs).
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """Give every DataLoader worker a different but reproducible seed."""
    # PyTorch already gives each worker its own seed; reuse it for NumPy and random,
    # otherwise all workers would produce the same "random" augmentations.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(preferred: str = "auto") -> torch.device:
    """Return the device to use: "auto" picks the GPU if there is one, otherwise the CPU."""
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)

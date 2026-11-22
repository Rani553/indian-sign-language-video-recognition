"""Classification metrics for the sign recognition models (plain NumPy / scikit-learn, no torch).

Inputs are always:
    y_true  (N,)    true label indices
    probs   (N, C)  predicted class probabilities (softmax) or any scores where larger = better
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def topk_accuracy(y_true: np.ndarray, probs: np.ndarray, k: int) -> float:
    """Share of samples whose true class is among the ``k`` highest scores."""
    k = min(k, probs.shape[1])
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float((topk == y_true[:, None]).any(axis=1).mean())


def wilson_interval(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson confidence interval for an accuracy of ``correct`` out of ``n``.

    With ~137 test videos one video is 0.7 points, so the interval is wide (about +-4 points).
    It tells the reader which differences between models are noise.
    """
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Top-1 / top-5 accuracy, macro and weighted F1, macro precision / recall, 95 % interval of top-1."""
    y_true = np.asarray(y_true)
    y_pred = probs.argmax(axis=1)
    n = int(len(y_true))
    correct = int((y_pred == y_true).sum())
    labels = np.arange(probs.shape[1])
    prec, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                      average="macro", zero_division=0)
    lo, hi = wilson_interval(correct, n)
    return {
        "n": n,
        "correct": correct,
        "top1": correct / max(1, n),
        "top5": topk_accuracy(y_true, probs, 5),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "top1_ci95": [lo, hi],
    }


def confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """(C, C) matrix: rows = true class, columns = predicted class."""
    return confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))


def per_class_table(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    """One row per class: support, correct, precision, recall, F1. Sorted worst class first."""
    labels = np.arange(len(class_names))
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    correct = np.array([int(((y_true == c) & (y_pred == c)).sum()) for c in labels])
    df = pd.DataFrame({"class": class_names, "support": support, "correct": correct,
                       "precision": prec.round(4), "recall": rec.round(4), "f1": f1.round(4)})
    return df.sort_values(["f1", "class"]).reset_index(drop=True)


def top_confusions(cm: np.ndarray, class_names: list[str], k: int = 10) -> list[dict]:
    """The ``k`` most frequent wrong (true -> predicted) pairs."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.argsort(-off, axis=None)[:k]
    out = []
    for flat in order:
        t, p = np.unravel_index(flat, off.shape)
        if off[t, p] > 0:
            out.append({"true": class_names[t], "predicted": class_names[p], "count": int(off[t, p])})
    return out


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], path: str | Path,
                          title: str = "", normalize: bool = True) -> None:
    """Save the confusion matrix as a PNG (row-normalised by default: each row sums to 1)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = cm.astype(float)
    if normalize:
        data = data / np.maximum(data.sum(axis=1, keepdims=True), 1)
    c = len(class_names)
    size = max(6.0, min(18.0, c * 0.32))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(c))
    ax.set_yticks(range(c))
    ax.set_xticklabels(class_names, rotation=90, fontsize=max(4, 9 - c // 12))
    ax.set_yticklabels(class_names, fontsize=max(4, 9 - c // 12))
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

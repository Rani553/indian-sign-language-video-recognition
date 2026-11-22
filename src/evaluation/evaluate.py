"""Evaluate trained checkpoints on a split (test by default).

    python -m src.evaluation.evaluate --checkpoint results/pose_transformer/best.pt
    python -m src.evaluation.evaluate --checkpoint "results/*/best.pt" --split test
    python -m src.evaluation.evaluate --checkpoint results/x/best.pt --set paths.data_root=/content/isl_data

Writes, next to the checkpoint, ``eval_<split>/``:
    metrics.json            top-1, top-5, macro F1, 95 % interval, parameters, ...
    per_class.csv           precision / recall / F1 for every class (worst first)
    predictions.csv         one row per video: true label, prediction, confidence
    confusion_matrix.png
    top_confusions.json     the most frequent mistakes

The dataset type (frames or keypoints) comes from the config stored in the checkpoint, so the same
command works for every model. Evaluation is deterministic: no augmentation, fp32, fixed frame sampling.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import build_dataset
from src.evaluation import metrics as M
from src.evaluation.checkpoint import class_names_from, experiment_name, load_checkpoint
from src.models import count_parameters
from src.utils.config import apply_overrides
from src.utils.logger import get_logger
from src.utils.seed import get_device

log = get_logger("evaluate")


@torch.no_grad()
def predict_split(model: torch.nn.Module, cfg: dict, split: str, device: torch.device,
                  batch_size: int = 32) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run the model on a whole split. Returns ``(y_true, probs, video_ids)``."""
    ds = build_dataset(cfg, split, train=False)   # train=False: no augmentation
    if len(ds) == 0:
        raise ValueError(f"The '{split}' split is empty")
    # num_workers=0: simplest and safe on Windows; the datasets are cached files, so it is fast enough.
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    ys, ps = [], []
    for x, y in loader:
        logits = model(x.to(device))
        ps.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps), ds.df["video_id"].tolist()


def evaluate_checkpoint(path: str | Path, split: str = "test", device: str = "auto", batch_size: int = 32,
                        overrides: list[str] | None = None, out_name: str | None = None) -> dict:
    """Evaluate one checkpoint and write all result files. Returns the metrics dict."""
    path = Path(path)
    dev = get_device(device)
    model, cfg, ckpt = load_checkpoint(path, dev)
    cfg = apply_overrides(cfg, overrides)         # e.g. paths.data_root=... on another machine
    names = class_names_from(ckpt["label_map"])
    exp = experiment_name(path, cfg)

    y_true, probs, video_ids = predict_split(model, cfg, split, dev, batch_size)
    y_pred = probs.argmax(axis=1)
    res = M.compute_metrics(y_true, probs)
    res.update({
        "experiment": exp,
        "model": cfg["model"]["name"],
        "input_type": cfg.get("input_type"),
        "split": split,
        "checkpoint": str(path),
        "params_m": count_parameters(model) / 1e6,
        "epoch": ckpt.get("epoch"),
        "best_val_acc": ckpt.get("best_val_acc"),
        "num_classes": len(names),
    })

    out = path.parent / (out_name or f"eval_{split}")
    out.mkdir(parents=True, exist_ok=True)
    cm = M.confusion(y_true, y_pred, len(names))
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    M.per_class_table(y_true, y_pred, names).to_csv(out / "per_class.csv", index=False)
    pd.DataFrame({
        "video_id": video_ids,
        "true_label": [names[i] for i in y_true],
        "pred_label": [names[i] for i in y_pred],
        "correct": y_pred == y_true,
        "confidence": probs.max(axis=1).round(4),
    }).to_csv(out / "predictions.csv", index=False)
    (out / "top_confusions.json").write_text(json.dumps(M.top_confusions(cm, names), indent=2))
    M.plot_confusion_matrix(cm, names, out / "confusion_matrix.png", title=f"{exp} ({split})")

    lo, hi = res["top1_ci95"]
    log.info(f"{exp} [{split}] top-1 {res['top1']:.4f} ({res['correct']}/{res['n']}, 95% CI {lo:.3f}-{hi:.3f}) | "
             f"top-5 {res['top5']:.4f} | macro-F1 {res['macro_f1']:.4f} | -> {out}")
    return res


def expand_checkpoints(patterns: list[str]) -> list[Path]:
    """Expand globs ourselves, because PowerShell and cmd do not expand ``results/*/best.pt``."""
    found: list[Path] = []
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        found.extend(Path(m) for m in (matches or [pat]))
    return found


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", nargs="+", required=True, help="best.pt file(s); globs are allowed")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Override config values, e.g. paths.data_root=/content/isl_data")
    args = p.parse_args()

    rows = []
    for ckpt in expand_checkpoints(args.checkpoint):
        try:
            rows.append(evaluate_checkpoint(ckpt, args.split, args.device, args.batch_size, args.set))
        except Exception as e:  # one broken checkpoint must not stop the others
            log.warning(f"FAILED {ckpt}: {type(e).__name__}: {e}")
    if rows:
        print()
        print(pd.DataFrame(rows)[["experiment", "n", "top1", "top5", "macro_f1", "params_m"]]
              .round(4).to_string(index=False))


if __name__ == "__main__":
    main()

"""Exploratory data analysis of the prepared INCLUDE-50 data (reads ``processed/``, needs no model).

    python -m src.data.eda
    python -m src.data.eda --out results/eda

Writes (default: ``<results_root>/eda/``):
    summary.json              sizes of the splits, class balance, categories
    class_counts.csv          videos per class and split
    class_balance.png
    keypoint_detection.csv    how often MediaPipe found the body / each hand, per class (if keypoints exist)
    keypoint_detection.png
    split_leak_check.json     do test videos have a near-duplicate take in the training set?

The leak check matters for how to read every accuracy of the project. INCLUDE videos are named
``MVI_<number>`` and one signer records a sign several times in a row (consecutive numbers). The
split is random per video, so takes of one session can land in train AND test. The check counts
test videos that have a training video of the same class with a number at most ``--gap`` away. It is
a heuristic (the dataset has no signer id), not a proof.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import add_common_args, config_from_args, resolve_paths
from src.utils.logger import get_logger

log = get_logger("eda")
BLUE, INK, INK_2, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_splits(splits_dir: Path) -> pd.DataFrame:
    """All videos with a ``split`` column (train / val / test)."""
    parts = []
    for s in ("train", "val", "test"):
        f = splits_dir / f"{s}.csv"
        if f.exists():
            parts.append(pd.read_csv(f).assign(split=s))
    if not parts:
        raise FileNotFoundError(f"No split CSVs in {splits_dir}. Run `python -m src.data.split` first.")
    return pd.concat(parts, ignore_index=True)


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Videos per class and split."""
    t = df.pivot_table(index="label", columns="split", values="video_id", aggfunc="count", fill_value=0)
    for s in ("train", "val", "test"):
        if s not in t:
            t[s] = 0
    t = t[["train", "val", "test"]]
    t["total"] = t.sum(axis=1)
    return t.sort_values("total", ascending=False).reset_index()


def mvi_number(video_id: str) -> int | None:
    """The camera file number in an INCLUDE video id ('adjectives_dry_mvi_5167' -> 5167)."""
    m = re.search(r"mvi_(\d+)", str(video_id).lower())
    return int(m.group(1)) if m else None


def leak_check(df: pd.DataFrame, gap: int = 2) -> dict:
    """Share of test videos with a same-class training video whose file number is at most ``gap`` away."""
    df = df.assign(num=df["video_id"].map(mvi_number))
    if df["num"].isna().all():
        return {"available": False, "reason": "video ids have no MVI number"}
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    near = 0
    for _, r in test.dropna(subset=["num"]).iterrows():
        same = train[(train["label"] == r["label"]) & train["num"].notna()]
        if ((same["num"] - r["num"]).abs() <= gap).any():
            near += 1
    n = int(test["num"].notna().sum())
    return {"available": True, "gap": gap, "test_videos": n, "with_near_duplicate_in_train": near,
            "share": near / max(1, n)}


def keypoint_detection(df: pd.DataFrame, keypoints_dir: Path) -> pd.DataFrame | None:
    """Per class: share of frames in which the body and each hand were found (from the cached .npz files)."""
    if not keypoints_dir.exists():
        return None
    from src.features.keypoint_io import load_keypoints, part_present

    rows = []
    for r in df.itertuples(index=False):
        f = keypoints_dir / f"{r.video_id}.npz"
        if not f.exists():
            continue
        k = load_keypoints(f)
        rows.append({"label": r.label, "pose": part_present(k["pose"]).mean(),
                     "left_hand": part_present(k["left_hand"]).mean(),
                     "right_hand": part_present(k["right_hand"]).mean()})
    if not rows:
        return None
    return pd.DataFrame(rows).groupby("label").mean().round(4).reset_index()


# --------------------------------------------------------------------------- #
def _axes(figsize):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def plot_class_balance(t: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = _axes((max(8, 0.28 * len(t)), 4.2))
    ax.bar(range(len(t)), t["total"], color=BLUE, width=0.7, zorder=3)
    ax.set_xticks(range(len(t)))
    ax.set_xticklabels(t["label"], rotation=90)
    ax.set_ylabel("videos", color=INK_2)
    ax.set_title(f"Videos per class ({len(t)} classes, {int(t['total'].sum())} videos; "
                 f"min {int(t['total'].min())}, max {int(t['total'].max())})", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def plot_keypoint_detection(k: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    k = k.assign(hands=(k["left_hand"] + k["right_hand"]) / 2).sort_values("hands")
    fig, ax = _axes((max(8, 0.28 * len(k)), 4.2))
    ax.bar(range(len(k)), k["hands"], color=BLUE, width=0.7, zorder=3)
    ax.set_xticks(range(len(k)))
    ax.set_xticklabels(k["label"], rotation=90)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of frames with a hand found", color=INK_2)
    ax.set_title("MediaPipe hand detection per class (mean of left and right hand)",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def run_eda(cfg: dict, out_dir: Path | None = None, gap: int = 2) -> dict:
    paths = resolve_paths(cfg)
    out = Path(out_dir) if out_dir else paths["results_root"] / "eda"
    out.mkdir(parents=True, exist_ok=True)
    df = load_splits(paths["splits_dir"])
    bal = class_balance(df)
    bal.to_csv(out / "class_counts.csv", index=False)
    plot_class_balance(bal, out / "class_balance.png")

    summary = {
        "videos": int(len(df)), "classes": int(df["label"].nunique()),
        "split_sizes": df["split"].value_counts().to_dict(),
        "videos_per_class": {"min": int(bal["total"].min()), "max": int(bal["total"].max()),
                             "mean": round(float(bal["total"].mean()), 2)},
        "classes_missing_from_a_split": int(((bal[["train", "val", "test"]] == 0).any(axis=1)).sum()),
        "categories": df["category"].value_counts().to_dict() if "category" in df else {},
    }
    leak = leak_check(df, gap)
    (out / "split_leak_check.json").write_text(json.dumps(leak, indent=2))
    k = keypoint_detection(df, paths["keypoints_dir"])
    if k is not None:
        k.to_csv(out / "keypoint_detection.csv", index=False)
        plot_keypoint_detection(k, out / "keypoint_detection.png")
        summary["keypoint_detection"] = {c: round(float(k[c].mean()), 3) for c in ("pose", "left_hand", "right_hand")}
    summary["split_leak_check"] = leak
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"{summary['videos']} videos, {summary['classes']} classes, splits {summary['split_sizes']} -> {out}")
    if leak.get("available"):
        log.info(f"Leak check: {leak['with_near_duplicate_in_train']}/{leak['test_videos']} test videos "
                 f"({leak['share']:.0%}) have a same-class training video within {leak['gap']} file numbers")
    return summary


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                     formatter_class=argparse.RawDescriptionHelpFormatter),
                             "configs/data.yaml")
    parser.add_argument("--out", default=None, help="output folder (default: <results_root>/eda)")
    parser.add_argument("--gap", type=int, default=2, help="file-number distance for the leak check")
    args = parser.parse_args()
    run_eda(config_from_args(args), args.out, args.gap)


if __name__ == "__main__":
    main()

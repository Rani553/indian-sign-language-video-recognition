"""Scan the raw videos, choose the classes, and make a stratified train/val/test split.

Outputs (in ``processed/splits``):
    metadata.csv          every video found (path, label, category, video_id)
    train.csv / val.csv / test.csv
    label_map.json        {label: index}  -- shared by all members
    classes.txt           the classes used (handy to re-use as a class list)

Usage:
    python -m src.data.split
    python -m src.data.split --set dataset.num_classes=263   # full INCLUDE
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.download import clean_label, read_class_list
from src.utils.config import PROJECT_ROOT, add_common_args, config_from_args, resolve_paths
from src.utils.logger import get_logger
from src.utils.video import VIDEO_EXTENSIONS

log = get_logger("split")


def make_video_id(rel_path: Path) -> str:
    stem = "_".join(rel_path.with_suffix("").parts)
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_").lower()


def scan_videos(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(raw_dir.rglob("*")):
        if p.suffix.lower() not in VIDEO_EXTENSIONS or not p.is_file():
            continue
        rel = p.relative_to(raw_dir)
        if len(rel.parts) < 2:
            continue
        rows.append({
            "video_id": make_video_id(rel),
            "video_path": rel.as_posix(),
            "label": clean_label(rel.parts[-2]),
            "category": rel.parts[-3] if len(rel.parts) >= 3 else "",
        })
    if not rows:
        raise FileNotFoundError(f"No videos found under {raw_dir}. Run src.data.download first.")
    return pd.DataFrame(rows)


def choose_classes(meta: pd.DataFrame, ds_cfg: dict) -> list[str]:
    class_file = ds_cfg.get("class_list")
    if class_file and not Path(class_file).is_absolute():
        class_file = PROJECT_ROOT / class_file
    wanted = read_class_list(class_file)
    counts = meta["label"].value_counts()
    eligible = counts[counts >= int(ds_cfg.get("min_videos_per_class", 3))]

    if wanted:
        missing = sorted(wanted - set(eligible.index))
        if missing:
            log.warning(f"{len(missing)} classes from the list are missing/too small: {missing[:10]}")
        return sorted(wanted & set(eligible.index))

    n = ds_cfg.get("num_classes")
    if not n or n >= len(eligible):
        return sorted(eligible.index)
    log.warning("No class list file found -> choosing classes with "
                f"strategy '{ds_cfg.get('subset_strategy')}'. Report this in your results!")
    if ds_cfg.get("subset_strategy") == "random":
        rng = np.random.default_rng(int(ds_cfg.get("seed", 42)))
        return sorted(rng.choice(sorted(eligible.index), size=n, replace=False))
    # most_videos: sort by count (desc) then name for determinism
    ranked = sorted(eligible.items(), key=lambda kv: (-kv[1], kv[0]))
    return sorted(name for name, _ in ranked[:n])


def stratified_split(meta: pd.DataFrame, val_size: float, test_size: float, seed: int):
    """Per-class split so every class appears in train, val and test."""
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for _, group in meta.groupby("label", sort=True):
        idx = rng.permutation(group.index.to_numpy())
        n = len(idx)
        n_test = max(1, int(round(n * test_size))) if n >= 3 else 0
        n_val = max(1, int(round(n * val_size))) if n - n_test >= 2 else 0
        parts["test"].extend(idx[:n_test])
        parts["val"].extend(idx[n_test:n_test + n_val])
        parts["train"].extend(idx[n_test + n_val:])
    return {k: meta.loc[v].sort_values("video_id").reset_index(drop=True) for k, v in parts.items()}


def build_splits(cfg: dict) -> dict:
    paths = resolve_paths(cfg)
    ds = cfg["dataset"]
    meta = scan_videos(paths["raw_dir"])
    log.info(f"Found {len(meta)} videos, {meta['label'].nunique()} classes")

    classes = choose_classes(meta, ds)
    label_map = {c: i for i, c in enumerate(sorted(classes))}
    meta = meta[meta["label"].isin(label_map)].copy()
    meta["label_idx"] = meta["label"].map(label_map)

    splits = stratified_split(meta, ds["val_size"], ds["test_size"], int(ds.get("seed", 42)))
    out = paths["splits_dir"]
    out.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out / "metadata.csv", index=False)
    for name, df in splits.items():
        df.to_csv(out / f"{name}.csv", index=False)
    (out / "label_map.json").write_text(json.dumps(label_map, indent=2))
    (out / "classes.txt").write_text("\n".join(sorted(classes)) + "\n")
    log.info("Split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items())
             + f" | classes={len(label_map)} | saved to {out}")
    return splits


# --------------------------------------------------------------------------- #
# Reading splits (used by every dataset class)
# --------------------------------------------------------------------------- #
def load_label_map(splits_dir: Path) -> dict[str, int]:
    path = Path(splits_dir) / "label_map.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.data.split` first.")
    return json.loads(path.read_text())


def read_split(splits_dir: Path, split: str, cfg: dict) -> pd.DataFrame:
    """Read a split CSV and apply tiny mode if enabled."""
    df = pd.read_csv(Path(splits_dir) / f"{split}.csv")
    dcfg = cfg.get("data", {})
    if dcfg.get("tiny"):
        keep = sorted(df["label_idx"].unique())[: int(dcfg.get("tiny_classes", 5))]
        df = (df[df["label_idx"].isin(keep)]
              .groupby("label_idx", group_keys=False)
              .head(int(dcfg.get("tiny_videos_per_class", 4)))
              .reset_index(drop=True))
    return df


def load_all_split_rows(splits_dir: Path) -> pd.DataFrame:
    dfs = [pd.read_csv(splits_dir / f"{s}.csv") for s in ("train", "val", "test")
           if (splits_dir / f"{s}.csv").exists()]
    if not dfs:
        raise FileNotFoundError(f"No split CSVs in {splits_dir}. Run src.data.split first.")
    return pd.concat(dfs, ignore_index=True)


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__), "configs/data.yaml")
    args = parser.parse_args()
    build_splits(config_from_args(args))


if __name__ == "__main__":
    main()

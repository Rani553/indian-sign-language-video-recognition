"""Decode every video once and cache a fixed set of frames: <frames_dir>/<video_id>.npy.

Each file has shape (frames.stored_frames, stored_size, stored_size, 3), uint8. Training reads
only these files, never the raw videos. Existing files are skipped, so the script can be
re-run after a disconnect (use --overwrite to redo everything).

    python -m src.features.extract_frames
    python -m src.features.extract_frames --tiny
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.split import read_split
from src.utils.config import add_common_args, config_from_args, resolve_paths
from src.utils.logger import get_logger
from src.utils.video import sample_and_resize

log = get_logger("extract_frames")


def extract_one(row, raw_dir: Path, out_dir: Path, num_frames: int, size: int, mode: str,
                overwrite: bool) -> str:
    """Cache one video. Returns "done", "skipped" or an error message."""
    out = out_dir / f"{row.video_id}.npy"
    if out.exists() and not overwrite:
        return "skipped"
    try:
        clip = sample_and_resize(raw_dir / row.video_path, num_frames, size, mode)
    except Exception as e:  # a broken video must not stop the whole run
        return f"{row.video_path}: {e}"
    tmp = out.with_name(out.stem + ".tmp.npy")
    np.save(tmp, clip)
    tmp.replace(out)  # atomic: a disconnect never leaves a half-written file
    return "done"


def extract_frames(cfg: dict, overwrite: bool = False) -> dict:
    paths = resolve_paths(cfg)
    f = cfg["frames"]
    out_dir = paths["frames_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    # read_split applies tiny mode, so --tiny only processes the few videos it will use.
    rows = pd.concat([read_split(paths["splits_dir"], s, cfg) for s in ("train", "val", "test")],
                     ignore_index=True)
    log.info(f"Extracting {len(rows)} videos -> {out_dir}")
    work = lambda r: extract_one(r, paths["raw_dir"], out_dir, int(f["stored_frames"]),  # noqa: E731
                                 int(f["stored_size"]), f.get("resize_mode", "pad"), overwrite)
    # Threads are enough: OpenCV decoding releases the GIL.
    with ThreadPoolExecutor(max_workers=max(1, int(f.get("workers", 2)))) as pool:
        results = list(pool.map(work, rows.itertuples(index=False)))
    errors = [r for r in results if r not in ("done", "skipped")]
    summary = {"done": results.count("done"), "skipped": results.count("skipped"), "failed": len(errors)}
    for e in errors:
        log.warning(f"FAILED {e}")
    log.info(f"Finished: {summary}")
    return summary


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                     formatter_class=argparse.RawDescriptionHelpFormatter),
                             "configs/data.yaml")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract videos that already have a cache")
    args = parser.parse_args()
    extract_frames(config_from_args(args), args.overwrite)


if __name__ == "__main__":
    main()

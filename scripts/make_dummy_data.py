"""Make a tiny synthetic dataset so the whole pipeline (and the tests) run without INCLUDE.

Writes the same layout as the real preparation steps, under ``<data-root>/processed``:
    splits/   metadata.csv, train.csv, val.csv, test.csv, label_map.json, classes.txt
    frames/   <video_id>.npy   (frames, size, size, 3) uint8

Each fake class has its own colour and its own motion direction, so a model can actually
learn it (loss goes down) - useful for checking the train -> eval loop end to end.

Usage:
    python scripts/make_dummy_data.py --data-root /tmp/isl_dummy
    export ISL_DATA_ROOT=/tmp/isl_dummy     # then run any script with --tiny
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python scripts/...` from anywhere

from src.data.split import stratified_split  # noqa: E402


def fake_clip(class_idx: int, num_classes: int, frames: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """A coloured square moving across a noisy background; colour and direction depend on the class."""
    clip = rng.integers(0, 40, (frames, size, size, 3), dtype=np.uint8)           # dark noise
    colour = (np.array([(class_idx * 97) % 256, (class_idx * 57 + 90) % 256,
                        (class_idx * 31 + 160) % 256]) * 0.8 + 50).clip(0, 255).astype(np.uint8)
    angle = 2 * np.pi * class_idx / num_classes                                    # class = direction
    dx, dy = np.cos(angle), np.sin(angle)
    box = max(4, size // 5)
    start = rng.uniform(0.35, 0.65, 2) * size                                      # small per-video jitter
    for t in range(frames):
        frac = t / max(1, frames - 1) - 0.5
        cx = int(np.clip(start[0] + dx * frac * size * 0.6, 0, size - box))
        cy = int(np.clip(start[1] + dy * frac * size * 0.6, 0, size - box))
        clip[t, cy:cy + box, cx:cx + box] = colour
    return clip


def make_dummy_data(data_root: Path, num_classes: int = 6, videos_per_class: int = 8,
                    frames: int = 24, size: int = 64, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    processed = data_root / "processed"
    frames_dir, splits_dir = processed / "frames", processed / "splits"
    frames_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    classes = [f"sign{i:02d}" for i in range(num_classes)]
    label_map = {c: i for i, c in enumerate(classes)}
    rows = []
    for c in classes:
        for v in range(videos_per_class):
            video_id = f"dummy_{c}_{v:02d}"
            np.save(frames_dir / f"{video_id}.npy", fake_clip(label_map[c], num_classes, frames, size, rng))
            rows.append({"video_id": video_id, "video_path": f"Dummy/{c}/{video_id}.mov",
                         "label": c, "category": "Dummy", "label_idx": label_map[c]})
    meta = pd.DataFrame(rows)

    # Use the real splitter, so dummy data follows exactly the same rules as real data.
    splits = stratified_split(meta, val_size=0.2, test_size=0.2, seed=seed)
    meta.to_csv(splits_dir / "metadata.csv", index=False)
    for name, df in splits.items():
        df.to_csv(splits_dir / f"{name}.csv", index=False)
    (splits_dir / "label_map.json").write_text(json.dumps(label_map, indent=2))
    (splits_dir / "classes.txt").write_text("\n".join(classes) + "\n")
    return splits


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, help="Folder to write into (processed/ is created inside)")
    p.add_argument("--classes", type=int, default=6)
    p.add_argument("--videos-per-class", type=int, default=8)
    p.add_argument("--frames", type=int, default=24, help="Stored frames per video (matches frames.stored_frames)")
    p.add_argument("--size", type=int, default=64, help="Frame side in pixels (kept small: real data is 224)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fake-keypoints", action="store_true",
                   help="Accepted for the README quick start; keypoint files are Member 3's format "
                        "and are not generated here yet")
    args = p.parse_args()

    splits = make_dummy_data(Path(args.data_root), args.classes, args.videos_per_class,
                             args.frames, args.size, args.seed)
    print("Dummy data written to", Path(args.data_root) / "processed",
          "|", ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    if args.fake_keypoints:
        print("Note: --fake-keypoints is a no-op until Member 3 defines the keypoint .npz format.")
    print(f"Use it with:  ISL_DATA_ROOT={args.data_root}  and  --set frames.image_size={args.size}")


if __name__ == "__main__":
    main()

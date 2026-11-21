"""Frame extraction (src/features/extract_frames.py) on a tiny generated video."""
import cv2
import numpy as np
import pandas as pd

from src.features.extract_frames import extract_frames
from src.utils.config import load_config


def test_extract_frames_caches_and_skips(tmp_path):
    video = tmp_path / "raw" / "Cat" / "1. hello" / "a.mp4"
    video.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 90))
    for i in range(30):                                   # brightness rises over time
        writer.write(np.full((90, 160, 3), i * 8, np.uint8))
    writer.release()

    splits = tmp_path / "processed" / "splits"
    splits.mkdir(parents=True)
    row = pd.DataFrame({"video_id": ["cat_1_hello_a"], "video_path": ["Cat/1. hello/a.mp4"],
                        "label": ["hello"], "label_idx": [0]})
    row.to_csv(splits / "train.csv", index=False)
    row.iloc[:0].to_csv(splits / "val.csv", index=False)    # empty splits are fine
    row.iloc[:0].to_csv(splits / "test.csv", index=False)

    cfg = load_config("configs/data.yaml")
    cfg["paths"]["data_root"] = str(tmp_path)
    cfg["frames"].update(stored_frames=24, stored_size=32)

    assert extract_frames(cfg) == {"done": 1, "skipped": 0, "failed": 0}
    clip = np.load(tmp_path / "processed" / "frames" / "cat_1_hello_a.npy")
    assert clip.shape == (24, 32, 32, 3) and clip.dtype == np.uint8
    assert clip[0, 16, 16, 0] < clip[-1, 16, 16, 0]          # frames are in time order

    assert extract_frames(cfg) == {"done": 0, "skipped": 1, "failed": 0}   # second run reuses the cache


def test_extract_frames_reports_broken_video(tmp_path):
    splits = tmp_path / "processed" / "splits"
    splits.mkdir(parents=True)
    pd.DataFrame({"video_id": ["x"], "video_path": ["missing.mp4"], "label": ["a"], "label_idx": [0]}
                 ).to_csv(splits / "train.csv", index=False)
    for s in ("val", "test"):
        pd.DataFrame(columns=["video_id", "video_path", "label", "label_idx"]).to_csv(splits / f"{s}.csv", index=False)
    cfg = load_config("configs/data.yaml")
    cfg["paths"]["data_root"] = str(tmp_path)
    assert extract_frames(cfg)["failed"] == 1                # logged, does not crash the run

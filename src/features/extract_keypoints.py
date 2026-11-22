"""Run MediaPipe Holistic on the cached frames and save RAW landmarks: <keypoints_dir>/<video_id>.npz.

Input is ``<frames_dir>/<video_id>.npy`` (made by extract_frames.py), so the raw videos are not
needed. The 24 cached frames are spread over the whole clip and are NOT consecutive, so
MediaPipe runs in image mode (no tracking between frames). Parts that are not found are NaN;
see ``keypoint_io.py`` for the format. Existing files are skipped, so the script can be
re-run after a disconnect (use --overwrite to redo everything).

    python -m src.features.extract_keypoints
    python -m src.features.extract_keypoints --tiny
    python -m src.features.extract_keypoints --set keypoints.workers=4

Needs ``requirements-pose.txt`` (mediapipe==0.10.21). CPU is fine.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.split import read_split
from src.features.keypoint_io import PARTS, part_present, save_keypoints
from src.utils.config import add_common_args, config_from_args, resolve_paths
from src.utils.logger import get_logger

log = get_logger("extract_keypoints")

_holistic = None  # one MediaPipe model per worker process (a model must not be shared)


def _make_holistic(model_complexity: int):
    import mediapipe as mp  # imported here so the rest of the repo works without mediapipe

    # static_image_mode=True: every frame is treated as a separate picture. Our 24 frames are far
    # apart in time, so MediaPipe's frame-to-frame tracking would follow a hand that has already
    # moved somewhere else. min_detection_confidence=0.5 is MediaPipe's default.
    # refine_face_landmarks=False: we do not use the face, and refining it costs extra time.
    return mp.solutions.holistic.Holistic(
        static_image_mode=True, model_complexity=model_complexity,
        min_detection_confidence=0.5, refine_face_landmarks=False)


def _init_worker(model_complexity: int) -> None:
    # Runs once when a worker process starts. Each process builds its OWN model, because a
    # MediaPipe model cannot be sent between processes and is not meant to be shared.
    global _holistic
    _holistic = _make_holistic(model_complexity)


def _landmarks(lms, n: int, dims: int) -> np.ndarray:
    """MediaPipe landmark list -> (n, dims) array; NaN if the part was not detected."""
    if lms is None:
        return np.full((n, dims), np.nan, np.float32)   # NaN, not 0: 0 is a real coordinate (see keypoint_io)
    # Only the pose landmarks have a visibility value, so dims == 4 means "pose".
    rows = [(p.x, p.y, p.z, p.visibility) if dims == 4 else (p.x, p.y, p.z) for p in lms.landmark]
    return np.asarray(rows, np.float32)


def process_clip(frames: np.ndarray, holistic=None) -> dict[str, np.ndarray]:
    """(T, H, W, 3) uint8 RGB frames -> dict of raw landmark arrays (see keypoint_io)."""
    # The worker processes use the module-level model; KeypointExtractor passes its own.
    holistic = holistic or _holistic
    # MediaPipe keeps state between calls, so the hand landmarks of a video depended slightly
    # (0.005-0.03 of the image) on which video was processed before it. Resetting first makes
    # the result of a video independent of the history (checked in test_keypoint_extractor.py).
    holistic.reset()
    per_frame = []
    for f in frames:
        # ascontiguousarray: frames cut from a bigger array can be non-contiguous in memory,
        # and MediaPipe needs a plain contiguous image.
        res = holistic.process(np.ascontiguousarray(f))
        per_frame.append({"pose": _landmarks(res.pose_landmarks, *PARTS["pose"]),
                          "left_hand": _landmarks(res.left_hand_landmarks, *PARTS["left_hand"]),
                          "right_hand": _landmarks(res.right_hand_landmarks, *PARTS["right_hand"])})
    # From "a list of frames, each with three parts" to "three arrays, each with all frames".
    return {k: np.stack([d[k] for d in per_frame]) for k in PARTS}


class KeypointExtractor:
    """Raw video (or already-cached frames) -> raw landmarks, for the demo, the predictor and benchmarks.

    A video goes through EXACTLY the preprocessing of training (``frames.stored_frames`` evenly
    spaced frames, letterboxed to ``frames.stored_size``, see extract_frames.py), then MediaPipe.
    Using any other preprocessing would feed the model a different kind of input than it was
    trained on. Feed the result to ``keypoint_features`` and the model:

        ext = KeypointExtractor(cfg)                     # cfg = the config of the checkpoint
        raw = ext.from_video("clip.mov")                 # dict: pose / left_hand / right_hand
        x = torch.from_numpy(keypoint_features(raw, cfg))[None]      # (1, T, F)
        logits = model(x)

    Needs ``requirements-pose.txt``. Create one extractor and reuse it: loading MediaPipe is slow.
    """

    def __init__(self, cfg: dict | None = None):
        if cfg is None:
            # No config given: use the shared data settings, which are what the cached frames were made with.
            from src.utils.config import load_config
            cfg = load_config("configs/data.yaml")
        f, k = cfg["frames"], cfg["keypoints"]
        # The same three numbers extract_frames.py used for the cache. Reading them from the config
        # (instead of hard-coding 24 / 224 / "pad") keeps video and cached frames identical.
        self.num_frames = int(f["stored_frames"])
        self.size = int(f["stored_size"])
        self.mode = f.get("resize_mode", "pad")
        self._holistic = _make_holistic(int(k.get("model_complexity", 1)))

    def from_frames(self, frames: np.ndarray) -> dict[str, np.ndarray]:
        """(T, H, W, 3) uint8 RGB frames, already preprocessed like the cached frames."""
        return process_clip(frames, self._holistic)

    def from_video(self, path) -> dict[str, np.ndarray]:
        """Path of a raw video file."""
        # Imported here so that using only from_frames does not need OpenCV video code.
        from src.utils.video import sample_and_resize
        return self.from_frames(sample_and_resize(path, self.num_frames, self.size, self.mode))

    def close(self) -> None:
        # Frees MediaPipe's native resources; the ``with`` form below calls this automatically.
        self._holistic.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def extract_one(args) -> tuple[str, str, float, float, float]:
    """Process one video. Returns (video_id, status, pose rate, left hand rate, right hand rate)."""
    # One tuple argument (not four) because pool.map passes exactly one argument per job.
    video_id, frames_path, out_path, overwrite = args
    out_path = Path(out_path)
    # Skipping finished files is what makes the script safe to restart after a disconnect.
    if out_path.exists() and not overwrite:
        return video_id, "skipped", np.nan, np.nan, np.nan
    try:
        arrays = process_clip(np.load(frames_path))
        save_keypoints(out_path, arrays)
    except Exception as e:  # a broken video must not stop the whole run
        return video_id, f"{video_id}: {e}", np.nan, np.nan, np.nan
    # The detection rates are collected so the summary at the end shows how well MediaPipe did.
    rates = [float(part_present(arrays[k]).mean()) for k in ("pose", "left_hand", "right_hand")]
    return (video_id, "done", *rates)


def extract_keypoints(cfg: dict, overwrite: bool = False) -> dict:
    paths = resolve_paths(cfg)
    k = cfg["keypoints"]
    out_dir = paths["keypoints_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    # All three splits: the models need keypoints for train, val and test. read_split also applies
    # --tiny mode, so a tiny run only processes the few videos it will use.
    rows = pd.concat([read_split(paths["splits_dir"], s, cfg) for s in ("train", "val", "test")],
                     ignore_index=True)
    jobs = [(r.video_id, str(paths["frames_dir"] / f"{r.video_id}.npy"),
             str(out_dir / f"{r.video_id}.npz"), overwrite) for r in rows.itertuples(index=False)]
    workers = max(1, int(k.get("workers", 2)))
    log.info(f"Extracting keypoints for {len(jobs)} videos -> {out_dir} ({workers} workers)")
    # Processes, not threads: every worker has its own MediaPipe model (see _init_worker), and
    # MediaPipe is CPU-heavy, so several processes use several cores. chunksize=4 sends four videos
    # at a time, which saves the overhead of handing out the jobs one by one.
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(int(k.get("model_complexity", 1)),)) as pool:
        results = list(pool.map(extract_one, jobs, chunksize=4))

    errors = [r[1] for r in results if r[1] not in ("done", "skipped")]
    done = [r for r in results if r[1] == "done"]
    summary = {"done": len(done), "skipped": sum(r[1] == "skipped" for r in results), "failed": len(errors)}
    if done:  # share of frames in which each part was found, averaged over the new videos
        summary.update({name: round(float(np.mean([r[i] for r in done])), 3)
                        for i, name in ((2, "pose_rate"), (3, "left_hand_rate"), (4, "right_hand_rate"))})
    for e in errors:
        log.warning(f"FAILED {e}")
    log.info(f"Finished: {summary}")
    return summary


def main() -> None:
    # Same command-line options as every other script (--config, --set, --tiny), so it is used the same way.
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                     formatter_class=argparse.RawDescriptionHelpFormatter),
                             "configs/data.yaml")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract videos that already have a cache")
    args = parser.parse_args()
    extract_keypoints(config_from_args(args), args.overwrite)


if __name__ == "__main__":
    main()

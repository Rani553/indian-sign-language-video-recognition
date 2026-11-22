"""KeypointExtractor: raw video -> raw landmarks with the SAME preprocessing as training.

Skipped automatically when MediaPipe is not installed (it is only in requirements-pose.txt)."""
import cv2
import numpy as np
import pytest

# Everything below needs MediaPipe. importorskip skips the whole file when it is missing, so the
# normal test run (and CI, which does not install MediaPipe) stays green.
pytest.importorskip("mediapipe")

from src.features.extract_keypoints import KeypointExtractor  # noqa: E402
from src.features.keypoint_features import feature_dim, keypoint_features  # noqa: E402
from src.features.keypoint_io import PARTS  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.video import sample_and_resize  # noqa: E402


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> str:
    """A small 16:9 noise video (no person in it, so MediaPipe finds nothing)."""
    # A made-up video, so no real INCLUDE file is needed. 16:9 like the real videos, so the
    # letterbox (black bars) code is exercised too.
    path = str(tmp_path_factory.mktemp("vid") / "noise.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 180))
    rng = np.random.default_rng(0)
    for _ in range(40):
        w.write(rng.integers(0, 255, (180, 320, 3), dtype=np.uint8))
    w.release()
    return path


@pytest.fixture(scope="module")
def extractor():
    # One extractor for the whole file: loading MediaPipe is slow, so it is done once.
    with KeypointExtractor(load_config("configs/pose_lstm.yaml")) as ext:
        yield ext


def test_video_gives_the_cache_format(extractor, video):
    # The output must have exactly the format of the saved .npz files (shapes, float32),
    # because the rest of the pipeline reads both in the same way.
    raw = extractor.from_video(video)
    assert set(raw) == set(PARTS)
    for name, (n, d) in PARTS.items():
        assert raw[name].shape == (24, n, d) and raw[name].dtype == np.float32
    assert np.isnan(raw["left_hand"]).all()          # nobody in the video -> "not detected" is NaN


def test_video_uses_the_training_preprocessing(extractor, video):
    # The key promise of the class: a raw video is turned into the same 24 letterboxed 224x224
    # frames that were cached for training. If this differed, the model would be tested on
    # a different kind of input than it was trained on.
    cfg = load_config("configs/pose_lstm.yaml")
    f = cfg["frames"]
    frames = sample_and_resize(video, f["stored_frames"], f["stored_size"], f["resize_mode"])
    assert frames.shape == (24, 224, 224, 3)         # 24 letterboxed 224x224 frames, as cached
    a, b = extractor.from_video(video), extractor.from_frames(frames)
    for k in PARTS:
        assert np.array_equal(a[k], b[k], equal_nan=True)


def test_result_does_not_depend_on_previously_processed_videos(extractor, video, tmp_path):
    # MediaPipe remembers the previous video; the extractor resets it so that a video always gives
    # the same result. Note: this test is weak. The noise video contains no person, so MediaPipe
    # finds nothing and the test mostly shows that the reset call runs without error. The real
    # check (hand coordinates differed by up to 0.03 before the reset) was done by hand on real frames.
    rng = np.random.default_rng(1)
    other = rng.integers(0, 255, (24, 224, 224, 3), dtype=np.uint8)
    first = extractor.from_video(video)
    extractor.from_frames(other)                      # process something else in between
    second = extractor.from_video(video)
    for k in PARTS:
        assert np.array_equal(first[k], second[k], equal_nan=True)


def test_output_feeds_the_feature_pipeline(extractor, video):
    # The hand-over to the model: extractor output -> keypoint_features gives a finite
    # (num_frames, F) matrix, even for a video where nothing was detected (all NaN becomes zeros).
    cfg = load_config("configs/pose_lstm.yaml")
    x = keypoint_features(extractor.from_video(video), cfg)
    assert x.shape == (cfg["keypoints"]["num_frames"], feature_dim(cfg)) and np.isfinite(x).all()

"""Tests for the keypoint feature pipeline (no MediaPipe, no real data needed)."""
import numpy as np
import pytest

from src.features.keypoint_features import (
    DEFAULT_POSE_INDICES, augment_clip, feature_dim, fill_gaps, keypoint_features, mirror_clip,
    prepare_clip, resample_clip)
from src.features.keypoint_io import fake_keypoints, load_keypoints, part_present, save_keypoints

BASE = {"keypoints": {"parts": ["pose", "left_hand", "right_hand"], "hand_relative": True,
                      "use_velocity": True, "num_frames": 32}}


def cfg(**kw):
    return {"keypoints": {**BASE["keypoints"], **kw}}


def raw_clip(seed=0, cls=3, frames=24, p=0.75):
    return fake_keypoints(cls, 10, frames, np.random.default_rng(seed), hand_detect_prob=p)


# ----------------------------------------------------------------- gap filling
def test_fill_gaps_interpolates_between_detections():
    a = np.full((5, 1, 3), np.nan, np.float32)
    a[0], a[4] = 0.0, 4.0
    out = fill_gaps(a, np.array([True, False, False, False, True]))
    assert np.allclose(out[:, 0, 0], [0, 1, 2, 3, 4])


def test_fill_gaps_holds_at_the_ends_and_zeros_if_never_seen():
    a = np.full((5, 1, 3), np.nan, np.float32)
    a[2] = 7.0
    out = fill_gaps(a, np.array([False, False, True, False, False]))
    assert np.allclose(out, 7.0)
    never = fill_gaps(np.full((5, 1, 3), np.nan, np.float32), np.zeros(5, bool))
    assert np.all(never == 0) and not np.isnan(never).any()


# --------------------------------------------------------------------- features
@pytest.mark.parametrize("kw", [
    {}, {"parts": ["left_hand", "right_hand"]}, {"parts": ["pose"]}, {"use_velocity": False},
    {"hand_relative": False}, {"num_frames": 16}, {"parts": ["right_hand"], "use_velocity": False},
])
def test_feature_dim_matches_output(kw):
    c = cfg(**kw)
    f = keypoint_features(raw_clip(), c)
    assert f.shape == (c["keypoints"]["num_frames"], feature_dim(c))
    assert f.dtype == np.float32 and np.isfinite(f).all()


def test_feature_dim_value_for_default_config():
    # 7 pose points * 3 + 2 hands * (21*3 + 20*3) = 267 coordinates; x2 for velocity; + 2 flags
    assert feature_dim(BASE) == 267 * 2 + 2


def test_unknown_part_is_rejected():
    with pytest.raises(ValueError):
        feature_dim(cfg(parts=["face"]))


def test_invariant_to_signer_position_and_size():
    raw = raw_clip(seed=1)
    moved = {k: v.copy() for k, v in raw.items()}
    for k in moved:
        moved[k][..., 0] = moved[k][..., 0] * 1.7 + 0.2
        moved[k][..., 1] = moved[k][..., 1] * 1.7 - 0.1
        moved[k][..., 2] = moved[k][..., 2] * 1.7
    assert np.allclose(keypoint_features(raw, BASE), keypoint_features(moved, BASE), atol=1e-4)


def test_hand_presence_flags_follow_detection():
    raw = raw_clip(seed=2)
    f = keypoint_features(raw, cfg(num_frames=24, use_velocity=False))
    assert np.array_equal(f[:, -2] > 0.5, part_present(raw["left_hand"]))
    assert np.array_equal(f[:, -1] > 0.5, part_present(raw["right_hand"]))


def test_hand_never_detected_gives_zero_features_and_flag():
    raw = raw_clip(seed=3)
    raw["left_hand"][:] = np.nan
    f = keypoint_features(raw, cfg(parts=["left_hand"]))
    assert np.isfinite(f).all() and np.all(f == 0)


def test_pose_missing_completely_still_works():
    raw = raw_clip(seed=4)
    raw["pose"][:] = np.nan
    assert np.isfinite(keypoint_features(raw, BASE)).all()


def test_evaluation_is_deterministic():
    raw = raw_clip(seed=5)
    assert np.array_equal(keypoint_features(raw, BASE), keypoint_features(raw, BASE))


# ------------------------------------------------------------------ augmentation
def test_augmentation_changes_features_but_not_shape_and_stays_finite():
    raw = raw_clip(seed=6)
    plain = keypoint_features(raw, BASE)
    aug = keypoint_features(raw, BASE, train=True, rng=np.random.default_rng(0))
    assert aug.shape == plain.shape and np.isfinite(aug).all()
    assert not np.allclose(aug, plain)


def test_augmentation_off_equals_evaluation():
    raw = raw_clip(seed=7)
    c = cfg(augment={"enabled": False})
    assert np.array_equal(keypoint_features(raw, c, train=True, rng=np.random.default_rng(0)),
                          keypoint_features(raw, c))


def test_augmentation_is_reproducible_with_the_same_seed():
    raw = raw_clip(seed=8)
    a = keypoint_features(raw, BASE, train=True, rng=np.random.default_rng(42))
    b = keypoint_features(raw, BASE, train=True, rng=np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_hand_dropout_never_leaves_nan_and_removes_some_detections():
    raw = raw_clip(seed=9, p=1.0)       # both hands detected in every frame
    c = cfg(augment={"hand_dropout": 1.0, "scale": 0, "rotate_deg": 0, "shift": 0, "noise": 0,
                     "time_jitter": 0})
    clip = augment_clip(prepare_clip(raw, c), c, np.random.default_rng(0))
    assert (~clip["left_ok"]).any() and (~clip["right_ok"]).any()
    assert all(np.isfinite(clip[k]).all() for k in ("pose", "left", "right"))


# ------------------------------------------------------------------------ mirror
def test_mirror_twice_is_identity_and_swaps_hands():
    clip = prepare_clip(raw_clip(seed=10), BASE)
    idx = list(DEFAULT_POSE_INDICES)
    m = mirror_clip(clip, idx)
    assert np.allclose(m["left"][..., 1:], clip["right"][..., 1:])
    assert np.allclose(m["left"][..., 0], -clip["right"][..., 0])
    assert np.array_equal(m["left_ok"], clip["right_ok"])
    mm = mirror_clip(m, idx)
    for k in ("pose", "left", "right"):
        assert np.allclose(mm[k], clip[k], atol=1e-6)
    assert np.array_equal(mm["left_ok"], clip["left_ok"])


def test_mirror_swaps_left_and_right_shoulder():
    clip = prepare_clip(raw_clip(seed=11), BASE)
    idx = list(DEFAULT_POSE_INDICES)
    m = mirror_clip(clip, idx)
    assert np.allclose(m["pose"][:, idx.index(11), 0], -clip["pose"][:, idx.index(12), 0])


# ---------------------------------------------------------------------- resample
def test_resample_endpoints_and_length():
    clip = prepare_clip(raw_clip(seed=12), BASE)
    for n in (16, 24, 32):
        r = resample_clip(clip, n)
        assert r["pose"].shape[0] == n and r["left_ok"].shape == (n,)
        assert np.allclose(r["pose"][0], clip["pose"][0]) and np.allclose(r["pose"][-1], clip["pose"][-1])


# ---------------------------------------------------------------------------- io
def test_save_and_load_roundtrip(tmp_path):
    raw = raw_clip(seed=13)
    save_keypoints(tmp_path / "a.npz", raw)
    back = load_keypoints(tmp_path / "a.npz")
    for k in raw:
        assert np.array_equal(np.isnan(back[k]), np.isnan(raw[k]))
        assert np.allclose(np.nan_to_num(back[k]), np.nan_to_num(raw[k]))
    assert not list(tmp_path.glob("*.tmp.npz"))


def test_save_rejects_wrong_shape(tmp_path):
    raw = raw_clip(seed=14)
    raw["left_hand"] = raw["left_hand"][:, :10]
    with pytest.raises(ValueError):
        save_keypoints(tmp_path / "bad.npz", raw)

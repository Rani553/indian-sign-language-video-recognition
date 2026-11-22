"""Turn RAW MediaPipe landmarks (see keypoint_io.py) into the (T, F) feature matrix of the pose models.

Pipeline for one video (all steps are plain NumPy, no MediaPipe needed):

    raw landmarks (T=24, NaN = not detected)
      1. fill gaps          linear interpolation over missing frames, hold at the clip ends
      2. normalise          origin = shoulder midpoint, unit = shoulder width (median over the clip)
      3. augment            training only: scale / rotate / shift / noise / hand dropout / mirror
      4. resample in time   to ``keypoints.num_frames`` frames (random jitter when training)
      5. build features     coordinates, wrist-relative hand shape, velocity, hand-presence flags

Everything is controlled by the ``keypoints`` section of the config (parts, hand_relative,
use_velocity, num_frames) and an optional ``keypoints.augment`` section, so each ablation
(hands only, body only, no velocity, no augmentation, 16 frames, mirror) is one config change.

    feats = keypoint_features(raw, cfg)                       # evaluation: deterministic
    feats = keypoint_features(raw, cfg, train=True, rng=rng)  # training: augmented
    F = feature_dim(cfg)                                      # feats.shape == (num_frames, F)
"""
from __future__ import annotations

import numpy as np

# Upper-body MediaPipe pose landmarks: nose, shoulders, elbows, wrists. The face, hips and legs
# carry little sign information and the legs are often out of frame.
DEFAULT_POSE_INDICES = (0, 11, 12, 13, 14, 15, 16)
LEFT_RIGHT_POSE = {11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15}   # swapped when mirroring
L_SHOULDER, R_SHOULDER = 11, 12
HAND_WRIST, HAND_MIDDLE_MCP = 0, 9      # used to measure the size of a hand
VALID_PARTS = ("pose", "left_hand", "right_hand")

DEFAULT_AUGMENT = {
    "enabled": True,
    "scale": 0.10,          # random zoom: x1 +- 0.10
    "rotate_deg": 8.0,      # random rotation in the image plane
    "shift": 0.10,          # random translation, in shoulder widths
    "noise": 0.01,          # gaussian jitter on every coordinate, in shoulder widths
    "hand_dropout": 0.20,   # chance per hand and clip to lose a short span (mimics MediaPipe misses)
    "time_jitter": 0.5,     # random shift of the sampled time positions, in source frames
    "mirror": 0.0,          # chance to mirror the signer left <-> right (off: it swaps the dominant hand)
}


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def _kp_cfg(cfg: dict) -> dict:
    k = dict(cfg.get("keypoints", {}))
    parts = list(k.get("parts", VALID_PARTS))
    bad = [p for p in parts if p not in VALID_PARTS]
    if bad or not parts:
        raise ValueError(f"keypoints.parts must be a non-empty subset of {VALID_PARTS}, got {parts}")
    k["parts"] = parts
    k["pose_indices"] = list(k.get("pose_indices", DEFAULT_POSE_INDICES))
    k["hand_relative"] = bool(k.get("hand_relative", True))
    k["use_velocity"] = bool(k.get("use_velocity", True))
    k["num_frames"] = int(k.get("num_frames", 32))
    aug = dict(DEFAULT_AUGMENT)
    aug.update(k.get("augment") or {})
    k["augment"] = aug
    return k


def feature_dim(cfg: dict) -> int:
    """Number of features F per frame for this config (what the models take as input size)."""
    k = _kp_cfg(cfg)
    coords = 0
    flags = 0
    if "pose" in k["parts"]:
        coords += 3 * len(k["pose_indices"])
    for hand in ("left_hand", "right_hand"):
        if hand in k["parts"]:
            coords += 3 * 21 + (3 * 20 if k["hand_relative"] else 0)
            flags += 1
    return coords * (2 if k["use_velocity"] else 1) + flags


# --------------------------------------------------------------------------- #
# 1. Gap filling
# --------------------------------------------------------------------------- #
def fill_gaps(a: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Fill the frames where ``present`` is False. ``a`` is (T, ...) and is not modified.

    Missing frames between two detections are linearly interpolated; missing frames before the
    first / after the last detection copy the nearest detection. If the part was never detected
    the result is all zeros.
    """
    T = a.shape[0]
    flat = np.nan_to_num(a.reshape(T, -1).astype(np.float32), nan=0.0)
    idx = np.flatnonzero(present)
    if idx.size == 0:
        return np.zeros_like(a, dtype=np.float32)
    if idx.size == T:
        return a.astype(np.float32).copy()
    t = np.arange(T)
    out = np.empty_like(flat)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(t, idx, flat[idx, j])   # np.interp holds the end values
    return out.reshape(a.shape)


def _present(a: np.ndarray) -> np.ndarray:
    return ~np.isnan(a.reshape(a.shape[0], -1)).any(axis=1)


# --------------------------------------------------------------------------- #
# 2. Normalisation
# --------------------------------------------------------------------------- #
def prepare_clip(raw: dict, cfg: dict) -> dict:
    """Gap-fill and normalise raw landmarks. Returns arrays at the source frame rate:

        pose (T, P, 3)  left (T, 21, 3)  right (T, 21, 3)   normalised coordinates
        left_ok (T,)  right_ok (T,)                          True where the hand was detected
    """
    k = _kp_cfg(cfg)
    pose_raw = raw["pose"][..., :3]
    pose_ok = _present(raw["pose"])
    pose_full = fill_gaps(pose_raw, pose_ok)          # all 33 landmarks, needed for the shoulders
    left_ok, right_ok = _present(raw["left_hand"]), _present(raw["right_hand"])
    left = fill_gaps(raw["left_hand"], left_ok)
    right = fill_gaps(raw["right_hand"], right_ok)

    # origin = middle of the shoulders, unit = shoulder width. Median over time: one noisy frame
    # must not change the scale of the whole clip.
    sh = pose_full[:, [L_SHOULDER, R_SHOULDER], :2]
    origin = sh.mean(axis=1).mean(axis=0) if pose_ok.any() else np.array([0.5, 0.5], np.float32)
    width = float(np.median(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=1))) if pose_ok.any() else 0.0
    if width < 1e-3:          # pose never found (or degenerate): fall back to a typical width
        width = 0.2
    shift = np.array([origin[0], origin[1], 0.0], np.float32)

    def norm(x, ok=None):
        y = (x - shift) / width
        if ok is not None and not ok.any():
            y = np.zeros_like(y)      # never detected -> zeros, not "-origin / width"
        return y.astype(np.float32)

    return {"pose": norm(pose_full[:, k["pose_indices"]]),
            "left": norm(left, left_ok), "right": norm(right, right_ok),
            "left_ok": left_ok, "right_ok": right_ok}


# --------------------------------------------------------------------------- #
# 3. Augmentation
# --------------------------------------------------------------------------- #
def mirror_clip(clip: dict, pose_indices: list[int]) -> dict:
    """Mirror left <-> right: flip x, swap the hands, swap left/right body landmarks."""
    pose = clip["pose"].copy()
    pose[..., 0] *= -1
    order = [pose_indices.index(LEFT_RIGHT_POSE.get(i, i)) if LEFT_RIGHT_POSE.get(i, i) in pose_indices
             else pose_indices.index(i) for i in pose_indices]
    pose = pose[:, order]
    new_left, new_right = clip["right"].copy(), clip["left"].copy()
    new_left[..., 0] *= -1
    new_right[..., 0] *= -1
    return {"pose": pose, "left": new_left, "right": new_right,
            "left_ok": clip["right_ok"].copy(), "right_ok": clip["left_ok"].copy()}


def augment_clip(clip: dict, cfg: dict, rng: np.random.Generator) -> dict:
    """Random spatial changes applied to the whole clip (same draw for every frame)."""
    k = _kp_cfg(cfg)
    a = k["augment"]
    if not a["enabled"]:
        return clip
    if a["mirror"] > 0 and rng.random() < a["mirror"]:
        clip = mirror_clip(clip, k["pose_indices"])
    clip = {key: (v.copy()) for key, v in clip.items()}

    s = 1.0 + rng.uniform(-a["scale"], a["scale"])
    th = np.deg2rad(rng.uniform(-a["rotate_deg"], a["rotate_deg"]))
    rot = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]], np.float32)
    t = np.array([*rng.uniform(-a["shift"], a["shift"], 2), 0.0], np.float32)
    for name, ok in (("pose", None), ("left", clip["left_ok"]), ("right", clip["right_ok"])):
        x = (clip[name] * s) @ rot.T + t
        x = x + rng.normal(0, a["noise"], x.shape).astype(np.float32)
        if ok is not None and not ok.any():
            x = np.zeros_like(x)      # a hand that was never seen stays "absent"
        clip[name] = x.astype(np.float32)

    # Hand dropout: lose a short span of one hand, then re-interpolate it exactly like the real
    # gaps, so the model learns to cope with MediaPipe misses.
    for name in ("left", "right"):
        ok = clip[name + "_ok"]
        if ok.sum() > 3 and rng.random() < a["hand_dropout"]:
            span = int(rng.integers(2, max(3, len(ok) // 4) + 1))
            start = int(rng.integers(0, len(ok) - span + 1))
            ok = ok.copy()
            ok[start:start + span] = False
            if ok.any():
                clip[name] = fill_gaps(np.where(ok[:, None, None], clip[name], np.nan), ok)
                clip[name + "_ok"] = ok
    return clip


# --------------------------------------------------------------------------- #
# 4. Time resampling
# --------------------------------------------------------------------------- #
def resample_clip(clip: dict, num_frames: int, rng: np.random.Generator | None = None,
                  jitter: float = 0.0) -> dict:
    """Linear interpolation to ``num_frames`` frames; flags use the nearest source frame."""
    T = clip["pose"].shape[0]
    pos = np.linspace(0, T - 1, num_frames)
    if rng is not None and jitter > 0:
        pos = np.sort(np.clip(pos + rng.uniform(-jitter, jitter, num_frames), 0, T - 1))
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (pos - lo).astype(np.float32)[:, None, None]
    near = np.rint(pos).astype(int)
    out = {n: (clip[n][lo] * (1 - w) + clip[n][hi] * w).astype(np.float32) for n in ("pose", "left", "right")}
    out["left_ok"], out["right_ok"] = clip["left_ok"][near], clip["right_ok"][near]
    return out


# --------------------------------------------------------------------------- #
# 5. Feature matrix
# --------------------------------------------------------------------------- #
def _hand_relative(h: np.ndarray) -> np.ndarray:
    """Hand shape independent of where the hand is: (h - wrist) / hand size, wrist dropped."""
    size = np.linalg.norm(h[:, HAND_MIDDLE_MCP] - h[:, HAND_WRIST], axis=1)[:, None, None]
    rel = np.where(size > 1e-4, (h - h[:, HAND_WRIST:HAND_WRIST + 1]) / np.maximum(size, 1e-4), 0.0)
    return rel[:, 1:].astype(np.float32)


def clip_to_features(clip: dict, cfg: dict) -> np.ndarray:
    """(T, F) float32 from a prepared (and resampled) clip."""
    k = _kp_cfg(cfg)
    T = clip["pose"].shape[0]
    coords, flags = [], []
    if "pose" in k["parts"]:
        coords.append(clip["pose"].reshape(T, -1))
    for part, name in (("left_hand", "left"), ("right_hand", "right")):
        if part in k["parts"]:
            coords.append(clip[name].reshape(T, -1))
            if k["hand_relative"]:
                coords.append(_hand_relative(clip[name]).reshape(T, -1))
            flags.append(clip[name + "_ok"].astype(np.float32)[:, None])
    c = np.concatenate(coords, axis=1)
    parts = [c]
    if k["use_velocity"]:
        vel = np.zeros_like(c)
        vel[1:] = c[1:] - c[:-1]
        parts.append(vel)
    parts.extend(flags)
    return np.concatenate(parts, axis=1).astype(np.float32)


def keypoint_features(raw: dict, cfg: dict, train: bool = False,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Raw landmarks of one video -> (num_frames, feature_dim(cfg)) float32. No NaN in the output."""
    return features_from_prepared(prepare_clip(raw, cfg), cfg, train, rng)


def features_from_prepared(clip: dict, cfg: dict, train: bool = False,
                           rng: np.random.Generator | None = None) -> np.ndarray:
    """Steps 3-5 on a clip from ``prepare_clip``. The dataset caches the (slow, deterministic)
    steps 1-2 per video and calls this every time a video is loaded."""
    k = _kp_cfg(cfg)
    jitter = 0.0
    if train and k["augment"]["enabled"]:
        rng = rng if rng is not None else np.random.default_rng()
        clip = augment_clip(clip, cfg, rng)
        jitter = k["augment"]["time_jitter"]
    clip = resample_clip(clip, k["num_frames"], rng if train else None, jitter)
    return clip_to_features(clip, cfg)

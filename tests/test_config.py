"""Tests for the config system (src/utils/config.py)."""
from src.utils.config import apply_overrides, load_config, resolve_paths


def test_inheritance_and_overrides():
    # pose_transformer.yaml inherits pose_lstm.yaml, which inherits data.yaml.
    # Check that values from all three levels end up in the final config.
    cfg = load_config("configs/pose_transformer.yaml")
    assert cfg["model"]["name"] == "pose_transformer"
    assert cfg["input_type"] == "keypoints"          # from pose_lstm.yaml
    assert "frames" in cfg                            # from data.yaml
    # --set style overrides: a number, a list and null must be parsed correctly.
    cfg = apply_overrides(cfg, ["train.lr=1e-3", "keypoints.parts=[pose]", "dataset.class_list=null"])
    assert cfg["train"]["lr"] == 1e-3 and isinstance(cfg["train"]["lr"], float)
    assert cfg["keypoints"]["parts"] == ["pose"]
    assert cfg["dataset"]["class_list"] is None


def test_paths_follow_processed_dir(tmp_path):
    # When processed_dir points to an absolute path (like a Kaggle dataset),
    # frames/ and splits/ must move with it, while raw/ stays in data_root.
    cfg = apply_overrides(load_config("configs/data.yaml"),
                          [f"paths.data_root={tmp_path}", "paths.processed_dir=/kaggle/input/x"])
    p = resolve_paths(cfg)
    assert p["raw_dir"] == tmp_path / "raw"
    assert str(p["frames_dir"]) == "/kaggle/input/x/frames"
    assert str(p["splits_dir"]) == "/kaggle/input/x/splits"

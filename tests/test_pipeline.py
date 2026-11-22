"""The whole chain on tiny data: train -> evaluate -> benchmark -> compare -> predict from a raw video."""
import json

import cv2
import numpy as np
import pytest

from src.evaluation.benchmark_speed import benchmark_checkpoint, count_flops, dummy_input
from src.evaluation.checkpoint import load_checkpoint
from src.evaluation.compare import collect, plot_accuracy_bars, plot_accuracy_vs_latency, to_markdown
from src.evaluation.evaluate import evaluate_checkpoint
from src.inference.predictor import Predictor
from src.utils.config import resolve_paths


def _overrides(trained, kind):
    return [f"paths.data_root={trained[kind + '_cfg']['paths']['data_root']}"]


def test_training_wrote_checkpoints(trained):
    for kind in ("cnn", "pose"):
        _, _, ckpt = load_checkpoint(trained[kind], "cpu")
        assert {"model_state", "config", "label_map", "num_classes", "epoch", "best_val_acc"} <= set(ckpt)


def test_benchmark_writes_a_complete_report(trained):
    r = benchmark_checkpoint(trained["pose"], "cpu", warmup=1, runs=3, throughput_batch=4)
    saved = json.loads((trained["pose"].parent / "benchmark_cpu.json").read_text())
    assert saved["gflops"] > 0 and saved["params_m"] > 0 and saved["checkpoint_mb"] > 0
    assert saved["latency_ms"]["median"] > 0 and saved["latency_ms_1thread"]["median"] > 0
    assert saved["throughput"]["clips_per_s"] > 0 and saved["end_to_end"] is None
    assert r["input_shape"][0] == 1


def test_flop_count_matches_hand_calculation_for_a_linear_layer():
    import torch.nn as nn
    import torch
    flops = count_flops(nn.Linear(10, 5, bias=False), torch.randn(1, 10))
    assert flops == 2 * 10 * 5                                  # 2 FLOPs per multiply-add


def test_lstm_flops_are_counted(trained):
    """torch's counter ignores LSTMs; the benchmark adds them, so the pose-LSTM must not report ~0."""
    model, cfg, ckpt = load_checkpoint(trained["pose"], "cpu")
    assert count_flops(model, dummy_input(cfg, ckpt, 1)) > 2 * ckpt["input_dim"] * 16  # more than the input layer alone


def test_compare_builds_table_and_charts(trained, tmp_path):
    for kind in ("cnn", "pose"):
        evaluate_checkpoint(trained[kind], "test", "cpu", 4, _overrides(trained, kind))
        benchmark_checkpoint(trained[kind], "cpu", warmup=1, runs=3, throughput_batch=4)
    root = trained["pose"].parent.parent
    df = collect(root)
    assert set(df["experiment"]) >= {"tiny_cnn", "tiny_pose"}
    assert set(df["pipeline"]) >= {"frames", "keypoints"}
    md = to_markdown(df)
    assert "tiny_cnn" in md and "tiny_pose" in md
    plot_accuracy_vs_latency(df, tmp_path / "a.png")
    plot_accuracy_bars(df, tmp_path / "b.png")
    assert (tmp_path / "a.png").stat().st_size > 0 and (tmp_path / "b.png").stat().st_size > 0


# ---------------------------- predictor: raw video -> signs ---------------------------- #
@pytest.fixture
def fake_video(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 64))
    rng = np.random.default_rng(0)
    for _ in range(30):
        writer.write(rng.integers(0, 255, (64, 96, 3), dtype=np.uint8))
    writer.release()
    return path


def test_predictor_on_a_raw_video(trained, fake_video):
    p = Predictor(trained["cnn"], "cpu")
    top, timing = p.predict(fake_video, k=3, return_timing=True)
    assert len(top) == 3 and top[0][1] >= top[1][1] >= top[2][1]          # sorted by probability
    assert {name for name, _ in top} <= set(p.class_names)
    assert 0 < sum(prob for _, prob in top) <= 1.0 + 1e-6
    assert timing["total_ms"] >= timing["model_ms"] > 0


def test_predictor_input_equals_training_input(trained, dummy_root):
    """The predictor must feed the model exactly what the dataset feeds it for the same cached frames."""
    from src.data import build_dataset

    p = Predictor(trained["cnn"], "cpu")
    cfg = trained["cnn_cfg"]
    ds = build_dataset(cfg, "test", train=False)
    video_id = ds.df.iloc[0]["video_id"]
    frames = np.load(resolve_paths(cfg)["frames_dir"] / f"{video_id}.npy")
    x_predictor = p.features(frames)[0]
    x_dataset = ds[0][0]
    assert x_predictor.shape == x_dataset.shape
    assert np.allclose(x_predictor.numpy(), x_dataset.numpy(), atol=1e-6)

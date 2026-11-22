"""Metrics, checkpoint loading and evaluation of a trained model."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation import metrics as M
from src.evaluation.checkpoint import class_names_from, load_checkpoint
from src.evaluation.evaluate import evaluate_checkpoint, expand_checkpoints


# ---------------------------- metrics (no model needed) ---------------------------- #
def test_perfect_predictions():
    y = np.array([0, 1, 2, 0, 1, 2])
    probs = np.eye(3)[y]
    r = M.compute_metrics(y, probs)
    assert r["top1"] == 1.0 and r["top5"] == 1.0 and r["macro_f1"] == 1.0 and r["correct"] == 6


def test_known_confusion_and_f1():
    y = np.array([0, 0, 1, 1])
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.3, 0.7]])   # predictions 0, 1, 1, 1
    r = M.compute_metrics(y, probs)
    assert r["top1"] == 0.75
    cm = M.confusion(y, probs.argmax(1), 2)
    assert cm.tolist() == [[1, 1], [0, 2]]
    assert M.top_confusions(cm, ["a", "b"]) == [{"true": "a", "predicted": "b", "count": 1}]


def test_topk_counts_runner_up():
    probs = np.array([[0.5, 0.4, 0.1]])
    assert M.topk_accuracy(np.array([1]), probs, 1) == 0.0
    assert M.topk_accuracy(np.array([1]), probs, 2) == 1.0


def test_wilson_interval_contains_the_estimate_and_shrinks_with_n():
    lo, hi = M.wilson_interval(128, 137)
    assert lo < 128 / 137 < hi
    lo2, hi2 = M.wilson_interval(1280, 1370)
    assert (hi2 - lo2) < (hi - lo)
    assert M.wilson_interval(0, 0) == (0.0, 0.0)


def test_per_class_table_is_sorted_worst_first():
    y, p = np.array([0, 0, 1, 1]), np.array([0, 0, 1, 0])
    t = M.per_class_table(y, p, ["a", "b"])
    assert t.iloc[0]["class"] == "b" and t["support"].sum() == 4


# ---------------------------- with a trained model ---------------------------- #
@pytest.mark.parametrize("kind", ["cnn", "pose"])
def test_load_checkpoint_roundtrip(trained, kind):
    model, cfg, ckpt = load_checkpoint(trained[kind], "cpu")
    assert not model.training                                    # eval mode
    assert cfg["model"].get("pretrained", False) is False        # never downloads weights
    assert len(class_names_from(ckpt["label_map"])) == ckpt["num_classes"] == 4


def test_load_checkpoint_rejects_foreign_files(tmp_path):
    bad = tmp_path / "bad.pt"
    torch.save({"weights": 1}, bad)
    with pytest.raises(ValueError, match="not a checkpoint"):
        load_checkpoint(bad)
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt")


@pytest.mark.parametrize("kind", ["cnn", "pose"])
def test_evaluate_writes_all_files_and_consistent_numbers(trained, kind):
    r = evaluate_checkpoint(trained[kind], "test", "cpu", batch_size=4,
                            overrides=[f"paths.data_root={trained[kind + '_cfg']['paths']['data_root']}"])
    out = trained[kind].parent / "eval_test"
    for f in ("metrics.json", "per_class.csv", "predictions.csv", "confusion_matrix.png", "top_confusions.json"):
        assert (out / f).exists(), f
    saved = json.loads((out / "metrics.json").read_text())
    preds = pd.read_csv(out / "predictions.csv")
    assert saved["n"] == len(preds) == 8                         # 4 classes x 2 test videos
    assert saved["correct"] == int(preds["correct"].sum())       # the CSV and the metric agree
    assert 0.0 <= r["top1"] <= 1.0 and r["top5"] == 1.0          # only 4 classes: top-5 is always right
    lo, hi = saved["top1_ci95"]
    assert lo <= r["top1"] <= hi


def test_evaluation_is_repeatable(trained):
    kw = dict(split="test", device="cpu", batch_size=4,
              overrides=[f"paths.data_root={trained['pose_cfg']['paths']['data_root']}"])
    a = evaluate_checkpoint(trained["pose"], **kw)
    b = evaluate_checkpoint(trained["pose"], **kw)
    assert a["top1"] == b["top1"] and a["macro_f1"] == b["macro_f1"]


def test_expand_checkpoints_handles_globs(trained):
    root = trained["pose"].parent.parent
    found = expand_checkpoints([str(root / "*" / "best.pt")])
    assert trained["pose"] in found and trained["cnn"] in found

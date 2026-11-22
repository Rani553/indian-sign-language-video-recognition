"""Speed and size benchmark of trained checkpoints. Run every model on the SAME machine.

    python -m src.evaluation.benchmark_speed --checkpoint results/pose_transformer/best.pt --device cpu
    python -m src.evaluation.benchmark_speed --checkpoint "results/*/best.pt" --device cpu cuda
    python -m src.evaluation.benchmark_speed --checkpoint "results/*/best.pt" --device cpu --video clip.mov

For every checkpoint and device it writes ``benchmark_<device>.json`` next to the checkpoint:

    params_m, checkpoint_mb      model size
    gflops                       compute for ONE clip (2 FLOPs per multiply-add)
    latency_ms                   model only, batch 1: mean / median / p95 (after warm-up)
    latency_ms_1thread           the same on CPU with ONE thread (stable; best for comparing models)
    throughput_clips_per_s       model only, batch of --throughput-batch clips
    end_to_end                   (needs --video) raw video file -> prediction, split into
                                 decode+resize / features (CNN input or MediaPipe+features) / model

"Model only" starts from the cached input (frames or keypoints). "End to end" starts from a raw video,
which is what a user of the demo pays for. For the pose models the MediaPipe step dominates it, so
the two numbers tell very different stories and both are reported.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.checkpoint import experiment_name, load_checkpoint
from src.evaluation.evaluate import expand_checkpoints
from src.models import count_parameters
from src.utils.logger import get_logger

log = get_logger("benchmark")


# --------------------------------------------------------------------------- #
# Input, size and compute
# --------------------------------------------------------------------------- #
def dummy_input(cfg: dict, ckpt: dict, batch: int = 1) -> torch.Tensor:
    """A random batch with the exact shape the model gets in training (values do not matter for speed)."""
    if cfg.get("input_type", "frames") == "frames":
        f = cfg["frames"]
        return torch.randn(batch, int(f["num_frames"]), 3, int(f["image_size"]), int(f["image_size"]))
    return torch.randn(batch, int(cfg["keypoints"]["num_frames"]), int(ckpt["input_dim"]))


def _lstm_flops(module: nn.LSTM, x: torch.Tensor) -> float:
    """FLOPs of an LSTM over the sequence ``x`` (B, T, I). torch's FlopCounterMode does not count LSTMs.

    Per layer, direction and time step: 4 gates, each a matrix product with the input and the hidden
    state: 2 * 4 * H * (I + H) FLOPs. Element-wise gate maths is ignored (it is tiny).
    """
    b, t, in_size = x.shape
    h, dirs, total = module.hidden_size, 2 if module.bidirectional else 1, 0.0
    for layer in range(module.num_layers):
        i = in_size if layer == 0 else h * dirs
        total += b * t * dirs * 2 * 4 * h * (i + h)
    return total


def count_flops(model: nn.Module, x: torch.Tensor) -> float:
    """Total FLOPs of one forward pass (matrix products, convolutions, attention, LSTMs)."""
    from torch.utils.flop_counter import FlopCounterMode

    # A copy in train mode: in eval mode PyTorch uses a fused transformer kernel that the counter
    # cannot see. Dropout does not change the FLOPs. The original model is left untouched.
    m = copy.deepcopy(model).cpu().train()
    lstm_flops = [0.0]
    hooks = [mod.register_forward_hook(lambda mod_, inp, out: lstm_flops.__setitem__(0, lstm_flops[0] + _lstm_flops(mod_, inp[0])))
             for mod in m.modules() if isinstance(mod, nn.LSTM)]
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        m(x.cpu())
    for h in hooks:
        h.remove()
    return float(counter.get_total_flops() + lstm_flops[0])


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def measure_latency(model: nn.Module, x: torch.Tensor, device: torch.device,
                    warmup: int = 10, runs: int = 50) -> dict:
    """Time of one forward pass in milliseconds. Warm-up runs first (allocation, kernel selection)."""
    x = x.to(device)
    for _ in range(warmup):
        model(x)
    _sync(device)
    times = []
    for _ in range(runs):
        _sync(device)
        t0 = time.perf_counter()
        model(x)
        _sync(device)
        times.append((time.perf_counter() - t0) * 1e3)
    times_arr = np.array(times)
    return {"mean": float(times_arr.mean()), "median": float(np.median(times_arr)),
            "p95": float(np.percentile(times_arr, 95)),
            "std": float(times_arr.std()), "min": float(times_arr.min()), "runs": runs}


def measure_throughput(model: nn.Module, cfg: dict, ckpt: dict, device: torch.device,
                       batch: int, warmup: int = 3, runs: int = 10) -> dict:
    """Clips per second with a batch of ``batch`` clips. Halves the batch if the GPU runs out of memory."""
    while batch >= 1:
        try:
            x = dummy_input(cfg, ckpt, batch)
            lat = measure_latency(model, x, device, warmup, runs)
            return {"batch_size": batch, "clips_per_s": batch / (lat["mean"] / 1e3)}
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch //= 2
    return {"batch_size": 0, "clips_per_s": float("nan")}


def measure_end_to_end(checkpoint: Path, device: str, video: Path, runs: int) -> dict:
    """Raw video -> prediction, averaged over ``runs`` calls after one warm-up call."""
    from src.inference.predictor import Predictor

    try:
        pred = Predictor(checkpoint, device)
        pred.predict(video)   # warm-up: loads MediaPipe / fills caches, not counted
        timings = [pred.predict(video, return_timing=True)[1] for _ in range(runs)]
    except RuntimeError as e:   # e.g. MediaPipe not installed for a pose model
        return {"error": str(e)}
    keys = timings[0].keys()
    out = {k: float(statistics.mean(t[k] for t in timings)) for k in keys}
    out.update({"runs": runs, "video": video.name})
    return out


def hardware_info(device: torch.device) -> dict:
    info = {"platform": platform.platform(), "processor": platform.processor(),
            "python": platform.python_version(), "torch": torch.__version__,
            "cpu_threads": torch.get_num_threads()}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info


# --------------------------------------------------------------------------- #
def benchmark_checkpoint(path: str | Path, device_name: str, warmup: int = 10, runs: int = 50,
                         throughput_batch: int = 16, video: str | Path | None = None,
                         e2e_runs: int = 5) -> dict:
    path = Path(path)
    device = torch.device(device_name)
    model, cfg, ckpt = load_checkpoint(path, device)
    x1 = dummy_input(cfg, ckpt, 1)
    lat_1thread = None
    if device.type == "cpu":
        # Tiny models get SLOWER with more threads (thread start-up costs more than the work), and the
        # time then depends on what else the machine does. One thread gives a stable, comparable number.
        n_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        lat_1thread = measure_latency(model, x1, device, warmup, runs)
        torch.set_num_threads(n_threads)
    res = {
        "experiment": experiment_name(path, cfg),
        "model": cfg["model"]["name"],
        "input_type": cfg.get("input_type"),
        "device": device_name,
        "params_m": count_parameters(model) / 1e6,
        "checkpoint_mb": path.stat().st_size / 1e6,
        "input_shape": list(x1.shape),
        "gflops": count_flops(model, x1) / 1e9,
        "latency_ms": measure_latency(model, x1, device, warmup, runs),
        "latency_ms_1thread": lat_1thread,   # CPU only: the stable number to compare models with
        "throughput": measure_throughput(model, cfg, ckpt, device, throughput_batch),
        "end_to_end": measure_end_to_end(path, device_name, Path(video), e2e_runs) if video else None,
        "hardware": hardware_info(device),
    }
    (path.parent / f"benchmark_{device_name}.json").write_text(json.dumps(res, indent=2))
    e2e = res["end_to_end"]
    e2e_txt = (f" | end-to-end {e2e['total_ms']:.0f} ms" if e2e and "total_ms" in e2e else
               f" | end-to-end unavailable: {e2e['error'][:60]}" if e2e else "")
    log.info(f"{res['experiment']} [{device_name}] {res['params_m']:.2f}M params | {res['gflops']:.3f} GFLOPs | "
             f"latency {res['latency_ms']['median']:.1f} ms | {res['throughput']['clips_per_s']:.0f} clips/s{e2e_txt}")
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", nargs="+", required=True, help="best.pt file(s); globs are allowed")
    p.add_argument("--device", nargs="+", default=["cpu"], help="cpu and/or cuda")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--throughput-batch", type=int, default=16)
    p.add_argument("--video", default=None, help="a raw video file: also measure raw video -> prediction")
    p.add_argument("--e2e-runs", type=int, default=5)
    args = p.parse_args()

    for dev in args.device:
        if dev == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA is not available, skipping --device cuda")
            continue
        for ckpt in expand_checkpoints(args.checkpoint):
            try:
                benchmark_checkpoint(ckpt, dev, args.warmup, args.runs, args.throughput_batch,
                                     args.video, args.e2e_runs)
            except Exception as e:
                log.warning(f"FAILED {ckpt} [{dev}]: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

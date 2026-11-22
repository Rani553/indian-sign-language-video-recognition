"""The main result of the project: accuracy versus speed of Pipeline A (frames) and Pipeline B (keypoints).

    python -m src.evaluation.compare
    python -m src.evaluation.compare --experiments cnn_lstm_mobilenet_v2 pose_transformer pose_lstm
    python -m src.evaluation.compare --results-root /content/drive/MyDrive/isl_results

Reads, for every experiment folder in the results root (run evaluate.py and benchmark_speed.py first):
    eval_<split>/metrics.json      accuracy, F1, 95 % interval
    benchmark_cpu.json, benchmark_cuda.json

Writes ``<results_root>/comparison/``:
    comparison.csv, comparison.md     one row per model
    accuracy_vs_latency.png           test accuracy against time per clip (left: model only, right: end to end)
    accuracy_bars.png                 test accuracy with 95 % intervals

Only experiments that were evaluated appear, so the ablation runs (validation only) stay out unless
you evaluate them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_config, resolve_paths
from src.utils.logger import get_logger

log = get_logger("compare")

# One colour per PIPELINE (the entity), the same in every chart. Colours 1 and 2 of the validated palette.
PIPELINES = {"frames": ("Pipeline A: raw frames", "#2a78d6"),
             "keypoints": ("Pipeline B: pose keypoints", "#eb6834")}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def collect(results_root: Path, split: str = "test", experiments: list[str] | None = None) -> pd.DataFrame:
    """One row per evaluated experiment, joining its accuracy and its speed numbers."""
    rows = []
    for d in sorted(p for p in results_root.iterdir() if p.is_dir()):
        if experiments and d.name not in experiments:
            continue
        m = _read_json(d / f"eval_{split}" / "metrics.json")
        if m is None:
            continue
        cpu, cuda = _read_json(d / "benchmark_cpu.json"), _read_json(d / "benchmark_cuda.json")
        any_b = cpu or cuda or {}

        def lat(b, key="latency_ms"):
            return b[key]["median"] if b and b.get(key) else np.nan

        def e2e(b):
            return b["end_to_end"]["total_ms"] if b and b.get("end_to_end") and "total_ms" in b["end_to_end"] else np.nan

        rows.append({
            "experiment": d.name, "model": m["model"], "pipeline": m["input_type"],
            "params_m": m["params_m"], "checkpoint_mb": any_b.get("checkpoint_mb", np.nan),
            "gflops": any_b.get("gflops", np.nan),
            "top1": m["top1"], "ci_lo": m["top1_ci95"][0], "ci_hi": m["top1_ci95"][1],
            "top5": m["top5"], "macro_f1": m["macro_f1"], "n_test": m["n"],
            "cpu_ms_1thread": lat(cpu, "latency_ms_1thread"), "cpu_ms": lat(cpu), "cuda_ms": lat(cuda),
            "e2e_cpu_ms": e2e(cpu), "e2e_cuda_ms": e2e(cuda),
        })
    if not rows:
        raise SystemExit(f"No evaluated experiments in {results_root}. Run src.evaluation.evaluate first.")
    return pd.DataFrame(rows).sort_values("top1", ascending=False).reset_index(drop=True)


def to_markdown(df: pd.DataFrame) -> str:
    from tabulate import tabulate

    t = pd.DataFrame({
        "Model": df["experiment"],
        "Pipeline": df["pipeline"].map(lambda p: PIPELINES.get(p, (p,))[0]),
        "Params (M)": df["params_m"], "GFLOPs": df["gflops"],
        "Top-1": df["top1"], "95% CI": [f"{lo:.3f}-{hi:.3f}" for lo, hi in zip(df["ci_lo"], df["ci_hi"])],
        "Top-5": df["top5"], "Macro-F1": df["macro_f1"],
        "CPU ms (1 thread)": df["cpu_ms_1thread"], "GPU ms": df["cuda_ms"],
        "End-to-end CPU ms": df["e2e_cpu_ms"],
    })
    t = t.astype(object).where(t.notna(), None)   # tabulate prints None (not NaN) as the missing value
    return tabulate(t, headers="keys", tablefmt="github", showindex=False, floatfmt=".3f", missingval="-")


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _legend(ax, df) -> None:
    import matplotlib.lines as mlines

    handles = [mlines.Line2D([], [], marker="o", linestyle="", markersize=8, color=color,
                             markeredgecolor=SURFACE, label=name)
               for key, (name, color) in PIPELINES.items() if key in set(df["pipeline"])]
    if len(handles) >= 2:
        ax.legend(handles=handles, frameon=False, labelcolor=INK_2, fontsize=9, loc="lower right")


def plot_accuracy_vs_latency(df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    panels = [("cpu_ms_1thread", "Model only, CPU (1 thread), ms per clip"),
              ("e2e_cpu_ms", "Raw video to prediction, CPU, ms per clip")]
    panels = [(c, t) for c, t in panels if df[c].notna().any()] or [("cpu_ms_1thread", panels[0][1])]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 4.6), facecolor=SURFACE, squeeze=False)
    for ax, (col, xlabel) in zip(axes[0], panels):
        _style(ax)
        d = df[df[col].notna()]
        for _, r in d.iterrows():
            color = PIPELINES.get(r["pipeline"], ("", INK_2))[1]
            ax.errorbar(r[col], r["top1"], yerr=[[r["top1"] - r["ci_lo"]], [r["ci_hi"] - r["top1"]]],
                        fmt="o", color=color, ecolor=color, elinewidth=1.2, capsize=0, markersize=9,
                        markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
            ax.annotate(r["experiment"], (r[col], r["top1"]), textcoords="offset points", xytext=(8, 7),
                        fontsize=8.5, color=INK)
        ax.set_xscale("log")
        # Room around the points (at least a factor 4 end to end), so a log axis always has readable ticks.
        ax.set_xlim(d[col].min() / 2, d[col].max() * 2)
        # Plain numbers (3, 5, 10, 20 ...) instead of matplotlib's default 10^x labels.
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1, 2, 3, 5), numticks=20))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
        ax.set_ylabel("Top-1 test accuracy", color=INK_2, fontsize=10)
        ax.set_ylim(max(0.0, d["ci_lo"].min() - 0.05), min(1.0, d["ci_hi"].max() + 0.04))
        _legend(ax, d)
    fig.suptitle("Accuracy against speed (bars: 95% interval; further left is faster)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def plot_accuracy_bars(df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.sort_values("top1")
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(d) + 1.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    colors = [PIPELINES.get(p, ("", INK_2))[1] for p in d["pipeline"]]
    y = np.arange(len(d))
    ax.barh(y, d["top1"], height=0.55, color=colors, zorder=3)
    ax.errorbar(d["top1"], y, xerr=[d["top1"] - d["ci_lo"], d["ci_hi"] - d["top1"]], fmt="none",
                ecolor=INK_2, elinewidth=1.2, capsize=0, zorder=4)
    for yi, (v, hi) in enumerate(zip(d["top1"], d["ci_hi"])):
        ax.text(min(hi, 0.995) + 0.008, yi, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(d["experiment"], fontsize=9, color=INK)
    ax.set_xlim(max(0.0, d["ci_lo"].min() - 0.1), 1.06)
    ax.set_xlabel("Top-1 test accuracy (bars: 95% interval)", color=INK_2, fontsize=10)
    _legend(ax, d)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default=None, help="default: the results root of configs/data.yaml")
    p.add_argument("--split", default="test")
    p.add_argument("--experiments", nargs="*", default=None, help="only these experiment folders")
    args = p.parse_args()

    root = Path(args.results_root) if args.results_root else resolve_paths(load_config("configs/data.yaml"))["results_root"]
    df = collect(root, args.split, args.experiments)
    out = root / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "comparison.csv", index=False)
    md = to_markdown(df)
    (out / "comparison.md").write_text(md + "\n", encoding="utf-8")
    plot_accuracy_vs_latency(df, out / "accuracy_vs_latency.png")
    plot_accuracy_bars(df, out / "accuracy_bars.png")
    print(md)
    log.info(f"Wrote {out}")


if __name__ == "__main__":
    main()

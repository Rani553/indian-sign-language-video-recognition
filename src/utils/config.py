"""Configuration handling.

* YAML configs can inherit from another config with a ``_base_`` key.
* Any value can be overridden from the command line with ``--set a.b=value``.
* Paths are resolved at runtime, so the same config works on Colab, Kaggle
  and a local machine (the checkpoint stores the *unresolved* config).
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any

import yaml

# Root folder of the repository (this file is src/utils/config.py, so go up 2 levels).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Loading / merging
# --------------------------------------------------------------------------- #
def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)  # copy, so the original base config is never changed
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            # Both sides are sections (dicts): merge them key by key.
            out[key] = deep_merge(out[key], value)
        else:
            # Plain value (number, string, list ...): the override wins.
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict:
    """Load a YAML config, following ``_base_`` inheritance."""
    path = Path(path)
    # Allow paths relative to the project root, e.g. "configs/data.yaml".
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # If this config says "_base_: data.yaml", load that file first and put
    # this file's values on top of it (works for chains of several files).
    base_name = cfg.pop("_base_", None)
    if base_name:
        base_cfg = load_config(path.parent / base_name)
        cfg = deep_merge(base_cfg, cfg)
    return cfg


def set_by_dotted_key(cfg: dict, dotted_key: str, value: Any) -> None:
    """Set a nested value, e.g. "train.lr" -> cfg["train"]["lr"] = value."""
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})  # create missing sections on the way
    node[keys[-1]] = value


def get_by_dotted_key(cfg: dict, dotted_key: str, default: Any = None) -> Any:
    """Read a nested value, e.g. "train.lr"; returns ``default`` if it does not exist."""
    node: Any = cfg
    for k in dotted_key.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def _parse_value(raw: str) -> Any:
    """YAML parsing, plus numbers like ``1e-3`` that YAML 1.1 reads as strings."""
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        # YAML reads "1e-3" as text; try to turn such text into a number.
        try:
            return float(value)
        except ValueError:
            return value
    return value


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply ``key=value`` overrides. Values are parsed with YAML (so numbers,
    booleans, lists and null work: ``train.lr=1e-3``, ``model.parts=[pose]``)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item}")
        key, raw = item.split("=", 1)  # split only on the first "=" (values may contain "=")
        set_by_dotted_key(cfg, key.strip(), _parse_value(raw))
    return cfg


def add_common_args(parser: argparse.ArgumentParser, default_config: str | None = None):
    """Add the command-line options every script shares: --config, --set and --tiny."""
    parser.add_argument("--config", default=default_config, help="Path to YAML config")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="Override config values, e.g. --set train.epochs=5 paths.data_root=/content/data",
    )
    parser.add_argument("--tiny", action="store_true",
                        help="Tiny mode: 5 classes, few videos, 2 epochs (quick CPU check)")
    return parser


def config_from_args(args) -> dict:
    """Build the final config: load the YAML file, apply --set overrides, then --tiny."""
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set)
    if getattr(args, "tiny", False):
        cfg = apply_tiny_mode(cfg)
    return cfg


def apply_tiny_mode(cfg: dict) -> dict:
    """Shrink a run so it finishes in a minute on CPU (for quick checks, not results)."""
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("data", {})["tiny"] = True   # read_split() then keeps only a few classes/videos
    train = cfg.setdefault("train", {})
    train["epochs"] = min(int(train.get("epochs", 2)), 2)
    train["num_workers"] = 0    # no worker processes: simpler and safer on CPU
    train["amp"] = False        # mixed precision only helps on GPU
    # Separate folder name so a tiny run never overwrites a real experiment.
    cfg["experiment"] = str(cfg.get("experiment", "exp")) + "_tiny"
    return cfg


# --------------------------------------------------------------------------- #
# Environment / paths
# --------------------------------------------------------------------------- #
def detect_environment() -> str:
    """Return "kaggle", "colab" or "local" depending on where the code is running."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").exists():
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or Path("/content").exists():
        return "colab"
    return "local"


def _default_data_root(env: str) -> Path:
    """Default folder for the dataset in each environment."""
    if env == "kaggle":
        return Path("/kaggle/working/isl_data")
    if env == "colab":
        # Prefer Google Drive (survives disconnects); fall back to the temporary disk.
        drive = Path("/content/drive/MyDrive")
        return drive / "isl_data" if drive.exists() else Path("/content/isl_data")
    return PROJECT_ROOT / "data"


def _default_results_root(env: str) -> Path:
    """Default folder for checkpoints, logs and plots in each environment."""
    if env == "kaggle":
        return Path("/kaggle/working/results")
    if env == "colab":
        drive = Path("/content/drive/MyDrive")
        return drive / "isl_results" if drive.exists() else Path("/content/results")
    return PROJECT_ROOT / "results"


def resolve_paths(cfg: dict) -> dict:
    """Return a dict of absolute ``Path`` objects for every entry in ``cfg['paths']``.

    Priority for the roots: explicit config value > environment variable
    (``ISL_DATA_ROOT`` / ``ISL_RESULTS_ROOT``) > auto-detected default.
    Relative sub-paths are joined to ``data_root``; absolute ones are kept, which
    lets Kaggle users point ``processed_dir`` at a read-only ``/kaggle/input/...``.
    """
    paths_cfg = dict(cfg.get("paths", {}))  # copy, because we pop keys below
    env = detect_environment()

    # Step 1: decide the two root folders.
    data_root = paths_cfg.pop("data_root", "auto")
    if data_root in (None, "auto"):
        data_root = os.environ.get("ISL_DATA_ROOT") or _default_data_root(env)
    results_root = paths_cfg.pop("results_root", "auto")
    if results_root in (None, "auto"):
        results_root = os.environ.get("ISL_RESULTS_ROOT") or _default_results_root(env)

    resolved = {"data_root": Path(data_root).expanduser(),
                "results_root": Path(results_root).expanduser()}

    def _resolve(value, parent: Path) -> Path:
        # Absolute path -> keep it; relative path -> put it inside ``parent``.
        p = Path(str(value)).expanduser()
        return p if p.is_absolute() else parent / p

    # Step 2: raw videos and processed data live inside data_root.
    resolved["raw_dir"] = _resolve(paths_cfg.pop("raw_dir", "raw"), resolved["data_root"])
    resolved["processed_dir"] = _resolve(paths_cfg.pop("processed_dir", "processed"),
                                         resolved["data_root"])
    # Sub-folders of processed data are relative to processed_dir, so pointing
    # processed_dir at an attached Kaggle dataset moves all of them together.
    for key, default in (("frames_dir", "frames"), ("keypoints_dir", "keypoints"),
                         ("splits_dir", "splits")):
        resolved[key] = _resolve(paths_cfg.pop(key, default), resolved["processed_dir"])
    for key, value in paths_cfg.items():  # any extra user-defined path
        resolved[key] = _resolve(value, resolved["data_root"])
    return resolved


def experiment_dir(cfg: dict) -> Path:
    """Folder for one experiment's outputs: <results_root>/<experiment name>/ (created if missing)."""
    paths = resolve_paths(cfg)
    out = paths["results_root"] / str(cfg.get("experiment", "experiment"))
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_config(cfg: dict, path: str | Path) -> None:
    """Save the final config next to the results, so every run can be reproduced."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

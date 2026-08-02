"""YAML config loading, dotted overrides, and global seeding."""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config, following an optional ``inherit`` key.

    ``inherit`` names another YAML file relative to the current one; the
    child's values win. ``overrides`` are ``a.b.c=value`` strings (values
    parsed as YAML, so ``lr=1e-4`` and ``max_steps=null`` work).
    """
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _deep_merge(load_config(path.parent / parent), cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override {item!r} is not of the form key=value")
        _set_dotted(cfg, key.strip(), _parse_value(raw))
    return cfg


def _parse_value(raw: str) -> Any:
    value = yaml.safe_load(raw)
    # YAML 1.1 reads "1e-4" as a string (it wants "1.0e-4"); accept the
    # common scientific notation on the command line anyway.
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
    return value


def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot descend into non-dict at {key!r} in {dotted!r}")
    node[keys[-1]] = value


def get_path(cfg: dict, key: str, default: str) -> Path:
    """Resolve a path from ``cfg['paths'][key]``, defaulting sensibly.

    Relative paths are resolved against TURKISH_ASR_ROOT if set, else CWD,
    so nothing is hardcoded to a machine-specific location.
    """
    raw = str(cfg.get("paths", {}).get(key, default))
    root = Path(os.environ.get("TURKISH_ASR_ROOT", "."))
    p = Path(raw)
    return p if p.is_absolute() else root / p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

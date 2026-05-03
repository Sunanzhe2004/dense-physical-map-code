from __future__ import annotations

from pathlib import Path
from typing import Any

from common.io.json_io import load_json

REQUIRED_CONFIG_KEYS = ("dataset", "model", "train", "eval", "output_dir", "seed")


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    config = load_json(path)
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"missing config keys: {missing}")
    return config

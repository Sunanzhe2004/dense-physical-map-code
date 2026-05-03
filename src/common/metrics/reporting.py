from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from common.io.json_io import load_json, save_json


def default_metrics(exp_group: str, run_name: str) -> dict[str, Any]:
    return {
        "exp_group": exp_group,
        "run_name": run_name,
        "num_samples": 1,
        "mae": 0.0,
        "rmse": 0.0,
    }


def write_metrics(output_dir: str | Path, metrics: dict[str, Any]) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_json(output_path / "metrics.json", metrics)
    with (output_path / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def collect_metrics(output_root: str | Path) -> list[dict[str, Any]]:
    metrics_list: list[dict[str, Any]] = []
    for metrics_file in Path(output_root).rglob("metrics.json"):
        metrics_list.append(load_json(metrics_file))
    return sorted(metrics_list, key=lambda item: (item["exp_group"], item["run_name"]))

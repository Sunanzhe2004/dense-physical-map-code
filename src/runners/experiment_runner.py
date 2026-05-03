from __future__ import annotations

from pathlib import Path
from typing import Any

from common.metrics.reporting import default_metrics, write_metrics


def execute_experiment(
    exp_group: str,
    config: dict[str, Any],
    stage: str,
    extra_metrics: dict[str, Any] | None = None,
) -> Path:
    run_name = config["model"]["name"]
    output_dir = Path(config["output_dir"]) / exp_group / run_name / stage
    metrics = default_metrics(exp_group=exp_group, run_name=run_name)
    metrics["stage"] = stage
    metrics["seed"] = config["seed"]
    if extra_metrics:
        metrics.update(extra_metrics)
    write_metrics(output_dir, metrics)
    return output_dir

from __future__ import annotations

from typing import Any


def convert_dataset_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["sample_id"],
        "image": record["image_path"],
        "targets": record["physical_properties"],
    }


def convert_baseline_metrics(raw_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "mae": float(raw_metrics.get("mae", 0.0)),
        "rmse": float(raw_metrics.get("rmse", 0.0)),
    }

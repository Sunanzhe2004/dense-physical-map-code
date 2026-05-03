from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.io.json_io import load_json
from common.metrics.reporting import collect_metrics
from experiments.comparison.adapters.baseline_adapter import (
    convert_baseline_metrics,
    convert_dataset_record,
)
from runners.experiment_runner import execute_experiment


class ComparisonAdapterTests(unittest.TestCase):
    def test_adapter_converts_record_and_metrics(self) -> None:
        record = load_json(ROOT / "annotation/examples/demo_annotation.json")
        converted = convert_dataset_record(record)
        metrics = convert_baseline_metrics({"mae": 0.2, "rmse": 0.3})
        self.assertEqual(converted["id"], "sample_0001")
        self.assertEqual(metrics["rmse"], 0.3)

    def test_metrics_can_be_collected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "dataset": {},
                "model": {"name": "baseline_adapter_demo"},
                "train": {},
                "eval": {},
                "output_dir": temp_dir,
                "seed": 0,
            }
            execute_experiment("comparison", config, "eval")
            metrics = collect_metrics(temp_dir)
            self.assertEqual(len(metrics), 1)


if __name__ == "__main__":
    unittest.main()

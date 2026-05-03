from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.utils.config import load_experiment_config
from runners.experiment_runner import execute_experiment


class MainConfigTests(unittest.TestCase):
    def test_main_config_loads_and_runs(self) -> None:
        config = load_experiment_config(ROOT / "experiments/main/configs/demo_main.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            config["output_dir"] = temp_dir
            output_dir = execute_experiment("main", config, "train")
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "summary.csv").exists())


if __name__ == "__main__":
    unittest.main()

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


class AblationConfigTests(unittest.TestCase):
    def test_ablation_config_loads_and_variant_name_is_preserved(self) -> None:
        config = load_experiment_config(ROOT / "experiments/ablation/configs/demo_ablation.json")
        self.assertIn("remove_depth_guidance", config["model"]["name"])
        with tempfile.TemporaryDirectory() as temp_dir:
            config["output_dir"] = temp_dir
            output_dir = execute_experiment("ablation", config, "eval")
            self.assertTrue((output_dir / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()

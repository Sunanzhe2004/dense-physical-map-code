from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.utils.config import load_experiment_config
from runners.experiment_runner import execute_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    output_dir = execute_experiment("comparison", config, "train")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()

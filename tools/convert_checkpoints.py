from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.io.json_io import load_json, save_json


def convert_checkpoint_metadata(source: str, destination: str) -> None:
    payload = load_json(source)
    converted = {
        "checkpoint_name": payload.get("name", "unknown"),
        "checkpoint_path": payload.get("path", ""),
        "notes": payload.get("notes", "converted metadata only"),
    }
    save_json(destination, converted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    convert_checkpoint_metadata(args.source, args.destination)
    print(f"wrote_metadata={args.destination}")


if __name__ == "__main__":
    main()

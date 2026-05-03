from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.dataset.annotation_dataset import load_annotation_record
from common.io.json_io import save_json


def build_manifest(annotation_path: str, output_path: str) -> None:
    record = load_annotation_record(annotation_path)
    manifest = {
        "records": [
            {
                "sample_id": record.sample_id,
                "image_path": record.image_path,
                "physical_properties": record.physical_properties,
            }
        ]
    }
    save_json(output_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation",
        default=str(ROOT / "annotation/examples/demo_annotation.json"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data/samples/annotations/demo_manifest.json"),
    )
    args = parser.parse_args()
    build_manifest(args.annotation, args.output)
    print(f"wrote_manifest={args.output}")


if __name__ == "__main__":
    main()

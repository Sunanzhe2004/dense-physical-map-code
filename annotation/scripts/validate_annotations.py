from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.io.json_io import load_json


def _validate_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_keys = ("sample_id", "image_path", "physical_properties")
    for key in required_keys:
        if key not in record:
            errors.append(f"missing key: {key}")

    properties = record.get("physical_properties", {})
    for key in ("albedo", "roughness", "metallic"):
        if key not in properties:
            errors.append(f"missing physical property: {key}")

    if "roughness" in properties and not isinstance(properties["roughness"], (int, float)):
        errors.append("roughness must be numeric")
    if "metallic" in properties and not isinstance(properties["metallic"], (int, float)):
        errors.append("metallic must be numeric")
    return errors


def validate_annotation_file(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    errors = _validate_record(payload)
    return {
        "valid": not errors,
        "errors": errors,
        "schema_path": str(ROOT / "annotation/schemas/annotation_schema.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    report = validate_annotation_file(args.input)
    if report["valid"]:
        print("annotation_valid=true")
        return
    print("annotation_valid=false")
    for error in report["errors"]:
        print(error)
    raise SystemExit(1)


if __name__ == "__main__":
    main()

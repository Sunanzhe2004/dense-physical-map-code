from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.io.json_io import load_json


@dataclass(frozen=True)
class AnnotationRecord:
    sample_id: str
    image_path: str
    physical_properties: dict[str, Any]
    metadata: dict[str, Any]


def load_annotation_record(path: str | Path) -> AnnotationRecord:
    payload = load_json(path)
    return AnnotationRecord(
        sample_id=payload["sample_id"],
        image_path=payload["image_path"],
        physical_properties=payload["physical_properties"],
        metadata=payload.get("metadata", {}),
    )

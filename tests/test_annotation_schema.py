from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from annotation.scripts.validate_annotations import validate_annotation_file


class AnnotationSchemaTests(unittest.TestCase):
    def test_demo_annotation_is_valid(self) -> None:
        report = validate_annotation_file(ROOT / "annotation/examples/demo_annotation.json")
        self.assertTrue(report["valid"])


if __name__ == "__main__":
    unittest.main()

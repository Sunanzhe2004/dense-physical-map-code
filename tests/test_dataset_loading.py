from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.dataset.annotation_dataset import load_annotation_record


class DatasetLoadingTests(unittest.TestCase):
    def test_load_annotation_record(self) -> None:
        record = load_annotation_record(ROOT / "annotation/examples/demo_annotation.json")
        self.assertEqual(record.sample_id, "sample_0001")
        self.assertIn("roughness", record.physical_properties)


if __name__ == "__main__":
    unittest.main()

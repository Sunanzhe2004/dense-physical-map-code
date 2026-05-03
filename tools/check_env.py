from __future__ import annotations

import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    print(f"repo_root={ROOT}")
    print(f"python_version={platform.python_version()}")
    print(f"platform={platform.platform()}")
    print(f"has_annotation_example={(ROOT / 'annotation/examples/demo_annotation.json').exists()}")


if __name__ == "__main__":
    main()

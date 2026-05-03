from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.metrics.reporting import collect_metrics


def summarize(output_root: str, destination: str) -> None:
    metrics = collect_metrics(output_root)
    if not metrics:
        raise SystemExit("no metrics found")

    fieldnames = sorted({key for item in metrics for key in item.keys()})
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics:
            writer.writerow(item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    parser.add_argument("--destination", default=str(ROOT / "outputs/summary_all.csv"))
    args = parser.parse_args()
    summarize(args.output_root, args.destination)
    print(f"wrote_summary={args.destination}")


if __name__ == "__main__":
    main()

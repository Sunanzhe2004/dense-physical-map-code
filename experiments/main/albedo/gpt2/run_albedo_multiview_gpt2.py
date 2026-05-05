#!/usr/bin/env python3

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def ensure_arg(flag: str, value: str) -> None:
    if flag not in sys.argv[1:]:
        sys.argv.extend([flag, value])


def map_env_if_missing(target: str, source: str) -> None:
    if not os.environ.get(target) and os.environ.get(source):
        os.environ[target] = os.environ[source]


def main() -> None:
    map_env_if_missing("AZURE_OPENAI_API_KEY", "AZURE_GPT_IMAGE_2_API_KEY")
    map_env_if_missing("AZURE_OPENAI_ENDPOINT", "AZURE_GPT_IMAGE_2_ENDPOINT")
    map_env_if_missing("AZURE_OPENAI_API_VERSION", "AZURE_GPT_IMAGE_2_API_VERSION")
    map_env_if_missing("AZURE_ALBEDO_OPENAI_API_KEY", "AZURE_GPT_IMAGE_2_API_KEY")
    map_env_if_missing("AZURE_ALBEDO_OPENAI_ENDPOINT", "AZURE_GPT_IMAGE_2_ENDPOINT")
    map_env_if_missing("AZURE_ALBEDO_OPENAI_API_VERSION", "AZURE_GPT_IMAGE_2_API_VERSION")
    ensure_arg("--albedo_model", "gpt-image-2")
    ensure_arg("--albedo_deployment", "gpt-image-2")
    target = Path(__file__).resolve().parents[1] / "gpt" / "run_albedo_multiview_gpt.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()

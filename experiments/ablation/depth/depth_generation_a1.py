#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

from depth_ablation_runner import VariantConfig, run_variant


CONFIG = VariantConfig(
    variant_id="A1",
    variant_name="RGB Plus Fixed Example Pair",
    description="A1 ablation: relative depth generation from the target RGB image plus one fixed RGB/depth exemplar pair.",
    prompt_version="relative_depth_benchmark_v9_near_white_only",
    input_mode="rgb_plus_example",
    route="seedream_rgb_relative_depth_example",
    evaluation_protocol="single_image_relative_depth_prediction",
    evaluation_note="Near-white only. No polarity inversion, scale fit, or shift alignment is applied during generation.",
    use_example_pair=True,
)


if __name__ == "__main__":
    try:
        run_variant(CONFIG, Path(__file__).resolve().parent)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)

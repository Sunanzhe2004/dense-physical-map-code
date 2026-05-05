#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

from depth_ablation_runner import VariantConfig, run_variant


CONFIG = VariantConfig(
    variant_id="A3",
    variant_name="RGB Plus Segmentation Prior",
    description="A3 ablation: relative depth generation from the target RGB image plus a paired segmentation map.",
    prompt_version="relative_depth_benchmark_v10_rgb_plus_seg_near_white",
    input_mode="rgb_plus_seg",
    route="seedream_rgb_seg_relative_depth",
    evaluation_protocol="single_image_relative_depth_prediction_with_segmentation_prior",
    evaluation_note="Near-white only. Segmentation is used as a weak spatial or boundary prior only. No polarity inversion, scale fit, or shift alignment is applied during generation.",
    use_segmentation=True,
)


if __name__ == "__main__":
    try:
        run_variant(CONFIG, Path(__file__).resolve().parent)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)

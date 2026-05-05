#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys

from albedo_ablation_runner import FULL_PROMPT_TEXT, VariantConfig, run_variant


CONFIG = VariantConfig(
    variant_id="A0",
    variant_name="Per-Image Analysis + Full Prompt",
    prompt_version="albedo_a0_analysis_conditioned_full_prompt_v2",
    description="A0 ablation: per-image analysis plus the full albedo prompt.",
    prompt_text=FULL_PROMPT_TEXT,
    analysis_mode="single_image",
)


if __name__ == "__main__":
    try:
        run_variant(CONFIG)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)

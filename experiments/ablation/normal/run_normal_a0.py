#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys

from normal_ablation_runner import FULL_RGB_PLUS_EXAMPLE_PROMPT, VariantConfig, run_variant


CONFIG = VariantConfig(
    variant_id="A0",
    variant_name="Full Prompt + Fixed Example Pair",
    prompt_version="normal_a0_full_prompt_fixed_example_v2",
    description="A0 ablation: RGB + full prompt + fixed example pair.",
    prompt_text=FULL_RGB_PLUS_EXAMPLE_PROMPT,
    use_example_pair=True,
    prompt_level="full",
    default_example_rgb="image2.png",
    default_example_normal="normal2.png",
)


if __name__ == "__main__":
    try:
        run_variant(CONFIG)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)

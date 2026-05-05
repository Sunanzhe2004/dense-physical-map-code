#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys

from normal_ablation_runner import MINIMAL_RGB_PLUS_EXAMPLE_PROMPT, VariantConfig, run_variant


CONFIG = VariantConfig(
    variant_id="A2",
    variant_name="Minimal Prompt + Fixed Example Pair",
    prompt_version="normal_a2_minimal_prompt_fixed_example_v2",
    description="A2 ablation: RGB + minimal prompt + fixed example pair.",
    prompt_text=MINIMAL_RGB_PLUS_EXAMPLE_PROMPT,
    use_example_pair=True,
    prompt_level="minimal",
    default_example_rgb="image3.png",
    default_example_normal="normal3.png",
)


if __name__ == "__main__":
    try:
        run_variant(CONFIG)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)

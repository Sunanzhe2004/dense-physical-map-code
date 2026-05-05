#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys

from normal_ablation_runner import MINIMAL_RGB_ONLY_PROMPT, VariantConfig, run_variant


CONFIG = VariantConfig(
    variant_id="A3",
    variant_name="Minimal Prompt",
    prompt_version="normal_a3_minimal_prompt_v2",
    description="A3 ablation: RGB + minimal prompt, without example pair.",
    prompt_text=MINIMAL_RGB_ONLY_PROMPT,
    use_example_pair=False,
    prompt_level="minimal",
)


if __name__ == "__main__":
    try:
        run_variant(CONFIG)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys

from normal_ablation_runner import FULL_RGB_ONLY_PROMPT, VariantConfig, run_variant


CONFIG = VariantConfig(
    variant_id="A1",
    variant_name="Full Prompt",
    prompt_version="normal_a1_full_prompt_v2",
    description="A1 ablation: RGB + full prompt, without example pair.",
    prompt_text=FULL_RGB_ONLY_PROMPT,
    use_example_pair=False,
    prompt_level="full",
)


if __name__ == "__main__":
    try:
        run_variant(CONFIG)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)

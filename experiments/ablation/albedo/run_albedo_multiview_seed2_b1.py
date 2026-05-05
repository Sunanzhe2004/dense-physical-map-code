#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys

from albedo_ablation_runner_b import MINIMAL_FIXED_EXEMPLAR_PROMPT_TEXT, VariantConfig, run_variant


CONFIG = VariantConfig(
    variant_id="B1",
    variant_name="Minimal Prompt + Fixed Exemplar Pair",
    prompt_version="albedo_b1_minimal_fixed_exemplar_v1",
    description="B1 diagnostic ablation: minimal prompt with a fixed RGB/albedo exemplar pair.",
    prompt_text=MINIMAL_FIXED_EXEMPLAR_PROMPT_TEXT,
    use_example_pair=True,
    prompt_level="minimal+fixed_exemplar",
)


if __name__ == "__main__":
    try:
        run_variant(CONFIG)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)

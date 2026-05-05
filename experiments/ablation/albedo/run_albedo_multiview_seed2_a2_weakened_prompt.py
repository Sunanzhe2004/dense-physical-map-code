#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys

from albedo_ablation_runner import WEAKENED_PROMPT_TEXT, VariantConfig, run_variant


CONFIG = VariantConfig(
    variant_id="A2",
    variant_name="Weakened Prompt",
    prompt_version="albedo_a2_weakened_prompt_v1",
    description="A2 ablation: RGB + Weakened Prompt, without analysis notes.",
    prompt_text=WEAKENED_PROMPT_TEXT,
)


if __name__ == "__main__":
    try:
        run_variant(CONFIG)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)

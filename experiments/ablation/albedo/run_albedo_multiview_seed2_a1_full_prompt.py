#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys

from albedo_ablation_runner import FULL_PROMPT_TEXT, VariantConfig, run_variant


CONFIG = VariantConfig(
    variant_id="A1",
    variant_name="Full Prompt",
    prompt_version="albedo_a1_full_prompt_v1",
    description="A1 ablation: RGB + Full Prompt, without analysis notes.",
    prompt_text=FULL_PROMPT_TEXT,
)


if __name__ == "__main__":
    try:
        run_variant(CONFIG)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys

from albedo_ablation_runner import MINIMAL_PROMPT_TEXT, VariantConfig, run_variant


CONFIG = VariantConfig(
    variant_id="A3",
    variant_name="Minimal Prompt",
    prompt_version="albedo_a3_minimal_prompt_v1",
    description="A3 ablation: RGB + Minimal Prompt, without analysis notes.",
    prompt_text=MINIMAL_PROMPT_TEXT,
)


if __name__ == "__main__":
    try:
        run_variant(CONFIG)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)

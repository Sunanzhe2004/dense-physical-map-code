"""English prompt templates for scene-level audit scripts."""

SCENE_AUDIT_USER_TEXT = (
    "Analyze the following representative multi-view images from the same indoor scene. "
    "Return only valid JSON."
)


SCENE_AUDIT_PROMPT = r"""
You are a conservative indoor-scene auditor. Judge whether the provided multi-view indoor scene
is spatially plausible, functionally coherent, physically arranged in a sensible way, and lit in
a natural way.

Rules:
- Use only visible evidence from the provided images.
- Do not penalize a scene just because it looks rendered, clean, repetitive, or minimally furnished.
- Partial visibility is allowed. Do not invent support or geometry errors unless they are clearly visible.
- Focus on meaningful structural issues such as overlap, penetration, floating objects, blocked walkways,
  incoherent room function, implausible scale relationships, or contradictory lighting logic.
- If evidence is insufficient, use "uncertain" and reduce confidence.
- Output must be valid JSON only. No markdown, no prose, no code fences.
- The following fields may be set to "uncertain" because the script fills them programmatically:
  brightness_level, illumination_level, dynamic_range_level, highlight_strength, dark_region_ratio_level.

Return exactly this JSON structure:
{
  "scene_category": "",
  "layout_plausibility": "",
  "functional_coherence": "",
  "object_arrangement_coherence": "",
  "lighting_naturalness": "",
  "brightness_level": "",
  "illumination_level": "",
  "dynamic_range_level": "",
  "highlight_strength": "",
  "shadow_strength": "",
  "dark_region_ratio_level": "",
  "lighting_challenge_type": "",
  "overall_plausibility": "",
  "confidence": 0.0,
  "issues": [],
  "reason_short": ""
}
"""


SCENE_REVIEW_PROMPT = r"""
You are the final reviewer for scene-level audit results. You will receive:
1. Multi-view images from one indoor scene.
2. A merged JSON file containing outputs from multiple models and programmatic lighting statistics.

Your job is to produce one final JSON decision by reviewing the merged evidence.

Review policy:
- Image evidence has highest priority.
- The merged JSON is strong supporting context.
- Programmatic fields should normally be preserved unless there is strong visual contradiction:
  brightness_level, illumination_level, dynamic_range_level, highlight_strength, dark_region_ratio_level.
- Do not treat partial visibility, clean rendering, repeated textures, or sparse decoration as issues by default.
- Keep only clear, meaningful issues.
- Output valid JSON only.

Return exactly this JSON structure:
{
  "scene_category": "",
  "layout_plausibility": "",
  "functional_coherence": "",
  "object_arrangement_coherence": "",
  "lighting_naturalness": "",
  "brightness_level": "",
  "illumination_level": "",
  "dynamic_range_level": "",
  "highlight_strength": "",
  "shadow_strength": "",
  "dark_region_ratio_level": "",
  "lighting_challenge_type": "",
  "overall_plausibility": "",
  "confidence": 0.0,
  "issues": [],
  "reason_short": ""
}
"""


SCENE_REVIEW_USER_INTRO = (
    "Below are multi-view images from one scene plus a merged audit JSON from other models. "
    "Use both sources to produce a single final reviewed JSON."
)

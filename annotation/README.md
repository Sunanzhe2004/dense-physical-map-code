# Annotation Module

This directory isolates the public annotation contract from model training code.

## Public Surface

- Schema: `annotation/schemas/annotation_schema.json`
- Example: `annotation/examples/demo_annotation.json`
- Validation: `annotation/scripts/validate_annotations.py`
- Scene audit tools:
  - `annotation/scripts/audit_scenes_doubao.py`
  - `annotation/scripts/review_scene_audits_doubao.py`
  - `annotation/scripts/audit_scenes_glm.py`
  - `annotation/scripts/audit_scenes_qwen.py`
  - `annotation/scripts/scene_audit_prompts.py`
  - `annotation/scripts/scene_audit_utils.py`
- Metallic ablation tools:
  - `annotation/scripts/metallic_generation_ablation_strict_final.py`
  - `annotation/scripts/metallic_ablation_runner_strict_final.py`

## Script Overview

- `validate_annotations.py`: validate public annotation JSON files before release or conversion.
- `audit_scenes_doubao.py`: batch scene audit with Doubao.
- `review_scene_audits_doubao.py`: final review for scenes whose three audit results remain inconsistent after merge; the merged result uses majority voting as supporting context, and the script asks Doubao to make the final arbitration from the images plus merged evidence.
- `audit_scenes_glm.py`: batch scene audit with GLM.
- `audit_scenes_qwen.py`: batch scene audit with Qwen plus lighting-stat based field completion.
- `scene_audit_prompts.py`: shared English prompt templates for the audit scripts.
- `scene_audit_utils.py`: shared scene listing, image encoding, JSON extraction, and lighting-stat helpers.
- `metallic_generation_ablation_strict_final.py`: metallic ablation variants A0/A1/A2/A3 that reuse the main Doubao metallic prompt core.
- `metallic_ablation_runner_strict_final.py`: batch runner for the metallic ablation variants, including manifest and per-image metadata writing.

These scripts are kept directly under `annotation/scripts/` so the annotation workflow is visible in one place.

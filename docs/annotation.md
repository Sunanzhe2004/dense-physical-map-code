# Annotation Workflow

The annotation workspace is isolated from training code.

## Contents

- `annotation/schemas/`: public annotation schema.
- `annotation/examples/`: tiny examples for documentation and tests.
- `annotation/scripts/`: validation, export, and scene-level audit scripts.
- `annotation/configs/`: annotation pipeline settings.

## Recommended Flow

1. Draft or collect annotations in your internal tooling.
2. Export to the public schema used in this repository.
3. Run `annotation/scripts/validate_annotations.py`.
4. Convert the validated records into a training manifest if needed, using the repository data-prep tooling.

## Script Roles

- `validate_annotations.py`: validates annotation JSON files before release or conversion.
- `audit_scenes_doubao.py`: runs batch scene audit with Doubao.
- `audit_scenes_glm.py`: runs batch scene audit with GLM.
- `audit_scenes_qwen_stats.py`: runs batch scene audit with Qwen and fills several lighting-related fields from programmatic image statistics.
- `review_scene_audits_doubao.py`: reviews scenes whose three audit outputs remain inconsistent after merge; the merged result uses majority voting as supporting context, and the script produces a final arbitration result from the images plus merged evidence.
- `scene_audit_prompts.py`: stores shared English prompt templates.
- `scene_audit_utils.py`: stores shared helpers for lighting statistics, image encoding, and result parsing.

These scripts are intended for internal or semi-public annotation workflow release. They are organized under `annotation/` because they belong to dataset labeling rather than model training.

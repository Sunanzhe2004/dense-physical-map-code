# Metallic Ablations

This directory contains the released metallic ablation scripts.

## Variants

- `A0`: RGB-only metallic generation.
- `A1`: RGB plus a soft segmentation prior.
- `A2`: RGB plus segmentation-guided diagnostic region fill.
- `A3`: RGB plus one fixed RGB/metallic exemplar pair.

## Shared Logic

- `metallic_generation_ablation_strict_final.py`: shared variant definitions and reusable generation helpers.
- `metallic_ablation_runner_strict_final.py`: multi-variant ablation runner for released `A0-A3` experiments.

## Usage Notes

- All scripts expect `ARK_API_KEY`.
- `metallic_ablation_runner_strict_final.py` can run one or more variants through `--variant` or `--variants`.
- Segmentation-based variants expect paired segmentation inputs supplied through `--seg_dir`.
- The scripts auto-locate the main metallic generator at `experiments/main/metallic/doubao/metallic_generation_doubao_final.py`, so they keep working after being placed under `experiments/ablation/metallic/`.

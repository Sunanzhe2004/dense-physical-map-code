# Roughness Ablations

This directory contains the released roughness ablation scripts and lightweight example assets.

## Variants

- `A0`: RGB-only roughness generation.
- `A1`: RGB plus a soft segmentation prior.
- `A2`: RGB plus segmentation-guided diagnostic region fill.
- `A3`: RGB plus one fixed RGB/roughness exemplar pair.

## Shared Logic

- `roughness_generation_ablation.py`: shared variant definitions and reusable generation helpers.
- `roughness_ablation_runner.py`: multi-variant ablation runner for released `A0-A3` experiments.

## Example Assets

Lightweight reference assets live under `examples/`:

- `examples/image.png`
- `examples/roughness.png`

`A3` uses these files by default unless `--example_rgb` and `--example_roughness` are provided explicitly.

## Usage Notes

- All scripts expect `ARK_API_KEY`.
- `roughness_ablation_runner.py` can run one or more variants through `--variant` or `--variants`.
- Segmentation-based variants expect paired segmentation inputs supplied through `--seg_dir`.

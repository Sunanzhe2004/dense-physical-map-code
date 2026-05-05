# Depth Ablations

This directory contains the released depth ablation scripts and lightweight example assets.

## Variants

- `depth_generation_a0.py`: RGB-only depth generation baseline.
- `depth_generation_a1.py`: RGB plus one fixed RGB/depth exemplar pair.
- `depth_generation_a3.py`: RGB plus a paired segmentation prior.

There is currently no `A2` depth script in this release directory.

## Shared Logic

- `depth_ablation_runner.py`: shared runner for all released depth ablation variants.

## Example Assets

Lightweight reference assets live under `examples/`:

- `examples/image.png`
- `examples/depth.png`

These files are used by the `A1` fixed-exemplar variant unless `--example_rgb` and `--example_depth` are provided explicitly.

## Usage Notes

- All scripts expect `ARK_API_KEY`.
- `A1` uses the bundled example pair by default but allows overrides through CLI arguments.
- `A3` expects segmentation maps paired with RGB filenames using the `*_im.png` to `*_seg.png` convention.

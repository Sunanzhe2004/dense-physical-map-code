# Normal Ablations

This directory contains the released normal-map ablation scripts and lightweight example assets.

## Variants

- `run_normal_a0.py`: full prompt plus a fixed RGB/normal exemplar pair.
- `run_normal_a1.py`: full prompt without an exemplar pair.
- `run_normal_a2.py`: minimal prompt plus a fixed RGB/normal exemplar pair.
- `run_normal_a3.py`: minimal prompt without an exemplar pair.

## Shared Logic

- `normal_ablation_runner.py`: shared runner for all released normal ablation variants.

## Example Assets

Lightweight reference assets live under `examples/`:

- `examples/image2.png`
- `examples/normal2.png`
- `examples/image3.png`
- `examples/normal3.png`

Default exemplar mapping:

- `A0` uses `image2.png` and `normal2.png`
- `A2` uses `image3.png` and `normal3.png`

You can override these defaults with `--example_rgb` and `--example_normal`.

## Usage Notes

- All scripts expect `ARK_API_KEY`.
- Variants without exemplar pairs ignore `--example_rgb` and `--example_normal`.

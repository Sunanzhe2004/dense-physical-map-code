# Albedo Ablations

This directory contains the released albedo ablation scripts and small example assets.

## Variants

- `run_albedo_multiview_seed2_a0.py`: analysis-conditioned albedo baseline with the strongest released albedo setup in this ablation group.
- `run_albedo_multiview_seed2_a1_full_prompt.py`: prompt-only full-prompt variant without analysis notes.
- `run_albedo_multiview_seed2_a2_weakened_prompt.py`: prompt-only weakened-prompt variant.
- `run_albedo_multiview_seed2_a3_minimal_prompt.py`: prompt-only minimal-prompt variant.
- `run_albedo_multiview_seed2_b1.py`: diagnostic variant using a minimal prompt plus a fixed RGB/albedo exemplar pair.

## Shared Logic

- `albedo_ablation_runner.py`: shared runner for `A0`, `A1`, `A2`, and `A3`.
- `albedo_ablation_runner_b.py`: shared runner for `B1`.
- `A0` is now represented as the analysis-conditioned configuration of the shared `A0-A3` runner.

## Example Assets

Lightweight reference assets live under `examples/`:

- `examples/image.png`
- `examples/albedo.png`

These files can be used as convenient manual exemplar inputs when testing `B1`.

## Usage Notes

- All scripts expect `ARK_API_KEY` unless `--api_key` is passed explicitly.
- `B1` additionally requires `--example_rgb` and `--example_albedo`.
- Output directories are created automatically under the path passed to `--output_dir`.

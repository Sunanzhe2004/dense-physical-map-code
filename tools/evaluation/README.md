# Evaluation Scripts

This directory collects the released standalone evaluation scripts for the benchmark targets:

- `eval_depth_maps.py`
- `eval_normal_maps.py`
- `eval_albedo_maps.py`
- `eval_roughness_maps.py`
- `eval_metallic_maps.py`

These scripts are kept under `tools/evaluation/` because they are command-line evaluators rather than lightweight reusable library helpers.

## Notes

- `eval_depth_maps.py` implements affine-invariant depth evaluation and writes per-image plus aggregate summaries.
- `eval_metallic_maps.py` evaluates continuous metallic maps with masked full-map distortion and diagnostic fidelity metrics.
- Several scripts optionally use heavier dependencies such as `OpenEXR`, `scipy`, `scikit-image`, `torch`, and `lpips`; see [docs/environment.md](../../docs/environment.md).
- The default `data/...` and `outputs/evaluation/...` paths are release-friendly placeholders and can be overridden with CLI flags.

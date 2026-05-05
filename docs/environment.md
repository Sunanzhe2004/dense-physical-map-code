# Environment

This project has three environment layers:

- the lightweight package skeleton used for shared utilities, tests, and demo data preparation;
- the released main experiment environment used by the real target-wise generation scripts under `experiments/main/`.
- the released ablation experiment environment used by the target-wise ablation scripts under `experiments/ablation/`.

The second and third layers have additional runtime dependencies and provider-specific environment variables. The Python package dependencies are now exposed through optional extras in `pyproject.toml`, while provider credentials and local path variables still need to be configured manually.

## Base Python Environment

- Python `>=3.10`
- `pip`
- editable install of this repository:

```bash
python -m pip install -e .
```

This base setup is sufficient for the shared package code in `src/`, the demo preparation tools, and the lightweight test suite.

Optional extras are available for the released experiment and evaluation stacks:

```bash
python -m pip install -e ".[main]"
python -m pip install -e ".[ablation]"
python -m pip install -e ".[evaluation]"
```

## Released Experiment Runtime Dependencies

The released main and ablation experiment scripts additionally rely on:

- `requests`
- `Pillow`
- `openai` for GPT / Azure OpenAI experiment families
- `volcenginesdkarkruntime` for Doubao / ARK experiment families
- `numpy` for the released roughness ablation diagnostics

These dependencies are used directly by the released generation scripts rather than by the minimal package skeleton in `src/`.

## Evaluation Script Dependencies

The standalone evaluators under `tools/evaluation/` additionally rely on target-specific scientific and perceptual-metric packages:

- `numpy`
- `Pillow`
- `scikit-image` for `SSIM`
- `torch` and `lpips` for `LPIPS`
- `scipy` for Kendall tau in affine-invariant depth evaluation
- `OpenEXR` and `Imath` when reading `.exr` depth ground truth

Some of these dependencies are optional at import time in the scripts, but the corresponding metrics or file-format readers are only available when the package is installed.

### Evaluation Family Details

- `tools/evaluation/eval_depth_maps.py`
  Requires `numpy` and `Pillow`; additionally uses `scipy` for Kendall tau and `OpenEXR` plus `Imath` when reading `.exr` depth ground truth.
- `tools/evaluation/eval_normal_maps.py`
  Requires `numpy` and `Pillow`.
- `tools/evaluation/eval_albedo_maps.py`
  Requires `numpy` and `Pillow`; additionally uses `scikit-image` for `SSIM` and `torch` plus `lpips` when `LPIPS` is enabled.
- `tools/evaluation/eval_roughness_maps.py`
  Requires `numpy` and `Pillow`; additionally uses `scikit-image` for `SSIM` and `torch` plus `lpips` when `LPIPS` is enabled.
- `tools/evaluation/eval_metallic_maps.py`
  Requires `numpy`, `Pillow`, and `scikit-image`; additionally uses `torch` plus `lpips` unless `--skip_lpips` is passed.

## Environment Variable Groups

Released experiment scripts typically require three categories of variables:

- local data and output paths such as `GT_ROOT`, `INPUT_DIR`, `OUTPUT_DIR`, and `BASE_OUTPUT_DIR`
- provider credentials such as `ARK_API_KEY`, `DASHSCOPE_API_KEY`, `OPENAI_API_KEY`, or target-specific Azure aliases
- target-specific auxiliary inputs such as `EXAMPLE_RGB`, `EXAMPLE_NORMAL`, `EXAMPLE_ROUGHNESS`, `PROMPT_SOURCE_DIR`, or `INPUT_MODE`

## Provider Families

### GPT / Azure OpenAI

Common variables include:

- `OPENAI_API_KEY` or Azure target-specific aliases
- target-specific Azure endpoint variables such as `AZURE_*_OPENAI_ENDPOINT`
- target-specific Azure API-version variables such as `AZURE_*_OPENAI_API_VERSION`

Used by the GPT experiment families under:

- `experiments/main/albedo/gpt/`
- `experiments/main/albedo/gpt2/`
- `experiments/main/depth/gpt/`
- `experiments/main/depth/gpt2/`
- `experiments/main/normal/gpt/`
- `experiments/main/roughness/gpt/`
- `experiments/main/roughness/gpt2/`
- `experiments/main/metallic/gpt/`
- `experiments/main/metallic/gpt2/`

For `experiments/main/albedo/gpt2/`, `experiments/main/depth/gpt2/`, and `experiments/main/roughness/gpt2/`, the `gpt2/` directories are thin wrappers over shared GPT-image generation logic and mainly pin the default model family, output root, and Azure/OpenAI alias chain.

### Qwen / DashScope

Common variables include:

- `DASHSCOPE_API_KEY`
- worker-sharded variants such as `DASHSCOPE_API_KEY_0`, `DASHSCOPE_API_KEY_1`, and so on
- `BASE_URL` or `DASHSCOPE_BASE_URL`

Used by the Qwen experiment families under:

- `experiments/main/albedo/qwen/`
- `experiments/main/depth/qwen/`
- `experiments/main/roughness/qwen/`
- `experiments/main/metallic/qwen/`

### Doubao / ARK

Common variables include:

- `ARK_API_KEY`
- worker-sharded variants such as `ARK_API_KEY_0..3`

Used by the Doubao experiment families under:

- `experiments/main/albedo/doubao/`
- `experiments/main/depth/doubao/`
- `experiments/main/normal/doubao/`
- `experiments/main/roughness/doubao/`
- `experiments/main/metallic/doubao/`
- `experiments/ablation/albedo/`
- `experiments/ablation/depth/`
- `experiments/ablation/normal/`
- `experiments/ablation/roughness/`

## Ablation Experiment Notes

The released ablation scripts under `experiments/ablation/` are currently Doubao / ARK based. In practice, they require:

- `ARK_API_KEY`
- local input and output paths such as `--input_dir`, `--output_dir`, and, for some variants, `--seg_dir`

### Bundled Example Assets

Several ablation folders now bundle fixed reference assets under `examples/` so that the repository can provide a stable default setup for example-based variants:

- `experiments/ablation/albedo/examples/`
- `experiments/ablation/depth/examples/`
- `experiments/ablation/normal/examples/`
- `experiments/ablation/roughness/examples/`

When a variant supports overriding these assets, the corresponding CLI flags still take precedence.

### Ablation Family Details

- `experiments/ablation/albedo/`
  Requires `requests`, `volcenginesdkarkruntime`, and, for the `A0` analysis-conditioned variant, `openai` plus `Pillow`.
- `experiments/ablation/depth/`
  Requires `requests`, `Pillow`, and `volcenginesdkarkruntime`.
- `experiments/ablation/normal/`
  Requires `requests` and `volcenginesdkarkruntime`.
- `experiments/ablation/roughness/`
  Requires `requests`, `Pillow`, `numpy`, and `volcenginesdkarkruntime`.

## Important Reproducibility Notes

- Many shell launchers intentionally keep placeholder paths such as `/path/to/benchmark_data/...` and `/path/to/benchmark_outputs/...`. Replace them before running.
- Some main albedo runs depend on precomputed Doubao-generated `per_image_analysis` JSON files.
- Shared package installation alone does not fully provision the released proprietary-provider experiment environment.

## Where To Find Family-Specific Details

For the full per-family environment-variable matrix and launcher expectations, see:

- `experiments/main/README.md`
- `experiments/ablation/README.md`
- `docs/main_experiments.md`
- `docs/ablations.md`

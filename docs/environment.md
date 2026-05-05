# Environment

This project has two environment layers:

- the lightweight package skeleton used for shared utilities, tests, and demo data preparation;
- the released main experiment environment used by the real target-wise generation scripts under `experiments/main/`.

The second layer has additional runtime dependencies and provider-specific environment variables that are not yet declared in `pyproject.toml`.

## Base Python Environment

- Python `>=3.10`
- `pip`
- editable install of this repository:

```bash
python -m pip install -e .
```

This base setup is sufficient for the shared package code in `src/`, the demo preparation tools, and the lightweight test suite.

## Main Experiment Runtime Dependencies

The released main experiment scripts under `experiments/main/` additionally rely on:

- `requests`
- `Pillow`
- `openai` for GPT / Azure OpenAI experiment families
- `volcenginesdkarkruntime` for Doubao / ARK experiment families

These dependencies are used directly by the released generation scripts rather than by the minimal package skeleton in `src/`.

## Environment Variable Groups

Main experiments typically require three categories of variables:

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
- `experiments/main/depth/gpt/`
- `experiments/main/normal/gpt/`
- `experiments/main/roughness/gpt/`
- `experiments/main/roughness/gpt2/`
- `experiments/main/metallic/gpt/`
- `experiments/main/metallic/gpt2/`

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

## Important Reproducibility Notes

- Many shell launchers intentionally keep placeholder paths such as `/path/to/benchmark_data/...` and `/path/to/benchmark_outputs/...`. Replace them before running.
- Some main albedo runs depend on precomputed Doubao-generated `per_image_analysis` JSON files.
- Shared package installation alone does not fully provision the released proprietary-provider experiment environment.

## Where To Find Family-Specific Details

For the full per-family environment-variable matrix and launcher expectations, see:

- `experiments/main/README.md`
- `docs/main_experiments.md`

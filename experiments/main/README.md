# Main Experiments

This directory now stores the real target-wise main experiment scripts rather than the earlier demo scaffold.

For consolidated environment setup, runtime dependencies, and provider-variable guidance, see [docs/environment.md](../../docs/environment.md).

## Layout

- `albedo/`: main albedo-generation scripts grouped by provider (`doubao`, `gpt`, `gpt2`, `qwen`) plus fixed documentation examples under `examples/`.
- `depth/`: main relative-depth-generation scripts grouped by provider (`doubao`, `gpt`, `gpt2`, `qwen`).
- `metallic/`: main metallic-generation scripts grouped by provider (`doubao`, `gpt`, `gpt2`, `qwen`).
- `normal/`: main normal-generation scripts grouped by provider (`doubao`, `gpt`) plus fixed one-shot example pairs under `examples/`.
- `roughness/`: main roughness-generation scripts grouped by provider (`doubao`, `gpt`, `gpt2`, `qwen`).

For `albedo/`, `depth/`, and `roughness/`, the `gpt2/` folders are thin wrappers around the shared GPT-image generation logic. They are kept as separate directories so that released experiment paths stay explicit at the provider-and-model-family level while still avoiding duplicate implementation logic.

Each provider folder now follows a cleaner split:

- the provider root keeps the main Python generation scripts;
- a nested `scripts/` directory stores shell entry points;
- `scripts/run_*.sh` launches the main multi-part or multi-worker jobs;
- `scripts/start_*.sh` starts detached background jobs;
- `scripts/progress_*.sh` reports dataset-level progress.

This keeps the experiment logic (`.py`) separate from reproducibility and orchestration helpers (`.sh`).

## Before Running

Most shell entry points expect you to set these values first:

- `GT_ROOT`: root directory of the benchmark GT data.
- `BASE_OUTPUT_DIR` or `OUTPUT_DIR`: where predictions and logs will be written.
- `EXAMPLE_RGB` / `EXAMPLE_NORMAL`: normal-map example pairs for one-shot normal scripts.
- Optional tuning variables such as `MAX_GENERATE`, `OVERWRITE`, `TIMEOUT`, `WORKERS`, `INPUT_MODE`, and `GENERATION_MODE`.

All default paths in these scripts are placeholders like `/path/to/benchmark_data/...` and `/path/to/benchmark_outputs/...`; replace them through environment variables before running.

### What You Must Replace

The shell scripts intentionally keep public-safe placeholder defaults. Before running any released main experiment, you must replace the following placeholder values with your own local settings:

- `/path/to/benchmark_data/...`: your local benchmark data root.
- `/path/to/benchmark_outputs/...`: your local output root.
- `your_key_0`, `your_key_1`, `your_key_2`, `your_key_3`: your own provider API keys for multi-worker Doubao scripts.

If you leave these placeholder values unchanged, the scripts are expected to fail early.

### What This Repo Provides

- Main experiment Python scripts.
- Shell launchers for foreground, detached, and progress-tracking runs.
- Fixed albedo documentation examples under `experiments/main/albedo/examples/`.
- Fixed normal exemplar image pairs under `experiments/main/normal/examples/`.
- Prompt construction logic that is embedded in the released Python scripts.
- README-level guidance on provider-specific environment variables and access assumptions.

### What This Repo Does Not Provide

- Any proprietary API keys, endpoints, or provider accounts.
- Benchmark GT data itself under `/path/to/benchmark_data/...`.
- Pre-created output directories under `/path/to/benchmark_outputs/...`.
- Guaranteed redistribution of proprietary model outputs.
- Automatically generated albedo analysis JSON for GPT/Qwen runs.

For albedo main experiments, GPT and Qwen reuse Doubao-generated `per_image_analysis` JSON files. Those files are an explicit prerequisite rather than a bundled asset.

### Required Variables by Script Family

The minimum variables depend on which experiment family you run:

| Family | Required local path vars | Required auth vars | Optional but common |
|---|---|---|---|
| `albedo/doubao` | `GT_ROOT`, `BASE_OUTPUT_DIR` | `ARK_API_KEY_0..3` | `PYTHON_BIN`, `MAX_GENERATE` |
| `albedo/gpt` | `GT_ROOT`, `BASE_OUTPUT_DIR` | `ALBEDO_API_KEY`, `ALBEDO_ENDPOINT` | `ALBEDO_API_VERSION`, `ALBEDO_MODEL`, `ALBEDO_DEPLOYMENT` |
| `albedo/gpt2` | `GT_ROOT`, `BASE_OUTPUT_DIR` | `ALBEDO_API_KEY` or `AZURE_GPT_IMAGE_2_API_KEY` aliases | `ALBEDO_API_VERSION`, `ALBEDO_MODEL`, `ALBEDO_DEPLOYMENT` |
| `albedo/qwen` | `GT_ROOT`, `BASE_OUTPUT_DIR` | `DASHSCOPE_API_KEY` | `BASE_URL`, `ALBEDO_MODEL` |
| `depth/doubao` | `GT_ROOT` or `INPUT_ROOT`, `BASE_OUTPUT_DIR` | `ARK_API_KEY` or `ARK_API_KEY_0..3` | `PYTHON_BIN`, `MAX_GENERATE` |
| `depth/gpt` | `GT_ROOT` or `INPUT_ROOT`, `BASE_OUTPUT_DIR` | provider-specific Azure/OpenAI depth key vars used by the script | size / retry overrides |
| `depth/gpt2` | `GT_ROOT` or `INPUT_ROOT`, `BASE_OUTPUT_DIR` | `AZURE_GPT_IMAGE_2_API_KEY` or broader Azure/OpenAI aliases | `IMAGE_MODEL`, `IMAGE_SIZE`, size / retry overrides |
| `depth/qwen` | `GT_ROOT` or `INPUT_ROOT`, `BASE_OUTPUT_DIR` | `DASHSCOPE_API_KEY` or `DASHSCOPE_API_KEY_0..3` | `DASHSCOPE_BASE_URL`, `MAX_GENERATE` |
| `normal/doubao` | `GT_ROOT`, `BASE_OUTPUT_DIR` | `ARK_API_KEY_0..3` | `EXAMPLE_RGB`, `EXAMPLE_NORMAL` |
| `normal/gpt` | `GT_ROOT`, `BASE_OUTPUT_DIR` | `AZURE_NORMAL_OPENAI_API_KEY`, `AZURE_NORMAL_OPENAI_ENDPOINT` | `AZURE_NORMAL_OPENAI_API_VERSION`, `EXAMPLE_RGB`, `EXAMPLE_NORMAL` |
| `roughness/doubao` | `GT_ROOT`, `BASE_OUTPUT_DIR` | `ARK_API_KEY_0..3` | `INPUT_MODE`, `EXAMPLE_RGB`, `EXAMPLE_ROUGHNESS` |
| `roughness/gpt` | `GT_ROOT`, `BASE_OUTPUT_DIR` | `AZURE_ROUGHNESS_OPENAI_API_KEY` or compatible Azure/OpenAI aliases | `INPUT_MODE`, `EXAMPLE_RGB`, `EXAMPLE_ROUGHNESS` |
| `roughness/gpt2` | `GT_ROOT`, `BASE_OUTPUT_DIR` | `AZURE_ROUGHNESS_OPENAI_API_KEY` or compatible Azure/OpenAI aliases | `INPUT_MODE`, `EXAMPLE_RGB`, `EXAMPLE_ROUGHNESS` |
| `roughness/qwen` | `GT_ROOT`, `BASE_OUTPUT_DIR` | `DASHSCOPE_API_KEY` or `QWEN_API_KEY` | `INPUT_MODE`, `EXAMPLE_RGB`, `EXAMPLE_ROUGHNESS` |
| `metallic/doubao` | `INPUT_DIR`, `OUTPUT_DIR` | `ARK_API_KEY_0..3` | `PROMPT_SOURCE_DIR`, `NUM_WORKERS` |
| `metallic/gpt` | `INPUT_DIR`, `OUTPUT_DIR` | `AZURE_METALLIC_OPENAI_API_KEY` or compatible Azure/OpenAI aliases | `PROMPT_SOURCE_DIR`, `NUM_WORKERS`, `IMAGE_MODEL` |
| `metallic/gpt2` | `INPUT_DIR`, `OUTPUT_DIR` | `AZURE_METALLIC_OPENAI_API_KEY` or compatible Azure/OpenAI aliases | `PROMPT_SOURCE_DIR`, `NUM_WORKERS`, `IMAGE_MODEL` |
| `metallic/qwen` | `INPUT_DIR`, `OUTPUT_DIR` | `DASHSCOPE_API_KEY_0..N` or equivalent worker keys | `PROMPT_SOURCE_DIR`, `NUM_WORKERS` |

If a script supports both a shared key and worker-specific keys, the worker-specific variables take precedence for sharded runs.

### GPT

The GPT albedo scripts are Azure OpenAI based.

- Model: `ALBEDO_MODEL` and `ALBEDO_DEPLOYMENT` default to `gpt-image-1.5`, but the same scripts can also be pointed to `gpt-image-2`.
- Key: `ALBEDO_API_KEY` or `AZURE_ALBEDO_OPENAI_API_KEY` or `AZURE_GPT_IMAGE_15_API_KEY` or `AZURE_GPT_IMAGE_2_API_KEY`.
- Endpoint: `ALBEDO_ENDPOINT` or `AZURE_ALBEDO_OPENAI_ENDPOINT` or `AZURE_GPT_IMAGE_15_ENDPOINT` or `AZURE_GPT_IMAGE_2_ENDPOINT`.
- API version: `ALBEDO_API_VERSION` or the matching Azure aliases.
- Analysis input: the GPT albedo script does not run its own analysis model in the main experiment path. Its `analysis_model` is hard-coded to `external_json`, and it reads precomputed per-image analysis JSON files from `analysis_dirs`.
- Default dependency: those `analysis_dirs` point to `benchmark_outputs/albedo_doubao/.../meta/per_image_analysis`, so the GPT albedo run expects Doubao-generated analysis files to already exist.

The GPT depth scripts follow the same reuse pattern.

- Model: `IMAGE_MODEL` defaults to `gpt-image-1.5`, but the same scripts can also be pointed to `gpt-image-2`.
- Key: `AZURE_GPT_IMAGE_15_API_KEY` or `AZURE_GPT_IMAGE_2_API_KEY`, with broader `AZURE_OPENAI_API_KEY` / `OPENAI_API_KEY` fallbacks.
- Endpoint: `AZURE_GPT_IMAGE_15_ENDPOINT` or `AZURE_GPT_IMAGE_2_ENDPOINT`, with broader `AZURE_OPENAI_ENDPOINT` fallback.
- API version: `AZURE_GPT_IMAGE_15_API_VERSION` or `AZURE_GPT_IMAGE_2_API_VERSION`, with broader `AZURE_OPENAI_API_VERSION` fallback.

The GPT normal scripts are also Azure OpenAI based.

- Model: the script uses the configured normal image deployment through the normal API vars.
- Key: `AZURE_NORMAL_OPENAI_API_KEY`.
- Endpoint: `AZURE_NORMAL_OPENAI_ENDPOINT`.
- API version: `AZURE_NORMAL_OPENAI_API_VERSION`.

The GPT roughness and metallic scripts use the same pattern but allow target-specific aliases first, then broader Azure/OpenAI fallbacks.

- Roughness key fallback chain: `AZURE_ROUGHNESS_OPENAI_API_KEY` -> `AZURE_GPT_IMAGE_15_API_KEY` or `AZURE_GPT_IMAGE_2_API_KEY` -> `AZURE_OPENAI_API_KEY` -> `OPENAI_API_KEY`.
- Metallic key fallback chain: `AZURE_METALLIC_OPENAI_API_KEY` -> `AZURE_GPT_IMAGE_15_API_KEY` or `AZURE_GPT_IMAGE_2_API_KEY` -> `AZURE_OPENAI_API_KEY` -> `OPENAI_API_KEY`.
- Endpoint and API-version variables follow the same target-specific-first pattern.

### Qwen

The Qwen albedo scripts are DashScope based.

- Model: `ALBEDO_MODEL` defaults to `wan2.7-image`.
- Key: `DASHSCOPE_API_KEY`.
- Endpoint: `BASE_URL` defaults to `https://dashscope.aliyuncs.com/api/v1`.
- Analysis input: the Qwen albedo script also does not run its own analysis model in the main experiment path. Its `analysis_model` is also `external_json`, and it reads precomputed per-image analysis JSON files from `analysis_dirs`.
- Default dependency: those `analysis_dirs` also point to `benchmark_outputs/albedo_doubao/.../meta/per_image_analysis`, so the Qwen albedo run reuses the same Doubao-generated analysis files.

The Qwen depth, roughness, and metallic scripts also expect DashScope credentials.

- Shared-key mode: set `DASHSCOPE_API_KEY`.
- Multi-worker mode: set `DASHSCOPE_API_KEY_0`, `DASHSCOPE_API_KEY_1`, and so on when a launcher shards work across workers.
- Base URL: `DASHSCOPE_BASE_URL` or `BASE_URL`, depending on the script family.

Some Qwen roughness scripts also accept `QWEN_API_KEY` as a fallback alias.

### Doubao

The Doubao scripts are ARK API based.

- Shared-key mode: some scripts accept `ARK_API_KEY`.
- Multi-worker mode: most sharded shell launchers expect `ARK_API_KEY_0`, `ARK_API_KEY_1`, `ARK_API_KEY_2`, and `ARK_API_KEY_3`.
- Public-safe placeholders such as `your_key_0` are intentionally invalid and must be replaced before running.

If you only set `ARK_API_KEY` for a launcher that explicitly checks `ARK_API_KEY_0..3`, that launcher is not fully configured.

### Albedo Analysis Dependency

For the released main albedo pipeline, only one analysis model is actually used to produce the analysis JSON:

- Generator script: `experiments/main/albedo/doubao/run_albedo_multiview_seed2.py`
- Analysis model default: `doubao-seed-2-0-pro-260215`
- Output location: `meta/per_image_analysis`

In other words:

- Doubao albedo runs both the analysis stage and the final albedo generation stage.
- GPT albedo runs only the final generation stage and consumes the saved Doubao analysis JSON.
- Qwen albedo runs only the final generation stage and consumes the same saved Doubao analysis JSON.

If you want to reproduce the current main-experiment setup faithfully, generate the Doubao `per_image_analysis` files first, then run the GPT or Qwen albedo scripts against those analysis directories.

This means the GPT and Qwen albedo launchers are not standalone first-stage pipelines. They are second-stage consumers of the saved Doubao analysis artifacts.

If you switch to a different model or deployment, override the corresponding environment variable explicitly.

### Input-Mode-Specific Variables

Some families require extra variables only for certain `INPUT_MODE` settings:

- `INPUT_MODE=rgb_plus_example`: requires `EXAMPLE_RGB` and the target-specific example map such as `EXAMPLE_NORMAL` or `EXAMPLE_ROUGHNESS`.
- `INPUT_MODE=rgb_plus_seg`: requires segmentation inputs where supported by the corresponding roughness script family.
- `INPUT_MODE=rgb_plus_prompt`: commonly uses `PROMPT_SOURCE_DIR` for metallic experiments when prompt text is supplied externally.

If you keep the default `rgb_only` mode, these extra inputs are usually not required.

## Typical Usage

```bash
export GT_ROOT=/your/data/GT
export BASE_OUTPUT_DIR=/your/outputs/albedo_gpt
export ALBEDO_API_KEY=...
export ALBEDO_ENDPOINT=...
bash experiments/main/albedo/gpt/scripts/run_albedo_singleview_4dirs.sh
```

The old placeholder files under `experiments/main/scripts/`, `experiments/main/methods/`, and `experiments/main/configs/demo_main.json` have been removed so that this directory reflects the actual benchmark-running entry points.

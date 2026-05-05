# Installation

## Requirements

- Python 3.10 or newer
- `pip`

## Setup

```bash
python -m pip install -e .
```

For released experiment scripts and evaluators, install the corresponding extras as needed:

```bash
python -m pip install -e ".[main]"
python -m pip install -e ".[ablation]"
python -m pip install -e ".[evaluation]"
```

## Main Experiment Note

The released scripts under `experiments/main/` require additional runtime libraries and provider-specific environment variables beyond the minimal package skeleton.

See [docs/environment.md](environment.md) for the consolidated environment guide.

## Smoke Checks

```bash
python tools/check_env.py
python -m unittest tests.test_annotation_schema tests.test_dataset_loading
```

# Main Experiments

The main experiment directory contains the primary method implementation and shared scripts for training and evaluation.

## Entry Points

```bash
python experiments/main/scripts/run_train.py --config experiments/main/configs/demo_main.json
python experiments/main/scripts/run_eval.py --config experiments/main/configs/demo_main.json
```

## Output Convention

Outputs are written under:

```text
outputs/main/<run_name>/
```

Each run should end with:

- `metrics.json`
- `summary.csv`

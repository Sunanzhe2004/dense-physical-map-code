# Comparison Experiments

Comparison experiments are organized around adapters instead of direct copies of third-party repositories.

## Entry Points

```bash
python experiments/comparison/scripts/run_train.py --config experiments/comparison/configs/demo_comparison.json
python experiments/comparison/scripts/run_eval.py --config experiments/comparison/configs/demo_comparison.json
```

## Third-Party Policy

- Keep baseline source repositories external when possible.
- Document their versions and patch steps in `experiments/comparison/third_party/README.md`.
- Use adapters to normalize inputs and outputs.

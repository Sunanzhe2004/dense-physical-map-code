# Ablation Experiments

Ablations live in their own directory and reuse only the shared utilities in `src/`.

## Entry Points

```bash
python experiments/ablation/scripts/run_train.py --config experiments/ablation/configs/demo_ablation.json
python experiments/ablation/scripts/run_eval.py --config experiments/ablation/configs/demo_ablation.json
```

## Naming

Use run names that encode the variant, for example:

- `remove_depth_guidance_seed0`
- `no_physics_loss_seed1`

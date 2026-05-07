# Ablation Experiments

Ablations live under `experiments/ablation/`.

The directory now contains released target-wise ablation scripts for `albedo`, `depth`, `metallic`, `normal`, and `roughness`.

The recently reorganized target-wise folders follow the same broad pattern where practical:

- a shared runner holds the reusable generation logic
- thin variant entry scripts define the specific ablation condition
- lightweight bundled assets live under `examples/` when a variant depends on a fixed exemplar pair

This `examples/` pattern now applies across the reorganized released folders:

- `experiments/ablation/albedo/examples/`
- `experiments/ablation/depth/examples/`
- `experiments/ablation/metallic/examples/`
- `experiments/ablation/normal/examples/`
- `experiments/ablation/roughness/examples/`

## Entry Points

For the released target-wise ablations, start from the corresponding subdirectories:

```text
experiments/ablation/albedo/
experiments/ablation/depth/
experiments/ablation/metallic/
experiments/ablation/normal/
experiments/ablation/roughness/
```

Representative entry scripts include:

```bash
python experiments/ablation/albedo/run_albedo_multiview_seed2_a0.py --help
python experiments/ablation/metallic/metallic_ablation_runner_strict_final.py --help
python experiments/ablation/normal/run_normal_a0.py --help
python experiments/ablation/roughness/roughness_ablation_runner.py --help
python experiments/ablation/depth/depth_generation_a0.py --help
```

## Depth Structure

The released depth ablations are now organized as a shared runner plus thin entry points:

- `experiments/ablation/depth/depth_ablation_runner.py`: shared logic for all released depth ablation variants
- `experiments/ablation/depth/depth_generation_a0.py`: `rgb_only`
- `experiments/ablation/depth/depth_generation_a1.py`: `rgb_plus_example`
- `experiments/ablation/depth/depth_generation_a3.py`: `rgb_plus_seg`
- `experiments/ablation/depth/examples/`: bundled fixed exemplar assets for the example-based variant

This means the depth ablation release is no longer just a set of unrelated large standalone scripts. The reusable logic now lives in one place, while the variant entry files stay small and easier to audit.

## Coverage

The current ablation import covers:

- `albedo`
- `depth`
- `metallic`
- `normal`
- `roughness`

## Naming

Use run names that encode the variant, for example:

- `no_physics_loss_seed1`
- `metallic_a2_region_fill_seed0`

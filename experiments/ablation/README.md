# Ablation Experiments

This directory now contains the released ablation code that accompanies the main benchmark pipelines, plus a small amount of earlier scaffold code kept for backward compatibility.

## Layout

- `albedo/`: released albedo ablation scripts and small example assets.
- `depth/`: released depth ablation scripts and small example assets.
- `normal/`: released normal ablation runner plus variant entry scripts and bundled exemplar pairs under `normal/examples/`.
- `roughness/`: released roughness ablation runner, variant logic, and bundled example assets under `roughness/examples/`.
- `configs/`, `scripts/`, `variants/`: earlier lightweight scaffold files kept as generic examples.

## Scope

The currently imported released ablation code covers:

- `albedo`
- `depth`
- `normal`
- `roughness`

`metallic` ablation code is not included in this directory yet, so the ablation release currently covers all main targets except metallic.

## Organization Notes

- The target-wise ablation folders preserve the original released script structure rather than forcing them into the older demo scaffold.
- Target folders that depend on fixed reference assets now keep those files under a dedicated `examples/` subdirectory, including `albedo/`, `depth/`, `normal/`, and `roughness/`.
- The older `configs/demo_ablation.json` and `scripts/run_*.py` files remain available as generic scaffold examples, but they are not the primary entry points for the released target-wise ablations.

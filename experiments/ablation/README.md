# Ablation Experiments

This directory now contains the released ablation code that accompanies the main benchmark pipelines.

## Layout

- `albedo/`: released albedo ablation scripts and small example assets.
- `depth/`: released depth ablation scripts and small example assets.
- `normal/`: released normal ablation runner plus variant entry scripts and bundled exemplar pairs under `normal/examples/`.
- `metallic/`: released metallic ablation runner, shared variant logic, and bundled example assets under `metallic/examples/`.
- `roughness/`: released roughness ablation runner, variant logic, and bundled example assets under `roughness/examples/`.

## Scope

The currently imported released ablation code covers:

- `albedo`
- `depth`
- `normal`
- `metallic`
- `roughness`

## Organization Notes

- The target-wise ablation folders preserve the original released script structure rather than forcing them into the older demo scaffold.
- Target folders that depend on fixed reference assets now keep those files under a dedicated `examples/` subdirectory, including `albedo/`, `depth/`, `normal/`, `metallic/`, and `roughness/`.

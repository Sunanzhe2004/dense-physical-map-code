# Main Experiments

The main experiment code is now organized by benchmark target instead of the previous demo `run_train.py` / `run_eval.py` scaffold.

## Directory Structure

```text
experiments/main/
|- albedo/
|  |- doubao/
|  |- examples/
|  |- gpt/
|  |- gpt2/
|  `- qwen/
|- depth/
|  |- doubao/
|  |- gpt/
|  |- gpt2/
|  `- qwen/
|- metallic/
|  |- doubao/
|  |- gpt/
|  |- gpt2/
|  `- qwen/
|- normal/
|  |- doubao/
|  |- examples/
|  `- gpt/
`- roughness/
   |- doubao/
   |- gpt/
   |- gpt2/
   `- qwen/
```

## Organization Rule

Each target directory groups the scripts that were actually used to run the main experiments:

- the core Python generation script;
- run wrappers for 4-way or multi-worker execution;
- detached-start helpers;
- progress-check helpers;
- fixed example assets when the protocol needs them, such as normal-map one-shot exemplars.

This layout keeps target-specific prompt logic, provider-specific API handling, and run wrappers close to each other, which is more faithful to the benchmark workflow than the old generic demo entry points.

For `albedo`, `depth`, and `roughness`, the `gpt2/` directories are thin wrappers around shared GPT-image generation logic. This keeps the released experiment paths explicit while avoiding large duplicated model-family implementations.

## Notes

- `albedo/examples/` stores fixed public-safe example assets bundled with the released albedo main-experiment folder.
- `normal/examples/` stores the fixed RGB-normal example pairs used by the exemplar-conditioned normal protocol.
- `metallic/` uses the target name `metallic` to stay consistent with the benchmark terminology, even though the legacy source directory was named `metallicity`.
- The old placeholder files under `experiments/main/scripts/`, `experiments/main/methods/`, and `experiments/main/configs/demo_main.json` have been removed.
- Environment setup for the released main experiment scripts is summarized in [docs/environment.md](environment.md).

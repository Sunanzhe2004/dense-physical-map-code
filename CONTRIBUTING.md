# Contributing

Thanks for helping improve this benchmark repository.

## Scope

This project focuses on benchmark release scaffolding, evaluation utilities, annotation tools, and experiment organization for dense physical-map prediction from single indoor images.

## How To Contribute

1. Open an issue before large changes so we can align on scope and benchmark compatibility.
2. Keep pull requests focused and describe the motivation, behavior change, and validation you ran.
3. Preserve existing directory boundaries such as `annotation/`, `docs/`, `experiments/`, `src/`, `tests/`, and `tools/`.
4. Avoid committing secrets, provider credentials, private datasets, or outputs that cannot be redistributed.
5. Add or update lightweight tests when a change affects validation, metrics, dataset parsing, or public schemas.

## Style Notes

- Prefer clear, reproducible behavior over benchmark-specific shortcuts.
- Document any access-setting assumptions that affect reported results.
- Keep public interfaces and file formats backward compatible when possible.

## Pull Request Checklist

- The change is scoped to a clearly explained purpose.
- Documentation is updated when behavior or public files change.
- No proprietary credentials or restricted assets are included.
- Relevant tests or smoke checks were run, or the reason they were not run is explained.

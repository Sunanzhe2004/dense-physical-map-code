# Dataset Layout

This repository keeps only lightweight public examples.

## Expected Structure

```text
data/
├─ samples/
│  ├─ images/
│  └─ annotations/
├─ splits/
├─ raw/             # local only, ignored by git
├─ intermediate/    # local only, ignored by git
└─ private/         # local only, ignored by git
```

## Conventions

- Public examples live in `data/samples/`.
- Real images should be referenced by relative paths.
- Split files should use JSON lists of sample ids.
- Experiment configs should point to manifests instead of scanning directories implicitly.

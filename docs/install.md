# Installation

## Requirements

- Python 3.10 or newer
- `pip`

## Setup

```bash
python -m pip install -e .
```

## Smoke Checks

```bash
python tools/check_env.py
python -m unittest discover -s tests -p "test_*.py"
```

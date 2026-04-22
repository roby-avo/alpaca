# Coverage_exp

This directory contains benchmark material used by `backend_lab` for local CRIA entity-linking experiments.

The expected dataset root is:

```bash
Coverage_exp/Datasets
```

`backend_lab` reads tables, CEA/CTA targets, and ground-truth labels from this directory. It does not modify these files.

## Included Datasets

Current local datasets include:

- `2T_2020`
- `2T_2022`
- `HardTableR1_2022`

You can confirm what the CLI sees with:

```bash
python -m backend_lab datasets
```

## Configuration

By default, `backend_lab` resolves this path relative to the repository root. To override it, set:

```bash
ALPACA_DATASET_ROOT=/absolute/path/to/alpaca/Coverage_exp/Datasets
```

## Notes

- Local virtual environments under this directory are ignored by git.
- Generated experiment outputs should go under `backend_lab/out/`, not here.
- Large benchmark files are intentionally data artifacts; avoid editing them unless updating the benchmark snapshot deliberately.

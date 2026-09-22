# market-physics-core

Deterministic analytics, forecast evaluation, and Bayesian updating for Market Physics.

This repository separates reusable, pure calculations from applications that choose a workflow and adapters that perform I/O. Slice 0000 provides package boundaries, documentation, and a minimal CLI only; financial calculations begin in Slice 0001.

## Development

```powershell
python -m pip install -e ".[test]"
python -m pytest
market-physics-core --help
market-physics-core --version
market-physics-core skew-baseline
```

The baseline command reads the manifest-selected, checksum-verified GCS option
surfaces and SPY/VIX history, then writes immutable local results below
`outputs/baseline-history`. See [the architecture](docs/architecture.md) and
[slice specifications](docs/slices/0001-spy-skew-baseline.md).

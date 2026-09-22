# Architecture

market-physics-core is a deterministic analysis engine. Agent orchestration, publishing services, scheduling, and infrastructure remain outside this repository.

## Layers

1. **Core** contains reusable pure calculations: skew extraction/interpolation, statistical baseline estimation, forecast scoring, and Bayesian updates. It accepts validated inputs and returns values.
2. **Applications** select instruments, expirations, observation windows, quality policies, and outputs. The SPY VIX-scaled-distance baseline is an application composed from volatility and statistics core capabilities.
3. **Adapters** read and normalize local/GCS inputs, persist baseline history, and render reports. Provider paths and credentials never enter core.
4. **CLI** parses user input and invokes applications without implementing analysis.

Dependencies point inward: CLI and adapters support applications; applications use core; core knows neither workflows nor I/O. Frequency does not determine layer placement—weekly workflow choices remain application concerns.

Source observations are immutable evidence. Results carry explicit units, timezone-aware timestamps, source provenance, methodology/configuration identity, and code identity.

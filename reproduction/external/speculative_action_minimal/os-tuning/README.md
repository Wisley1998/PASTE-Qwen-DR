# Speculative Actions Sysbench CPU Use Case

## About

This is the Lossy Speculative Actions OS Parameter Tuning use case.

It supports one workload and three setups:

- `sysbench_cpu`
- `actor_only` with `gemini-2.5-flash`
- `speculator_only` with `gemini-2.5-flash-lite`
- `dual` with both models, using `gemini-2.5-flash` for the actor and `gemini-2.5-flash-lite` for the speculator

This use case touches these scheduler knobs:

- fixed `latency_ns` (default `1000`)
- tuned `min_granularity_ns`

The main outputs are:

- Figure 5 style convergence plot
- Figure 5 style 20-second reaction summary
- Figure 7 reaction trace
- token and cost history plots

## Requirements

- Linux machine with `debugfs` support
- Permission to read and write `/sys/kernel/debug/sched/*`
- `sudo` access for setup and for benchmark runs
- Python 3
- `sysbench`
- Internet access for Gemini API calls
- Your own Gemini API key


## Important Note on Safety and Reproducibility

IMPORTANT: This use case changes live scheduler settings and enables HRTICK. Use a dedicated test machine, a disposable VM, or another environment where OS-level tuning is acceptable.

Kernel and platform behavior matter. Different CPUs, NUMA layouts, kernels, cpuidle support, power-management defaults, and scheduler implementations can change both the convergence path and the final values you observe.

## Setup

Setup requires `sudo`:

```bash
sudo ./scripts/setup.sh
```

The setup script:

- installs Python and `sysbench`
- mounts `debugfs` if needed
- installs Python dependencies
- installs the project in editable mode
- attempts to enable the scheduler `HRTICK` feature

The script prints warnings if the kernel config or `sched/features` path does not allow automatic HRTICK enablement.

## Authentication

Provide your own Gemini API key before running any experiments:

```bash
export GEMINI_API_KEY=your_key_here
```

Run commands with `sudo -E` so the environment variable survives privilege elevation.

## Running

Experiment runs require `sudo` because the use case writes scheduler knobs.

Run one config directly:

```bash
sudo -E env PYTHONPATH="$PWD/src" python3 -m barebones_optimizer.main --config config/use_case/actor_only_convergence.json
sudo -E env PYTHONPATH="$PWD/src" python3 -m barebones_optimizer.main --config config/use_case/speculator_only_convergence.json
sudo -E env PYTHONPATH="$PWD/src" python3 -m barebones_optimizer.main --config config/use_case/dual_convergence.json
```

Run the full use-case harness:

```bash
sudo -E ./scripts/run_use_case.sh all
```

Supported harness phases:

- `convergence`
- `reaction`
- `all`

The harness automatically:

- runs the requested setups
- calibrates a degraded reaction starting point for the local machine
- generates a `manifest.json`
- generates plots from that manifest

## Reaction Calibration

The paper uses a deliberately bad starting scheduler setting before measuring recovery.

On different hardware, the same `min_granularity_ns` value may or may not be bad enough. This use case therefore probes a fixed set of candidate values and selects a degraded starting point based on observed p95 latency on the current host.

The calibration result is written to `reaction_calibration.json` under the chosen results root.

## Reproducing The Plots

If you already have a manifest, regenerate the plots with:

```bash
python3 scripts/plot_use_case.py --manifest results/use_case_<timestamp>/manifest.json
```

The plotting script emits:

- `figure5_use_case.png`
- `figure7_reaction_trace.png`
- `figure9_tokens_cost.png`

## Output Layout

Each run writes a timestamped run directory with:

- `history.json`: normalized run history
- `windows/`: per-window sysbench logs

The harness writes:

- `manifest.json`: latest history paths for each setup
- `reaction_calibration.json`: calibrated degraded starting point for reaction runs
- `plots/`: generated figures

The normalized history contains:

- top-level run metadata
- OS defaults captured at run start
- per-window benchmark and system metrics
- per-request timing and token usage
- request type split (`actor` vs `speculator`)

## Experiment and Model Details:
- The optimizer targets `latency_p95`.
- `latency_p99` is recorded when available, depending on your `sysbench cpu` version, it does not always expose both percentiles in the installed build.
- The default sysbench preset in this use case is:
  - pinned cores: `0-9`
  - threads: `40`
  - `cpu-max-prime`: `50000`
- The local price map used for token/cost plots is:
  - `gemini-2.5-flash-lite`: `$0.10 / $0.40` per 1M input/output tokens
  - `gemini-2.5-flash`: `$0.30 / $2.50` per 1M input/output tokens


## Notes

Results can vary because of:

- CPU model and socket topology
- kernel version and scheduler implementation
- HRTICK availability
- default scheduler values exposed by the host
- power-management and cpuidle behavior
- Gemini model response time
- API or provider latency outside the host itself

Convergence speed and the exact final `min_granularity_ns` values may differ even when the qualitative behavior is similar.

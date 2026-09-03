# Speculative Action minimal source snapshot

This directory preserves the code needed to inspect and reproduce the four
Speculative Action workloads used as external PASTE references. It is a
source-only snapshot of:

- upstream repository: `https://github.com/naimengye/speculative-action.git`
- upstream commit: `dc938b9ef7474caf07fe4ad16549c1fa8c7d268c`
- snapshot date: `2026-09-03`

The upstream Apache-2.0 license is retained as `LICENSE`. The embedded
tau-bench code retains its own `e-commerce/tau-bench/LICENSE`.

## Included

- HotPotQA runner, prompts, source, and one small example trajectory;
- OS-tuning source, scripts, configurations, and tests;
- chess workflows plus the bundled TextArena runtime required by them; and
- tau-bench runner, agents, environments, tools, and benchmark fixtures.

## Deliberately excluded

- generated experiment results and plots;
- historical and bulk trajectories;
- the full HotPotQA dataset;
- notebooks and paper figures; and
- caches, virtual environments, logs, and repository metadata.

These exclusions reduce the imported tree from roughly 294 MiB to about
15 MiB while keeping the executable source and the small fixtures required by
the tau-bench environments. Download the full HotPotQA data from the source
identified in `hotpotqa/README.md` before running that workload.

## Relationship to PASTE-Qwen-DR

HotPotQA is the closest external comparison for Qwen-DR's multi-hop
`search -> visit` workflow. The other workloads exercise different effect
classes: local OS mutations, transactional retail/airline actions, and
prediction-only game actions. They are kept here as frozen comparison code;
the PASTE-Qwen-DR runtime does not import them as production dependencies.

Do not place API credentials in the checked-in configuration files. Use
environment variables or an untracked local configuration when adapting the
examples.

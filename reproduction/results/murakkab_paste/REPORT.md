# Murakkab-inspired PASTE comparison

> **Superseded for system comparison.** This report is an offline
> adaptive-`top_k` PASTE ablation, not a Murakkab-versus-PASTE experiment. Its
> 28.86% admitted-request proxy reduction and 4/4 synthetic SLO result must not
> be cited as Murakkab system performance. The replacement fixed-model design
> is in [`FIXED_MODEL_SAME_SETUP_PROTOCOL.md`](FIXED_MODEL_SAME_SETUP_PROTOCOL.md).

Experiment date: 2026-08-31

## Result

On the fixed final trace role, the Murakkab-inspired planner met all four aggregate SLO tiers while reducing the conservative admitted-tool request proxy by **28.86%** relative to static PASTE `top_k=5`.

| Policy | Weighted stall reduction | Tool request units / authoritative call | SLO tiers met |
|---|---:|---:|---:|
| Demand only (k=0) | 0.00% | 1.000 | 1/4 |
| Static PASTE (k=5) | 15.83% | 3.188 | 4/4 |
| Murakkab-inspired PASTE | 10.10% | 2.268 | 4/4 |

The resource metric is an admitted request-unit upper bound, not measured GPU energy, cloud cost, or completed network calls. Static demand-only uses fewer units but fails the non-basic latency tiers; static top-5 over-serves relaxed tiers.

## Selected configurations

| Tier | Required stall reduction | Planning margin | Selected | Final reduction | Final SLO |
|---|---:|---:|---:|---:|---:|
| basic | 0.00% | 0.00% | k=0 | 0.00% | pass |
| fair | 5.00% | 10.00% | k=3 | 12.12% | pass |
| good | 10.00% | 10.00% | k=4 | 12.48% | pass |
| best | 15.00% | 10.00% | k=5 | 15.83% | pass |

## Final configuration frontier

| Configuration | Hits / authoritative | Stall reduction (95% session bootstrap CI) | Tool request units / authoritative |
|---|---:|---:|---:|
| demand_only_k0 | 0/69 | 0.00% [0.00%, 0.00%] | 1.000 |
| paste_k1 | 20/69 | 8.79% [5.06%, 32.94%] | 1.261 |
| paste_k2 | 26/69 | 9.63% [5.55%, 37.66%] | 1.725 |
| paste_k3 | 31/69 | 12.12% [6.77%, 43.63%] | 2.203 |
| paste_k4 | 36/69 | 12.48% [6.75%, 47.65%] | 2.681 |
| paste_default_k5 | 39/69 | 15.83% [10.36%, 47.71%] | 3.188 |

## What was reproduced

This is an idea-level reproduction on PASTE-Qwen-DR because no official Murakkab code artifact is linked from the paper or USENIX page. It implements a typed declarative DAG, offline workflow profiles, SLO filtering, a resource-minimizing configuration planner, and exact isolated replay. Replay reconciliation was `pass` with 0 state-isolation violations.

It does **not** reproduce Murakkab's LLM-based executor discovery, model/GPU profiles, Gurobi MILP, Azure autoscaling, multi-model colocation, or the paper's energy/cost numbers. Those require unpublished code, model profiles, A100-80GB/H100-80GB fleets, and the authors' production-scale workload.

The latency result remains the repository's bounded trace counterfactual: `min(observed visit stall, preceding LLM decision window) × exact-hit fraction`. SLO pass/fail is aggregate over the final role, not a per-request guarantee. The typed DAG is validated but the recorded trace supplies execution order; each selected `k` is replayed separately and combined by declared demand weights rather than run as one online mixed-SLO service. The protocol is exploratory and was not preregistered.

Primary sources: [USENIX OSDI '26 paper page](https://www.usenix.org/conference/osdi26/presentation/chaudhry), [arXiv paper](https://arxiv.org/abs/2508.18298).

## Reproduce

```bash
python reproduction/scripts/run_murakkab_paste_comparison.py
PYTHONPATH=reproduction pytest -q reproduction/tests/test_murakkab_optimizer.py
```

Machine-readable evidence is in [`comparison.json`](comparison.json).

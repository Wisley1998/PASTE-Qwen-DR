# PASTE-Qwen-DR

Standalone reproduction of PASTE for Qwen DeepResearch. This repository keeps
five supported paths under one auditable layout:

| Path | Purpose | Entry point |
|---|---|---|
| Trace-learned tool execution | CPU-only, held-out causal evaluation without LLM co-design | `reproduction/scripts/run_speculative_tool_execution.sh` |
| Adaptive-width ablation | CPU-only profile-guided selection of PASTE speculation width; not a Murakkab system comparison | `reproduction/scripts/run_murakkab_paste_comparison.py` |
| Online speculative execution | Live Qwen-DR + live search/visit with trace-learned visit prediction | `reproduction/scripts/run_online_speculative_execution.py` |
| Full PASTE reproduction | Existing native-prefix and live A/B/E/F Joint experiments | `reproduction/README.md` |
| Speculative Actions baseline | Fixed-trace, local Qwen3-8B next-tool prediction with exact verification | `reproduction/SPECULATIVE_ACTION_QWEN3_BASELINE.md` |

The online and trace paths use the same checksummed learned-rank model. The
model learns which displayed search-result ranks were historically selected,
then binds those ranks only to URLs in the current visible search response.
Speculative results stay isolated until an exact authoritative invocation
arrives.

## 1. Trace experiment

This path requires no GPU, model server, or network access:

```bash
bash reproduction/scripts/run_speculative_tool_execution.sh
```

Artifacts are written to `reproduction/artifacts/speculative_tool_execution/`.
The checked-in trace snapshot reproduces:

- top-5 transition hit rate: 76.47%;
- authoritative URL invocation hit rate: 55.68%;
- exposed tool stall: 38.514 s → 19.647 s;
- stall reduction: 48.99%.

A frozen contextual-reranker audit kept the legacy mapper as the deployed
choice: the challenger met the local overhead gate (`p99=11.35 ms`, maximum
`24.57 ms`) but changed exact Top-1/3/5 from
`19.3% / 43.2% / 55.7%` to `18.2% / 40.9% / 56.8%`, failing the protected
Top-1/3 criteria. See the
[optimization report](reproduction/results/predictor_optimization/REPORT.md).

To drive top-level Agent-session arrivals with the 2024 Azure LLM Inference
Trace while preserving the native multi-turn Agent calls and tool waits, use
the integrated adapter and follow the
[Azure LLM 2024 Agent trace guide](docs/AZURE_LLM_2024_AGENT_TRACE_GUIDE.md).

## 2. Murakkab-inspired adaptive-width ablation

The Murakkab paper does not currently link a runnable official artifact. The
repository therefore implements its directly testable ideas on the PASTE
search/visit workflow: a typed declarative DAG, offline configuration profiles,
SLO filtering, and resource-minimizing selection of PASTE's speculative
`top_k` knob.

```bash
python reproduction/scripts/run_murakkab_paste_comparison.py
```

The checked-in result is now explicitly classified as an offline PASTE
adaptive-`top_k` ablation. It is not a Murakkab-versus-PASTE comparison: the
SLOs are synthetic, the resource value is an admission-count proxy, and no
fixed Tongyi/vLLM/GPU deployment runs in this path. See the
[superseded report](reproduction/results/murakkab_paste/REPORT.md) and its
[replacement fixed-model protocol](reproduction/results/murakkab_paste/FIXED_MODEL_SAME_SETUP_PROTOCOL.md).

## 3. Online Qwen-DR speculative execution

Start an OpenAI-compatible Qwen DeepResearch server. The full pinned setup is:

```bash
bash reproduction/scripts/setup_env.sh
source reproduction/configs/model.env.example
bash reproduction/scripts/start_vllm.sh
```

Then run the autonomous online agent with trace-learned visit speculation:

```bash
"${HOME}/.conda/envs/paste/bin/python" \
  reproduction/scripts/run_online_speculative_execution.py \
  --output-dir reproduction/artifacts/online_trace_learned_smoke \
  --source-limit 2
```

The wrapper uses the checked-in mapper by default. It fixes
`call_graph_mode=autonomous` and `speculation_mode=visit`: the predictor only
selects work for the speculative queue, while Qwen remains authoritative over
all URLs returned by the current search response.

To inspect the delegated command without contacting a server:

```bash
python reproduction/scripts/run_online_speculative_execution.py \
  --output-dir /tmp/paste-online-dry-run \
  --dry-run
```

## 4. Full reproduction

The existing prefix-cache experiment, bounded live tool broker, Joint
physical-KV scheduler, workloads, frozen protocols, validators, and A/B/E/F
matrix are preserved. See [the full reproduction guide](reproduction/README.md).

## Repository layout

```text
PASTE-Qwen-DR/
├── reproduction/       # package, runners, tests, workloads, protocols/results
├── scripts/            # frozen live driver, learned-online driver, trace driver/hook
├── traces/my_traces/   # 100 historical training/evaluation sessions
├── requirements.txt    # pinned full GPU/live environment
├── requirements-cpu.txt
└── standalone-manifest.json
```

Validate an exported checkout with:

```bash
python reproduction/scripts/validate_standalone_repo.py \
  --repository-root . --require-manifest --smoke
```

Generated artifacts, logs, model weights, credentials, and machine-local vLLM
state are intentionally excluded from the repository.

## License

Apache License 2.0. See [LICENSE](LICENSE).

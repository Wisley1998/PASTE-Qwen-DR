# PASTE-Qwen-DR

Standalone reproduction of PASTE for Qwen DeepResearch. This repository keeps
three supported paths under one auditable layout:

| Path | Purpose | Entry point |
|---|---|---|
| Trace-learned tool execution | CPU-only, held-out causal evaluation without LLM co-design | `reproduction/scripts/run_speculative_tool_execution.sh` |
| Online speculative execution | Live Qwen-DR + live search/visit with trace-learned visit prediction | `reproduction/scripts/run_online_speculative_execution.py` |
| Full PASTE reproduction | Existing native-prefix and live A/B/E/F Joint experiments | `reproduction/README.md` |

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

To drive top-level Agent-session arrivals with the 2024 Azure LLM Inference
Trace while preserving the native multi-turn Agent calls and tool waits, see
the [Azure LLM 2024 Agent trace guide](docs/AZURE_LLM_2024_AGENT_TRACE_GUIDE.md).

## 2. Online Qwen-DR speculative execution

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

## 3. Full reproduction

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

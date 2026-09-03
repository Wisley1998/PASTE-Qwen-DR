# Failed Granite c5k one-shot audit

Audit date: 2026-08-31

## Outcome

The irreversible development-only attempt `comment3-granite33-8b-c5k-r1`
failed closed in its first cell, A. Six of 80 tasks failed output-contract
validation, so the A cell did not satisfy the 80-task completion gate. The E
cell was never created or started. The one-shot reservation and `failure.json`
both prohibit a rerun.

There is **no A/E latency result**. The incomplete A-cell timings are not a
portability effect estimate and must not be compared, pooled, or reported as
an A/E result.

This is a read-only derived audit. No live artifact was changed, no server or
GPU was started, and no request was replayed.

## Separate pre-live compatibility screen

These are offline tokenizer/config preflight outcomes, not live attempts and
not part of the A-cell evidence above. Reapplying the frozen runner's
`_offline_chat_context_preflight` with network disabled confirmed:

- `mistralai/Mistral-7B-Instruct-v0.3` at
  `c170c708c41dac9275d15a8fff4eca08d52bab71` was rejected because its native
  chat template rejects the unchanged agent role sequence.
- `microsoft/Phi-3-mini-128k-instruct` at
  `f3c06aed622e14ca0abf5115094e4fc9a9948f36` and
  `microsoft/Phi-4-mini-instruct` at
  `cfbefacb99257ffa30c83adab238a50856ac3083` were rejected because their
  pinned `config.json` files declare dynamic `auto_map` remote code, while the
  protocol accepts only built-in Transformers classes.
- `tiiuae/Falcon3-7B-Instruct` at
  `1e57a0ecd176c7c139f289c60a74e57f887c3dfb` and Granite at the revision
  audited below both exceeded the fixed 16k tokenizer/chat headroom gate under
  the default c12k profile, beginning at `formal-v8-001`.

The explicit Granite c5k profile was therefore selected before any live model
outcome. The frozen run plan records
`selected_and_hashed_before_any_live_execution=true`, labels it a
`pre_live_cross_architecture_compatibility_fallback`, and prohibits later
profile switching or cross-profile pooling. These offline rejections do not
consume or create live-result evidence.

## Exact identity and bindings

- Model: `ibm-granite/granite-3.3-8b-instruct`
- Revision: `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b`
- Profile: `c5k-l80-cross-architecture-fallback`
- Shape: four NVIDIA A100-SXM4-40GB GPUs (`4,5,6,7`), TP4, bfloat16,
  16,384-token served context, 5,000-token context padding, load 80
- Attempt key:
  `a7e2157b2c93dfb616aa9041b085b4522a5f5d8d6d1391eb337e5d1d399703c6`
- Run-plan canonical SHA:
  `5335388562b13f236d87ccd93b40f1368f14e32ac36cd34ee7f15e6b55afcb2c`
- Snapshot manifest SHA:
  `3e58f386a35c620614c71ce8b48da3ff1427f68d15244132ad402b4c3c4dd583`
- Snapshot content SHA:
  `de3e7a472eefcbf8efdd3eb6405d9f4ef4065d24394ae2af689ee82af4e9ade8`
- Cross-model runner SHA:
  `8db385dadc8b709711432a6402407070322c0a93be45fb426b588e47cfa106e9`
- Bound live-sensitivity runner SHA:
  `df1286308096455e53de31520db0fd73663f5b0d27cff7c77d92bc62d0e25180`
- Bound workload SHA:
  `780671d8a00b7528e80c959373c2493a04d3b47018dc818a7c6bfb33a0c828d4`

The run plan and cell contract contain identical profile, model, snapshot, and
dependency bindings. The plan SHA and snapshot-manifest SHA were independently
recomputed from canonical JSON. All 14 bound repository files still match
their recorded SHA values. All 15 snapshot files (16,346,537,590 bytes) were
fully rehashed and match the recorded manifest and content identity.

## Six failed tasks

All six failures are model-output contract failures (`ValueError`), not local
LLM HTTP failures or external transport failures.

| Source | LLM calls | Committed tools | Classification |
| --- | ---: | --- | --- |
| `formal-v8-010` | 3 | search, visit | Final call reached the fixed 192-token length. Its strict JSON object parsed, but it left no required nonempty ASCII-space tail (`tail_nonempty=false`, `padding_char_count=0`). |
| `formal-v8-011` | 3 | search, visit | Final call reached length with an unterminated JSON string; strict final-object parsing failed. |
| `formal-v8-012` | 3 | search, visit | Final call reached length with an unterminated JSON string; strict final-object parsing failed. |
| `formal-v8-015` | 3 | search, visit | Final call reached length with an unterminated JSON string; strict final-object parsing failed. |
| `formal-v8-022` | 3 | search, visit | Final call reached length with an unterminated JSON string; strict final-object parsing failed. |
| `formal-v8-036` | 2 | search | The second guided tool-call response was not one valid JSON object. The task therefore made no visit or final call. |

Thus 79 tasks made three LLM calls and one made two, producing 239 total
calls. Five failures occurred at final call index 2; one occurred at guided
tool-call index 1.

## Local LLM and tool transport audit

- Local LLM endpoint recorded by the result: `http://127.0.0.1:18124`.
- LLM events: `239/239` successful, `239/239` HTTP 200, and exactly one
  attempt for every event.
- The vLLM server log independently contains exactly 239 chat-completion POST
  entries, all HTTP 200 and none non-200.
- LLM call indices: 80 index-0, 80 index-1, and 79 index-2 calls.
- Finish reasons: 159 `stop`, 80 `length`. Output-contract validation is a
  separate gate; HTTP success does not make malformed model output valid.
- Tool-attempt ledger: 159 records: 80 search and 79 visit.
- All 159 are authoritative, non-speculative, actual-transport records with
  `committed=true`, `outcome=committed`, response status 200, and exactly one
  physical HTTP attempt.
- The 159 nested HTTP-attempt records all have status 200 and none has
  `retried=true`; therefore physical HTTP retry count is zero.
- Search transport: 80 `bing_html_search` calls to `www.bing.com`.
- Visit transport: 79 `r.jina.ai` calls to `r.jina.ai`.
- The minimum observed gap between the 79 physical visit starts is
  3.000062664 seconds, satisfying the fixed 3.0-second pacing gate.

`result.summary.tool.authoritative_commit_count` is 148 because that summary
counts tools attached to the 74 successful tasks (`74 × 2`). The complete
attempt ledger and broker counters include the 11 tools from failed tasks and
both independently record 159 executions/completions/commits. There is no
transport-count contradiction.

The completion-gate shortfalls are therefore internally consistent:

| Gate | Required | Observed |
| --- | ---: | ---: |
| Successful tasks | 80 | 74 |
| Successful local LLM calls | 240 | 239 |
| Authoritative tool commits | 160 | 159 |
| Failed tasks | 0 | 6 |
| Physical HTTP retries | 0 | 0 |

The malformed second call for `formal-v8-036` accounts for both the missing
visit commit and the missing final LLM call.

## Fail-closed lifecycle

- A used FCFS; its recorded scheduler environment contains only
  `VLLM_SCHED_POLICY=fcfs` and null treatment knobs.
- `runner.stderr.log` and `server_lifecycle.stderr.log` are empty.
- The server log contains no `Traceback`, `ERROR`, CUDA OOM, or
  `OutOfMemory` marker.
- Lifecycle evidence records SIGTERM followed by
  `vLLM pid 1097431 stopped cleanly`; the server log records all four TP NCCL
  communicators destroyed and `Application shutdown complete`.
- The E-cell directory is absent. `summary.json` and `completed_pair.json` are
  absent, as required for a failed pair.
- `failure.json` records
  `a-c5k-l80-cross-architecture-fallback live runner failed`, the same attempt
  key/profile, and `rerun_allowed=false`.
- The durable reservation exists at the profile-namespaced one-shot path with
  the same attempt key/run tag and `rerun_allowed=false`.

Because A returned nonzero, no strict cell validation or cell-completion
manifest was emitted. This absence is part of the fail-closed outcome, not
missing successful evidence.

## Raw artifact SHA-256 manifest

All paths below are relative to
`reproduction/artifacts/live_joint/development/comment3_cross_model/c5k-l80-cross-architecture-fallback/comment3-granite33-8b-c5k-r1/`.

| Path | SHA-256 |
| --- | --- |
| `run_plan.json` | `31aad879076849a928fa5c47c8c32eabb0cc0808035bd632fcfb23b2dad65539` |
| `model_snapshot_manifest.json` | `036bd91233ed665184dc8ead53749a8becb7b07f291520a1d1a9883c4d7e2a71` |
| `execution_hardware.json` | `1105cfacd2f88380fc341c9d8214368b53d936d724ea0a9e6f485d1177fa0445` |
| `failure.json` | `8992df6cae587f3a48eb7bcebae7a079e1af06afc01049ac17caff580a5d8236` |
| `cells/01-a-c5k-l80-cross-architecture-fallback/cell_contract.json` | `a3cca76d838f26ce4b72562cc293a0b375a243f1868181979d06fa2bb5b8c3bc` |
| `cells/01-a-c5k-l80-cross-architecture-fallback/evidence/result.json` | `99f254692416e773e05414056015d2efeee6d9053683d571dfdd72f03fb11026` |
| `cells/01-a-c5k-l80-cross-architecture-fallback/evidence/queue_timeline.jsonl` | `abf90435f53bc2083a4aff64197ffa9473c818327a955f32d9e0b597ae7b5370` |
| `cells/01-a-c5k-l80-cross-architecture-fallback/runner.stdout.log` | `7ef1012e1c0743fb472be5c892d326cb63e41ccfffc20552ee8ae3e1a548dd63` |
| `cells/01-a-c5k-l80-cross-architecture-fallback/runner.stderr.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `cells/01-a-c5k-l80-cross-architecture-fallback/server/vllm_18124.log` | `184fbc7669759e6173d0acbc31f3a2b7114a3cf2c5acf7b98c62e65f9b8b2ee1` |
| `cells/01-a-c5k-l80-cross-architecture-fallback/server_lifecycle.stdout.log` | `3f22e1562f38cd10adf722b162c6ddc914db662919bb651b0ac836123dfa0b22` |
| `cells/01-a-c5k-l80-cross-architecture-fallback/server_lifecycle.stderr.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The external durable reservation
`reproduction/artifacts/live_joint/development/comment3_cross_model/.one_shot_attempts/c5k-l80-cross-architecture-fallback/a7e2157b2c93dfb616aa9041b085b4522a5f5d8d6d1391eb337e5d1d399703c6/reservation.json`
has SHA-256
`b3888c260017bf013bd26c76ffdb0d9c32abda128e8e22e528cef1045e4aa708`.

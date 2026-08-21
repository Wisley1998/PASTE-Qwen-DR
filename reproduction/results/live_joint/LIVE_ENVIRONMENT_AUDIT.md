# Live experiment environment audit

Snapshot: 2026-08-16 UTC.  This is a read-only capability audit, not experiment
evidence; every accepted cell must capture its own fresh-server state.

## Compute and serving

- Eight NVIDIA A100-SXM4-40GB GPUs are visible.
- GPUs 0–3 host the pre-existing vLLM service on `127.0.0.1:8000`; its command
  line uses TP=4, max model length 16384, `max-num-seqs=16`, chunked prefill,
  and native prefix caching.
- GPUs 4–7 were free enough for a second TP=4 server and were being initialized
  for development live-loop runs on port 8100 during the snapshot.  This
  transient process is not a protocol cell.
- The vLLM model endpoint on port 8000 reported
  `Alibaba-NLP/Tongyi-DeepResearch-30B-A3B`.

## Tool paths and dependencies

- The `paste` environment contains `aiohttp`, `httpx`, `requests`, `bs4`, and
  `vllm`; it does not contain the Python `searx` package.
- No `SERPER_API_KEY`, `JINA_API_KEY`, `SEARXNG_URL`, or `VLLM_API_KEY` was set
  in the experiment shell.  The frozen Bing HTML and Jina HTTP paths require no
  private key.
- A service was bound to port 8888 on a non-loopback interface, but a 10-second
  JSON search probe returned no bytes.  It is therefore not treated as a
  usable SearXNG dependency and is outside the frozen backend.
- Wikimedia Action and REST calls reached the public host but returned HTTP
  429 under concurrent load.  Their two rejected runs and hashes are recorded
  in `ENGINEERING_RUN_LEDGER.md`.
- Bing HTML search and `r.jina.ai` visit are implemented as actual HTTP calls;
  each completed physical row records backend, request host, status, bytes,
  and attempt count.

## Existing paths rejected for causal measurement

Older repository search/visit code is useful product code but not a valid
closed-loop benchmark path by itself: tool schedulers may be per task, blocking
calls may use an unbounded default executor, batching semantics do not always
match the computed batch size, fuzzy same-domain caching can change the fetched
page, and older replayers contain trace results or recorded waits.  The live
experiment therefore uses the shared bounded broker and exact session-scoped
commit path exclusively.

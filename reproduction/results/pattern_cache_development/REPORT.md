# Pattern-cache development evaluation

| Policy | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|
| M0 current-only blind | 50/236 (21.2%) | 102/236 (43.2%) | 131/236 (55.5%) |
| Pattern cache, ungated | 50/236 (21.2%) | 105/236 (44.5%) | 134/236 (56.8%) |
| Pattern cache, gated dispatch | 50/236 (21.2%) | 105/236 (44.5%) | 134/236 (56.8%) |

Gate confusion: TP=116, FP=198, TN=26, FN=0; precision=36.9%, recall=100.0%.
The abstain rule fired 26 times and reduced URL dispatches 1700→1570 (-130, 7.6%).
Gate gated-minus-ungated exact-hit deltas: Top-1 +0, Top-3 +0, Top-5 +0.

Top-5 all-window precision/waste: M0 7.7%/1564; ungated 7.9%/1566; gated 8.5%/1436.

Strict historical 70-session grouped OOF:

| Policy | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|
| M0 current-only blind | 33/148 (22.3%) | 64/148 (43.2%) | 81/148 (54.7%) |
| Pattern cache, ungated | 33/148 (22.3%) | 65/148 (43.9%) | 81/148 (54.7%) |
| Pattern cache, gated dispatch | 33/148 (22.3%) | 65/148 (43.9%) | 81/148 (54.7%) |

Retrospective fixed 70→30 ranking-depth diagnostic (the split containing the quoted 19.3%/43.2%/55.7%):

| Policy | Top-1 | Top-3 | Top-5 | Top-10 | Top-20 |
|---|---:|---:|---:|---:|---:|
| M0 current-only blind | 17/88 (19.3%) | 38/88 (43.2%) | 49/88 (55.7%) | 62/88 (70.5%) | 67/88 (76.1%) |
| Pattern cache, ungated | 17/88 (19.3%) | 40/88 (45.5%) | 52/88 (59.1%) | 64/88 (72.7%) | 75/88 (85.2%) |
| Pattern cache, gated | 17/88 (19.3%) | 40/88 (45.5%) | 52/88 (59.1%) | 64/88 (72.7%) | 75/88 (85.2%) |

All-100 whole-session grouped-OOF ranking-depth diagnostic:

| Policy | Top-1 | Top-3 | Top-5 | Top-10 | Top-20 |
|---|---:|---:|---:|---:|---:|
| M0 current-only blind | 50/236 (21.2%) | 102/236 (43.2%) | 131/236 (55.5%) | 169/236 (71.6%) | 182/236 (77.1%) |
| Pattern cache, ungated | 50/236 (21.2%) | 105/236 (44.5%) | 134/236 (56.8%) | 171/236 (72.5%) | 189/236 (80.1%) |
| Pattern cache, gated | 50/236 (21.2%) | 105/236 (44.5%) | 134/236 (56.8%) | 171/236 (72.5%) | 189/236 (80.1%) |

Outer30 Top-10/20 candidate oracles: current-response Top-10=70, Top-20=70; bounded-cache Top-10=84, Top-20=84 (out of 88 targets).
At diagnostic depth, the gate would reduce all-window URL predictions Top-10 950→860 and Top-20 1848→1668.

Top-10/20 are offline non-dispatch diagnostics. The frozen v2 runtime still emits at most Top-5, and every expanded ranking is prefix-checked against that runtime output.

Candidate coverage among authoritative visit targets:

- Current response: 81.8%
- LRU64, age<=2, plus preserved M0 Top-1: 92.8%
- After gate: 92.8%

Local pattern prediction latency: p50=0.312 ms, p95=0.623 ms, p99=0.792 ms, max=18.170 ms.

Trace-wide non-executable visit URL labels: 8 (retained as exact-label misses when scored; never dispatched or cached).

The policy uses exact raw URL equality and no embedding, neural network, or backpropagation.

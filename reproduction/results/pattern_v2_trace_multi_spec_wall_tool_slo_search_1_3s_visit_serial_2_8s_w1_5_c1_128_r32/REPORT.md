# Pattern-v2 per-task multi-spec real-trace wall replay

Each admitted task decision may start multiple URL candidates concurrently.
Speculation uses isolated capacity; wrong-call contention is not modeled.
Sessions use event-driven list scheduling without lockstep decision barriers.

Widths=`[1, 2, 3, 4, 5]`, task concurrency=`[1, 2, 4, 8, 16, 32, 64, 128]`, LLM duration scale=`0.7`, coordination cost=`1.0 ms/start`.

## Per-task speculative width=1

| Task C | Full-trace wall speedup | Eligible-segment wall speedup | Mean full task-flow reduction | Visit-stall reduction | Exact hit rate | Call amp. | Slot upper bound |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.45% | 4.26% | 1.45% | 9.64% | 20.85% | 2.128x | 1 |
| 2 | 1.45% | 4.26% | 1.45% | 9.64% | 20.85% | 2.128x | 2 |
| 4 | 1.42% | 3.87% | 1.45% | 9.64% | 20.85% | 2.128x | 4 |
| 8 | 1.37% | 1.22% | 1.45% | 9.64% | 20.85% | 2.128x | 8 |
| 16 | 0.87% | 0.57% | 1.45% | 9.64% | 20.85% | 2.128x | 16 |
| 32 | 0.17% | 0.19% | 1.45% | 9.64% | 20.85% | 2.128x | 32 |
| 64 | 0.05% | 0.02% | 1.45% | 9.64% | 20.85% | 2.128x | 64 |
| 128 | -0.00% | -0.00% | 1.45% | 9.64% | 20.85% | 2.128x | 100 |

## Per-task speculative width=2

| Task C | Full-trace wall speedup | Eligible-segment wall speedup | Mean full task-flow reduction | Visit-stall reduction | Exact hit rate | Call amp. | Slot upper bound |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.58% | 7.57% | 2.58% | 17.15% | 35.74% | 3.315x | 2 |
| 2 | 2.58% | 7.52% | 2.58% | 17.15% | 35.74% | 3.315x | 4 |
| 4 | 2.55% | 6.90% | 2.58% | 17.15% | 35.74% | 3.315x | 8 |
| 8 | 2.52% | 2.36% | 2.58% | 17.15% | 35.74% | 3.315x | 16 |
| 16 | 1.76% | 1.22% | 2.58% | 17.15% | 35.74% | 3.315x | 32 |
| 32 | 0.56% | 0.58% | 2.58% | 17.15% | 35.74% | 3.315x | 64 |
| 64 | 0.32% | 0.30% | 2.58% | 17.15% | 35.74% | 3.315x | 128 |
| 128 | 0.24% | 0.26% | 2.58% | 17.15% | 35.74% | 3.315x | 200 |

## Per-task speculative width=3

| Task C | Full-trace wall speedup | Eligible-segment wall speedup | Mean full task-flow reduction | Visit-stall reduction | Exact hit rate | Call amp. | Slot upper bound |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.28% | 9.63% | 3.28% | 21.81% | 41.70% | 4.591x | 3 |
| 2 | 3.32% | 9.53% | 3.28% | 21.81% | 41.70% | 4.591x | 6 |
| 4 | 3.29% | 8.76% | 3.28% | 21.81% | 41.70% | 4.591x | 12 |
| 8 | 3.21% | 3.08% | 3.28% | 21.81% | 41.70% | 4.591x | 24 |
| 16 | 2.40% | 1.71% | 3.28% | 21.81% | 41.70% | 4.591x | 48 |
| 32 | 0.87% | 0.94% | 3.28% | 21.81% | 41.70% | 4.591x | 96 |
| 64 | 0.59% | 0.57% | 3.28% | 21.81% | 41.70% | 4.591x | 192 |
| 128 | 0.50% | 0.54% | 3.28% | 21.81% | 41.70% | 4.591x | 300 |

## Per-task speculative width=4

| Task C | Full-trace wall speedup | Eligible-segment wall speedup | Mean full task-flow reduction | Visit-stall reduction | Exact hit rate | Call amp. | Slot upper bound |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.47% | 13.12% | 4.47% | 29.71% | 52.77% | 5.817x | 4 |
| 2 | 4.50% | 12.98% | 4.47% | 29.71% | 52.77% | 5.817x | 8 |
| 4 | 4.43% | 11.71% | 4.47% | 29.71% | 52.77% | 5.817x | 16 |
| 8 | 4.34% | 4.00% | 4.47% | 29.71% | 52.77% | 5.817x | 32 |
| 16 | 3.06% | 2.26% | 4.47% | 29.71% | 52.77% | 5.817x | 64 |
| 32 | 1.12% | 1.11% | 4.47% | 29.71% | 52.77% | 5.817x | 128 |
| 64 | 0.62% | 0.60% | 4.47% | 29.71% | 52.77% | 5.817x | 256 |
| 128 | 0.50% | 0.54% | 4.47% | 29.71% | 52.77% | 5.817x | 400 |

## Per-task speculative width=5

| Task C | Full-trace wall speedup | Eligible-segment wall speedup | Mean full task-flow reduction | Visit-stall reduction | Exact hit rate | Call amp. | Slot upper bound |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.04% | 14.80% | 5.04% | 33.51% | 57.02% | 7.111x | 5 |
| 2 | 5.06% | 14.66% | 5.04% | 33.51% | 57.02% | 7.111x | 10 |
| 4 | 4.99% | 13.06% | 5.04% | 33.51% | 57.02% | 7.111x | 20 |
| 8 | 4.88% | 4.45% | 5.04% | 33.51% | 57.02% | 7.111x | 40 |
| 16 | 3.38% | 2.49% | 5.04% | 33.51% | 57.02% | 7.111x | 80 |
| 32 | 1.15% | 1.20% | 5.04% | 33.51% | 57.02% | 7.111x | 160 |
| 64 | 0.67% | 0.60% | 5.04% | 33.51% | 57.02% | 7.111x | 320 |
| 128 | 0.50% | 0.54% | 5.04% | 33.51% | 57.02% | 7.111x | 500 |

The full-trace scope includes recorded search waits and every LLM turn.
The eligible segment includes only search-decision LLM lead and immediate
measurable visit stall. Multi-URL authority is replayed serially; concurrent
exact speculations keep progressing during the LLM lead and while earlier
authority URLs execute. Corrected traces provide per-URL service samples;
legacy traces use equal-share atomic service as a fallback.

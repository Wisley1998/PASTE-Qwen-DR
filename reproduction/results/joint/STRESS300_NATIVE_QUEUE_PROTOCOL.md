# Stress300 native-vLLM-queue A-only protocol

## Purpose

This is a prospective A-only load-selection probe, not an A/D comparison.
It tests whether 300 offered traces create a useful native vLLM resource queue
while the configured sequence-count cap is strictly above the offered request
concurrency.  Joint data must not be run or observed when accepting or
rejecting this load.

The 60 heldout source sessions appear exactly five times each.  These copies
increase offered load but do not increase the independent sample size beyond
60.  Recorded waits are replayed at 10x speed; this remains a recorded-wait
replay rather than a live-tool workload.

## Frozen engine shape and queue attribution

- `PASTE_MAX_ACTIVE_TRACES=300`, matching the 300 workload traces.  Each trace
  issues at most one LLM request at a time, so 300 is the offered
  request-concurrency upper bound.
- `VLLM_MAX_NUM_SEQS=320`, leaving 20 requests of structural headroom.  The
  sequence-count setting therefore cannot create the queue.
- `VLLM_CUDA_GRAPH_SIZES=256` is an independently configured CUDA-graph
  capture shape, not an admission ceiling.
- The model and revision, tensor parallelism 4, bfloat16, 16K model context,
  GPU-memory utilization 0.86, 8,192 batched-token budget, 512 output-token
  cap, and vLLM v1 are unchanged from the earlier native-queue probes.
- `VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION=1` is frozen prospectively.  A future
  Joint policy could reorder waiting requests but cannot install a private
  running-request cap; vLLM retains admission and preemption control.

The post-run validator calls the queue native only if configuration headroom
is positive, no timeline sample reaches 320 running requests, native admission
is established, and waiting is observed while running is below 320.  This
proves a native resource queue, but it does not by itself identify whether the
batched-token budget or physical KV availability is the dominant resource.

## A-only preregistered acceptance gates

The `stress300_native320_g256_u86_a_probe` profile accepts only `--cells A`.
Passing `A,D`, `D,A`, or `D` fails before output creation or server startup.
All of these gates are evaluated using A alone:

1. all logical requests succeed exactly once;
2. no CPU KV-cache swap occurs;
3. the natural-queue proof above passes;
4. at least 50% of timeline samples have `waiting>0` and `running<320`;
5. mean queue time is at least 20% of mean request latency; and
6. native recomputation preemptions per logical request are at most 0.25.

These thresholds are carried forward unchanged from the preregistered
stress240 A probe.  They must not be relaxed after observing stress300 A, and
stress300 must never be selected based on an A/D gain.

## Build and validation

Build the deterministic five-copy workload from the authoritative heldout60
manifest:

```bash
HF_HOME=/home/aiscuser/hf_cache \
  /home/aiscuser/.conda/envs/paste/bin/python \
  reproduction/scripts/build_stress_duplicate_workloads.py \
  --manifest reproduction/artifacts/workloads/fixed_three_way_cap512_floor64/manifest_heldout60.json \
  --output-root reproduction/artifacts/workloads/fixed_three_way_cap512_floor64/stress300 \
  --manifest-out reproduction/artifacts/workloads/fixed_three_way_cap512_floor64/manifest_stress300.json \
  --load-instance-count 300
```

Validate the frozen profile and workload without starting a server or using a
GPU:

```bash
bash reproduction/scripts/run_joint_stress_pair.sh stress300_check \
  --config reproduction/configs/joint_stress300_u86_native320_a_probe.env.example \
  --cells A --gpus 4,5,6,7 --port 8100 --check-only
```

After check-only succeeds, the exact A-only run command is:

```bash
bash reproduction/scripts/run_joint_stress_pair.sh stress300_a_probe_r1 \
  --config reproduction/configs/joint_stress300_u86_native320_a_probe.env.example \
  --cells A --gpus 4,5,6,7 --port 8100
```

Choose an actually idle four-GPU set and free localhost port before executing
the final command.  The wrapper writes `natural_queue_probe.json` before
enforcing the gates so a rejected A probe remains diagnosable.  No stress300 D
profile is authorized by this protocol; any later D experiment requires a
separate, prospectively frozen decision after A-only load acceptance.

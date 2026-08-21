# Stress240 native-vLLM-queue A-only protocol

## Purpose

This is a load-selection probe, not an A/D result.  It asks whether a
240-task replay creates a useful, stable native vLLM resource queue while the
configured sequence-count cap is mathematically non-binding.  Joint data must
not be observed when accepting or rejecting this load.

The 60 heldout source sessions are represented exactly four times each.  The
copies increase offered load, not statistical sample size: the independent
sample count remains 60.  Recorded waits are replayed at 10x speed; this is
not a live-tool workload.

## Frozen shape and queue attribution

- `PASTE_MAX_ACTIVE_TRACES=240` and the workload contains 240 traces.  A trace
  issues at most one LLM request at a time, so 240 is the offered request-
  concurrency upper bound.
- `VLLM_MAX_NUM_SEQS=256`, leaving 16 requests of structural headroom.
- `VLLM_CUDA_GRAPH_SIZES=256` fixes a CUDA-graph capture shape; it is not an
  admission limit.
- `VLLM_GPU_MEMORY_UTILIZATION=0.86`, 16K context, 8,192 batched-token budget,
  tensor parallelism 4, and vLLM v1 remain fixed.
- `VLLM_SCHED_JOINT_V2_NATIVE_ADMISSION=1` is frozen prospectively.  If D is
  later promoted, Joint can reorder the waiting queue but cannot lower
  `max_num_running_reqs`; native vLLM owns admission and preemption.

The post-run validator calls the queue native only if configuration headroom
is positive, no timeline sample reaches the sequence cap, native admission is
established, and waiting is observed while running is strictly below the cap.
This proves a native resource queue but does not by itself distinguish the
batched-token budget from physical KV availability as its dominant cause.

## A-only preregistered acceptance gates

The profile `stress240_native256_g256_u86_a_probe` accepts only `--cells A`.
Passing `A,D` or `D` fails before output creation or server startup.

All of the following must pass before any D run is authorized:

1. all logical requests succeed exactly once;
2. no CPU KV-cache swap occurs;
3. the natural-queue proof above passes;
4. at least 50% of timeline samples have `waiting>0` and `running<256`;
5. mean queue time is at least 20% of mean request latency; and
6. native recomputation preemptions per logical request are at most 0.25.

The queue proof and the preemption gate answer different questions.
Recomputation preemption is itself a native response to KV pressure and does
not invalidate a below-cap queue.  However, a rate above 0.25 rejects
stress240 as a primary latency-comparison workload because a later gain could
mainly reflect avoiding pathological recomputation churn.  Such a run may
still be reported explicitly as an overload diagnostic.

The queue-strength thresholds were fixed from A-side behavior before running
stress240 D; they must not be relaxed after seeing an A/D gain.

## Reproduction order

Build the checksummed workload once from the authoritative heldout60 bundle:

```bash
HF_HOME=/home/aiscuser/hf_cache \
  /home/aiscuser/.conda/envs/paste/bin/python \
  reproduction/scripts/build_stress_duplicate_workloads.py \
  --manifest reproduction/artifacts/workloads/fixed_three_way_cap512_floor64/manifest_heldout60.json \
  --output-root reproduction/artifacts/workloads/fixed_three_way_cap512_floor64/stress240 \
  --manifest-out reproduction/artifacts/workloads/fixed_three_way_cap512_floor64/manifest_stress240.json \
  --load-instance-count 240
```

Validate without a server or GPU:

```bash
bash reproduction/scripts/run_joint_stress_pair.sh stress240_check \
  --config reproduction/configs/joint_stress240_u86_native256_a_probe.env.example \
  --cells A --gpus 4,5,6,7 --port 8100 --check-only
```

Then run exactly one A-only load-selection probe on an otherwise idle GPU set:

```bash
bash reproduction/scripts/run_joint_stress_pair.sh stress240_a_probe_r1 \
  --config reproduction/configs/joint_stress240_u86_native256_a_probe.env.example \
  --cells A --gpus 4,5,6,7 --port 8100
```

The wrapper writes `natural_queue_probe.json` before evaluating the gates, so
a rejected run remains diagnosable.

## Accepted-A D-only screen

Only after the A producer has exited successfully may the selected policy be
screened on stress240.  The D profile is fixed to the stress180-selected
exact-stage policy with the sparse 120-second rescue:

- `GATE_MAX_WAIT_S=120` and `DEADLINE_MIN_RUNNING=48`;
- final and exact remaining-call lanes enabled;
- coarse lanes, soft-stage weight, and running priority disabled; and
- native admission enabled, with target/max 256 (non-operative as private
  capacity controls under native admission).

Supply the accepted artifact as a repository-relative path.  A missing,
partial, absolute, or repository-escaping path fails before output creation or
server startup:

```bash
PASTE_ACCEPTED_A_PROBE="reproduction/artifacts/stress240_u86_native256_g256_a_probe/<accepted-A-tag>/natural_queue_probe.json" \
  bash reproduction/scripts/run_joint_stress_pair.sh stress240_d_screen_r1 \
  --config reproduction/configs/joint_stress240_u86_native256_exact_rescue120.env.example \
  --cells D --gpus 4,5,6,7 --port 8100 --check-only
```

Remove `--check-only` only after that command succeeds.  The D wrapper does
not trust the saved JSON alone: it recomputes the natural-queue probe from the
referenced sibling `*_fcfs_none` cell and requires field-for-field identical
evidence.  It then rechecks profile/load=240/max-num-seqs=256, natural queue,
exactly-once completion, no CPU KV swap, the 0.50/0.20/0.25
wait/queue/preemption gates, and these A-vs-D engine-shape fields:
model ID and revision, TP size, dtype, max model length, GPU-memory utilization,
batched-token budget, max-num-seqs, CUDA-graph shape, and the vLLM v1 flag.
The D run snapshots the accepted JSON, its SHA-256, and the structured
validation result, then repeats validation after D drains to detect any change
to the accepted probe or its source cell during the screen.

This accepted-A/D-only result is a candidate screen, not a formal paired
estimate: it reuses the load-selection A and therefore is vulnerable to
run-to-run server noise.  If D is promising, freeze pair-level mean and tail
gates and run fresh A and D servers.  The 240 copies still provide only 60
independent source-level units for uncertainty estimates.

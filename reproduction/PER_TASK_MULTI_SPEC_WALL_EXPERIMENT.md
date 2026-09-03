# Per-task multi-spec real-trace wall 实验

更新日期：2026-09-02

## 结论

原始 trace 的 search 和 visit timing 都不满足真实工具口径：search 有约 `10 s/query` 的串行
膨胀及 `7277 s` 异常点；188 个可测 visit batch 的 p50 仅 `0.500 s`，其中 65 个只有
`25--30 ms`。本实验从原始 100-session corpus 一次性生成新的 tool-SLO 派生 trace：

- 每个 search call 独立采样 `Uniform(1,3 s)`；
- 每个 visit URL 独立采样 `Uniform(2,8 s)`，multi-URL batch 按 URL 原顺序串行求和；
- 所有后续 timestamps 按 duration delta 累积平移；
- 69 个没有后续 LLM 的 terminal tool 添加显式 completion marker，不再按零时长处理。

修复后，`0.70x LLM` E2E 中 LLM/search/visit 分别占 `57.03%`/`9.65%`/`32.80%`，tool
合计占 `42.47%`。允许每个 task decision 并发启动 Top-N speculative URLs 后，W1 到 W5 的
完整 task-flow reduction 从 `1.45%` 增至 `5.04%`，eligible visit-stall reduction 从
`9.64%` 增至 `33.51%`。

W5 在 C1/C8/C16/C64/C128 的 workload wall speedup 分别为 `5.04%`、`4.88%`、
`3.38%`、`0.67%`、`0.50%`。高并发 makespan 仍由 critical sessions 决定，因此不能用
workload makespan 替代 mean task-flow reduction。

## Tool-SLO trace 修复审计

原始 timing 的问题：

- 340 个可测 search 合计 `19293.504 s`、p50 `20.046 s`、最大 `7277.283 s`。
- 188 个可测 visit batch 含 423 个 URL，合计仅 `375.412 s`、batch p50 `0.500 s`；
  149/188 个 batch 小于 2 秒，171/188 个低于 `2 s × URL count`。
- 41 个 terminal visit（84 URLs）和 28 个 terminal search 没有 completion timestamp，旧
  wall 口径把它们的 service 记成零。

修复后的 timing：

- 368 个 search call：合计 `743.865 s`，均值 `2.021 s`，范围
  `1.0128--2.9946 s`。
- 229 个 visit batch、507 个 URL：串行 batch 合计 `2529.524 s`；单 URL 均值
  `4.989 s`、p50 `5.039 s`、p95 `7.727 s`、范围 `2.0318--7.9987 s`。
- Batch 均值 `11.046 s`、p50 `6.334 s`、p95 `34.506 s`、最大 `62.987 s`。
- 69 个 terminal tool completion marker 已加入 session wall。
- 随机数由 `SHA-256(seed, session filename, event index, call index, tool, unit index)`
  决定，固定 seed 为 `qwen-tool-slo-uniform-v1`；重复运行逐值相同。
- 100 个 session 的原事件 payload、tool arguments、LLM durations、predictor inputs 和
  authority labels 保持一致；只增加 timing metadata 与 terminal completion marker。
- 597 个修复工具的 timestamp gap 与抽样 duration 逐项一致，最大误差
  `3.44e-13 s`；全部 timestamps 单调。

## 修复后的时间构成

| Component | Raw time | Raw share | `0.70x LLM` time | Experiment share |
|---|---:|---:|---:|---:|
| LLM | `6271.371 s` | `65.43%` | `4397.821 s` | `57.03%` |
| Search | `743.865 s` | `7.76%` | `743.865 s` | `9.65%` |
| Visit | `2529.524 s` | `26.39%` | `2529.524 s` | `32.80%` |
| Other tool | `1.489 s` | `0.02%` | `1.489 s` | `0.02%` |
| Residual | `38.805 s` | `0.40%` | `38.805 s` | `0.50%` |
| Total | `9585.054 s` | `100%` | `7711.503 s` | `100%` |

Tool 内部 search/visit/other 分别占 `22.71%`/`77.24%`/`0.05%`。

## 实验口径

- Predictor 使用 Pattern-v2 nested whole-session OOF probability 和 OOF service estimate。
- 每个 admitted decision 同时启动 expected value 为正的 Top-N，`N={1,2,3,4,5}`。
- 每个 active task 假设拥有 N 个隔离 speculative slots；wrong speculation 不阻塞 authority。
  因此这是无 tool contention 的收益上界，同时报告 slot 上界和 physical-call amplification。
- Session 使用 event-driven closed-loop list scheduling，不设置跨 task lockstep barrier。
- Trace 输入为
  `traces/my_traces_tool_slo_search_uniform_1_3s_visit_serial_uniform_2_8s`。
- `full-trace wall` 包含完整 session span、terminal tool completion，并统一应用 `0.70x`
  LLM-duration counterfactual。
- `eligible-segment wall` 只包含 search-decision LLM lead 和紧随其后的 authority visit。
- Runner 直接读取 trace 中每个 URL 的 sampled service，按 authority URL 原顺序串行重放。
  Spec candidates 并发推进：遇到 miss 时 authority 执行该 URL；遇到 exact hit 时只等待 Spec
  尚未完成的尾部，因此后续 hit 也能利用 earlier authority URL 的执行时间。
- 当前 Spec 只预测 visit URL；search 属于 predictor 的因果输入，不在可省范围内。

## 增加每个 task 的 speculative width

下面固定 C1，以消除 workload critical-path effects：

| Width | Selected | Exact hits | Net saved visit stall | Eligible visit reduction | Eligible wall speedup | Full wall / mean flow reduction | Call amp. |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 314 | 49 | `111.806 s` | `9.64%` | `4.26%` | `1.45%` | `2.128x` |
| 2 | 628 | 84 | `198.807 s` | `17.15%` | `7.57%` | `2.58%` | `3.315x` |
| 3 | 942 | 98 | `252.888 s` | `21.81%` | `9.63%` | `3.28%` | `4.591x` |
| 4 | 1256 | 124 | `344.481 s` | `29.71%` | `13.12%` | `4.47%` | `5.817x` |
| 5 | 1570 | 134 | `388.527 s` | `33.51%` | `14.80%` | `5.04%` | `7.111x` |

Eligible authority visit wall 为 `1159.531 s`，占 scaled full wall 的 `15.04%`。W5 省掉
其中 `33.51%`，也就是全部 visit wall 的 `15.36%`。W4 到 W5 多启动 314 个候选，增加
10 个 exact hits 和 `44.046 s` 净 saving，但 call amplification 从 `5.817x` 增至
`7.111x`。

## W5 wall concurrency sweep

| Task C | Full-trace wall speedup | Eligible-segment wall speedup | Mean task-flow reduction | Isolated slot upper bound |
|---:|---:|---:|---:|---:|
| 1 | `5.038%` | `14.795%` | `5.038%` | 5 |
| 2 | `5.057%` | `14.658%` | `5.038%` | 10 |
| 4 | `4.994%` | `13.057%` | `5.038%` | 20 |
| 8 | `4.884%` | `4.454%` | `5.038%` | 40 |
| 16 | `3.379%` | `2.491%` | `5.038%` | 80 |
| 32 | `1.153%` | `1.198%` | `5.038%` | 160 |
| 64 | `0.667%` | `0.604%` | `5.038%` | 320 |
| 128 | `0.498%` | `0.535%` | `5.038%` | 500 |

`mean task-flow reduction` 是各 session duration 的总和变化，与 C 无关；workload wall 是
closed-loop list-scheduling makespan，会随 critical path 改变。

## 判读

1. 之前的 `1.405%` full-flow 结果确实由错误 visit timing 压低；修复后 W5 为 `5.038%`。
2. visit 自身的收益已经达到预期量级：W5 减少 `33.51%` eligible visit stall，eligible
   segment wall speedup 为 `14.80%`。
3. 当前仍不能达到 `20--40%` full E2E，因为 Spec 只覆盖占 full wall `15.04%` 的 eligible
   visit，且不覆盖 search。若要达到更高 E2E，需要扩展可 Spec 的 tool scope 或显著提高
   eligible visit coverage；不能再靠修改 visit duration 实现。
4. W4/W5 的 `5.8--7.1x` call amplification 很高；实际 bounded tool capacity 下还需用
   per-URL start/end/queue telemetry 测 wrong-call contention。

## 复现

生成 timestamp-consistent tool-SLO trace：

```bash
python reproduction/scripts/correct_trace_tool_slos.py \
  --input-dir traces/my_traces \
  --output-dir traces/my_traces_tool_slo_search_uniform_1_3s_visit_serial_uniform_2_8s \
  --search-min-s 1 --search-max-s 3 \
  --visit-min-s 2 --visit-max-s 8 \
  --seed qwen-tool-slo-uniform-v1
```

运行 multi-spec wall runner：

```bash
python reproduction/scripts/run_pattern_v2_trace_multi_spec_wall.py \
  --traces traces/my_traces_tool_slo_search_uniform_1_3s_visit_serial_uniform_2_8s \
  --per-task-widths 1 2 3 4 5 \
  --concurrencies 1 2 4 8 16 32 64 128 \
  --repetitions 32 \
  --llm-duration-scale 0.70 \
  --domain-prior-strength 10 \
  --coordination-cost-ms 1.0 \
  --output-dir reproduction/results/pattern_v2_trace_multi_spec_wall_tool_slo_search_1_3s_visit_serial_2_8s_w1_5_c1_128_r32
```

Artifacts：

- `traces/my_traces_tool_slo_search_uniform_1_3s_visit_serial_uniform_2_8s/CORRECTION_MANIFEST.json`
- `reproduction/results/pattern_v2_trace_multi_spec_wall_tool_slo_search_1_3s_visit_serial_2_8s_w1_5_c1_128_r32/REPORT.md`
- `reproduction/results/pattern_v2_trace_multi_spec_wall_tool_slo_search_1_3s_visit_serial_2_8s_w1_5_c1_128_r32/metrics.json`
- `reproduction/tests/test_correct_trace_tool_slos.py`
- `reproduction/tests/test_pattern_v2_trace_timing_net_benefit.py`
- `reproduction/tests/test_pattern_v2_trace_multi_spec_wall.py`

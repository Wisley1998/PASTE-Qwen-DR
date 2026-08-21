# Formal-v8 live tool–LLM 联合闭环诊断报告

## 结论与状态

`formal-v8-context10k-live-r1` **没有通过 formal promotion**。严格聚合器记录
`formal_promotion_passed=false`；36 个预注册 gate 中 34 个通过，且仅有两个失败：

1. `E_to_F_mean_reduction`：实测 `3.336224%`，低于预注册的 `5%`；
2. `all_cells_authoritative_retry_rate_at_most_2pct`：A/B/E/F 的聚合权威调用重试率分别为
   `4.1667% / 4.5833% / 5.0000% / 4.7917%`，均高于 `2%`。

因此，本轮可以作为方向一致、机制闭合的诊断证据，但不能表述为“正式通过”、
“已经最优”或“潜力已经穷尽”。特别是，E→F 的 paired-source bootstrap 区间虽严格为正，
但预注册的效果量门槛仍然失败；不能在看到结果后降低门槛。

主证据是 [strict aggregate](../../artifacts/live_joint/formal/formal-v8-context10k-live-r1/strict_four_cell_aggregate.json)
（SHA256 `acdd8fb2723d23e38d55d21bb4590c9bad5ff0e8ceef30282604aa962a0dff56`）和
[completed matrix](../../artifacts/live_joint/formal/formal-v8-context10k-live-r1/completed_matrix.json)
（SHA256 `3e965f75264f38e37b91755098266c4027fc31deddbe427dec9b4576b7371ad3`）。
以下数值均来自这两个冻结文件，或来自 completed matrix 所列 12 个 cell 的 SHA 绑定原始证据；
没有把运行中的观察代替最终聚合结果。

## 设计与样本口径

四个 cell 的受控因素如下。四者都启用相同的 native vLLM prefix cache，且都关闭显式
prefix-locality reorder。

| Cell | LLM serving | Tool execution |
|---|---|---|
| A | FCFS native | demand only |
| B | FCFS native | visit speculation |
| E | Joint physical-KV | demand only |
| F | Joint physical-KV | visit speculation |

三块顺序为 `A-B-E-F`、`B-A-F-E`、`A-B-F-E`，A/B 与 E/F 都包含正反顺序。
80 个独立 frozen source 在每个 cell、每块各运行一次；fresh block 是同一 source 的重复测量，
不是额外独立样本。效果估计先对每个 source 的三块观测取均值，再以 80 个 source 为 bootstrap
抽样单位（10,000 次，seed `20260816`）。

完整矩阵包含：

- `960` 个成功 task（80 source × 4 cell × 3 block）；
- `2,880` 个成功逻辑 LLM request（每 task 三次）；
- `1,920` 个 exact authoritative tool commit（每 task 一次 Bing search、一次 Jina visit）；
- `12` 个互不相同的 fresh vLLM server instance，逐 cell 空 cache、空 broker 启动并 drain；
- `960/960` 个 call-2 都严格为 `192` completion tokens，wire output 均可解析为唯一
  `{answer, source_url}` JSON 加非空 ASCII-space tail，且 guided-JSON recovery 为零。

冻结 workload 为
[live_joint_wikipedia_frozen_formal_v8.json](../../workloads/live_joint_wikipedia_frozen_formal_v8.json)，
文件 SHA256 为 `780671d8a00b7528e80c959373c2493a04d3b47018dc818a7c6bfb33a0c828d4`；
canonical source SHA256 为
`01b029c3427f5f04d4f1b83b4f9b13e5decd705e773ffdeaeebb15970150f0df`。

## 效果量、区间与方向

“更快 source”按每个 source 跨三块的配对均值计算。括号中的 block 项依次为三块的
相对降低及块内更快 source 数。95% CI 是 paired-source bootstrap 相对降低区间。

| 对比 | Mean E2E（baseline→candidate） | 降低 | 95% CI | 三块方向（降低；更快数） | 聚合更快 source |
|---|---:|---:|---:|---|---:|
| A→B | `146.2214→133.3163 s` | `12.9051 s` / `8.8257%` | `[5.6104%, 11.6988%]` | `10.0860%;48` / `8.2040%;44` / `8.1568%;45` | `46/80` |
| A→E | `146.2214→106.1520 s` | `40.0694 s` / `27.4033%` | `[24.7627%, 30.1855%]` | `28.5521%;80` / `26.8961%;79` / `26.7337%;79` | `80/80` |
| E→F | `106.1520→102.6105 s` | `3.5415 s` / `3.3362%` | `[2.8770%, 3.7995%]` | `3.4856%;71` / `3.3807%;66` / `3.1425%;62` | `70/80` |
| A→F | `146.2214→102.6105 s` | `43.6109 s` / `29.8252%` | `[27.2284%, 32.5831%]` | `31.0425%;79` / `29.3676%;80` / `29.0361%;80` | `80/80` |

四个 cell 的 tail 与整 cell makespan 如下；P95/P99 汇总 240 个 task observation/cell，
makespan 为三块均值。

| Cell | Task P50 | Task P95 | Task P99 | Mean makespan |
|---|---:|---:|---:|---:|
| A | `137.6109 s` | `220.7823 s` | `231.2672 s` | `236.8448 s` |
| B | `125.8984 s` | `180.1804 s` | `190.1272 s` | `197.0507 s` |
| E | `101.5152 s` | `185.8412 s` | `195.5151 s` | `201.7207 s` |
| F | `97.3021 s` | `180.1334 s` | `190.5563 s` | `197.3585 s` |

E→F 的 task P95 与 makespan 都改善，分别为 `185.8412→180.1334 s` 和
`201.7207→197.3585 s`；相应 safety gates 通过。A→F 的 combined point estimate
达到 `29.8252%`，80/80 source 更快，LLM request P99 为 `47.2806→50.5427 s`
（`1.0690×`，低于 `1.25×` gate）。但 formal promotion 是合取条件，不能用这些通过项
覆盖上述两个失败项。

A→B 与 A→E 是 reported-only 诊断，不是 promotion gate。交互项定义为
`(E-F)-(A-B)`，结果为 `-9.3636 s`，95% CI `[-14.3030, -4.4439] s`；这表示在本负载下，
speculation 的增量收益在 Joint 中小于在 FCFS 中，不能称为正协同或 synergy。

## E→F 机制归因

聚合器按每 task 的三次 LLM duration、已提交 search/visit exposed wait 与 orchestration
residual 做闭合分解：

| 每 task 均值分量 | E | F | E−F（正数为节省） |
|---|---:|---:|---:|
| LLM | `43.4215 s` | `45.4033 s` | `-1.9818 s` |
| Tool exposed wait | `62.5980 s` | `57.0740 s` | `+5.5240 s` |
| 其中 search exposed wait | `0.2205 s` | `0.2507 s` | `-0.0302 s` |
| 其中 visit exposed wait | `62.3775 s` | `56.8232 s` | `+5.5542 s` |
| Orchestration residual | `0.1325 s` | `0.1332 s` | `-0.0007 s` |
| E2E | `106.1520 s` | `102.6105 s` | `+3.5415 s` |

F 的 LLM 分量并没有“偷偷变快”，而是比 E **慢 `4.5641%`**。净 E2E 节省来自
`5.5240 s` tool exposed-wait 节省，其大小是 `3.5415 s` 净节省的 `1.5598×`；LLM
回退吃掉了其中 `1.9818 s`。因此 E→F 的正向点估计可以归因于真实 tool wait 被隐藏，
而不能归因于更少的 LLM 工作。

F 在 `438` 个可计入的 authoritative commit 中得到 `198` 个 exact speculative hit，
hit rate 为 `45.2055%`，wasted speculative worker fraction 为 `0`。更具体地，三块 F
分别是 `63/64/63` 次 queued promotion 和 `3/2/3` 次 completed reuse：合计
`190/198`（`95.96%`）的 hit 是把仍在共享队列中的 speculative job 提升为 authoritative，
只有 `8/198` 已经先完成；running promotion 为零。故这里的主机制是 **queue promotion**，
不是大量提前占用 worker 后缓存成品。B/F 的 speculative worker waste、cancel、failure
均为零。

## 自然双队列压力与“64 上限”检查

三个 A baseline 都自然提供 80 个并发 task，满足 `80 > 64` 且 `80 < max_num_seqs=96`；
因此 96 是非绑定 native ceiling，而不是把负载固定在 64。三个 A 块分别观测到：

| Block | Native LLM waiting（running<96） | Authoritative tool queue | 双队列同时压力样本 | 最长连续双压力 |
|---|---:|---:|---:|---:|
| 1 | `41.3022%` | `77.7454%` | `233` | `46.3660 s` |
| 2 | `41.7636%` | `75.8721%` | `237` | `48.8241 s` |
| 3 | `41.6910%` | `76.2877%` | `237` | `47.8922 s` |

三块都显著越过预注册的 `5%` queue fraction、10 个同时样本和 1 秒连续压力门槛。
所以这不是只排 tool queue、LLM 无等待，或反过来的单队列实验。

对六个 E/F fresh server log 的 `physical_kv` scheduler tick 做只读复算，共有 `1,479`
个带 `native_cap=96` 的 capacity-write sample；动态 `effective_cap` 为 `1..67`，中位数
`45`，共有 `50` 个不同取值，恰为 `64` 的只有 `73/1,479`（`4.94%`）。这直接排除
“Joint 仍在使用固定 64 上限”的解释；`67` 是本 workload/时刻观测到的动态最大值，
不是新的通用固定上限。

| Scheduler log | Samples | Min–max | Median | Distinct | `cap=64` | SHA256 |
|---|---:|---:|---:|---:|---:|---|
| [block-01/E](../../artifacts/live_joint/formal/formal-v8-context10k-live-r1/block-01/E/server/vllm_8100.log) | 263 | `1–67` | 46 | 46 | 6 | `f2371ba5adc9c32174d73d8f5cf949a26caa42bfc1f2439e93b311fa176fff60` |
| [block-01/F](../../artifacts/live_joint/formal/formal-v8-context10k-live-r1/block-01/F/server/vllm_8100.log) | 240 | `1–67` | 43 | 43 | 18 | `d08f79b6f588d39dc712d2b6935fe1a7fc6a9c1b805747db75124d73b9f4dace` |
| [block-02/E](../../artifacts/live_joint/formal/formal-v8-context10k-live-r1/block-02/E/server/vllm_8100.log) | 247 | `1–67` | 45 | 46 | 6 | `21540605a71beb9867bdcc1d1d1939bf6713c361365c8cc66277e046d938f4e9` |
| [block-02/F](../../artifacts/live_joint/formal/formal-v8-context10k-live-r1/block-02/F/server/vllm_8100.log) | 241 | `1–67` | 48 | 40 | 21 | `c8fae7508633729462acbbe6a5a4ddec28c7f0c1851bf7510b01fdcce84bcb20` |
| [block-03/E](../../artifacts/live_joint/formal/formal-v8-context10k-live-r1/block-03/E/server/vllm_8100.log) | 245 | `1–67` | 44 | 48 | 5 | `ba70919936926cb1df915883989328ee8166f2cbf4701964017aa5acc7a6fd3a` |
| [block-03/F](../../artifacts/live_joint/formal/formal-v8-context10k-live-r1/block-03/F/server/vllm_8100.log) | 243 | `1–67` | 45 | 46 | 17 | `ee54855e04e57c56ee2a75c547d798173578c9679224c273c92ddb63cbc1ccfc` |

## Live HTTP、重试与失败口径

1,920 个逻辑 tool commit 实际产生 `2,009` 次 HTTP GET：A/B/E/F 分别为
`500/502/504/503` 次。最终失败的 physical job 为零，uncontrolled retry 为零，且所有
authoritative commit 都是 exact、最终 HTTP 200。这里的“零失败”不等于“零瞬时错误”：
共有 `89` 个 commit 发生受控重试，逐条复算都满足同一模式：

- tool/backend/host 均为 `visit / r.jina.ai / r.jina.ai`；
- 第一次响应均为 HTTP `429`；
- 在冻结的单次约 1 秒 backoff 和共享 2.1 秒 visit start gate 后，第二次均为 HTTP `200`；
- 没有第三次 attempt，retry/backoff 全部留在 service time、E2E 与 waste 账本中。

重试按 cell 的块内计数为 A `4/8/8`、B `8/8/6`、E `8/8/8`、F `8/7/8`，每块分母均为
160 authoritative commit；最小块内 rate 已是 `2.5%`，所以失败并非只由一个异常汇总
方式造成。这一 transport 现象正是 formal promotion 的第二个失败 gate，不能被“最终都
成功”掩盖。

## Prefix 与外推边界

本矩阵绑定的 [live_agent.py](../../paste_repro/live_agent.py) SHA256 为
`6dab494fa65749b1d60a5b5cbfbb4d0eed3c804b91b3646e0388c707cb7ade8f`。配置明确记录
`native_prefix_cache_enabled=true` 与 `explicit_joint_prefix_locality_enabled=false`。
由于 A/B/E/F 全部使用相同 native cache，本矩阵 **不是 prefix ablation**：它不能重新
估计 P0→P1 native-cache 收益，也不能把 A→F 的 `29.8252%` 归功于 prefix locality。
它只是在已冻结的相同 prefix 条件下比较 serving/tool 因素。

网络执行是真实的：960 次 Bing HTML search、960 个最终成功的 Jina visit、共享有限
tool worker pool、实际 queue 和 HTTP attempt 都进入账本；LLM/tool 依赖也是真实的。
但 primary call graph 是 frozen trace，visit 目标使用 workload 中预先固定的
`expected_url`，相当于本实验中的 perfect URL prediction。对 12 个 cell 的原始结果复算，
实时 Bing 返回结果实际包含该 `expected_url` 的只有 `404/960`，即 `42.0833%`（约 40%）。

所以可支持的表述是：**真实 Bing/Jina 网络与共享 tool queue 上，frozen/perfect URL
prediction 下的 tool speculative execution–LLM serving 闭环诊断**。不能把它外推成
autonomous agent 会从实时 search results 自主选中同一 URL，也不能假设 autonomous
search selection 会保留 `45.21%` speculative hit rate 或 `3.34%` E→F 收益。按照
[live protocol](LIVE_TOOL_LLM_PROTOCOL.md) 的边界，这也只覆盖 Bing-search/Jina-visit
这两类 tool，不代表论文系统中的所有 tool。

最终判定保持不变：formal-v8 证明了真实 queue promotion 能节省 tool exposed wait，
也证明了 80-way 自然双队列压力与非固定 64 的动态 admission；但它同时显示 Joint 下的
增量 speculation 仅 `3.3362%`，并暴露 Jina 429 重试率超标。因此结果是有价值的失败诊断，
而不是正式 promotion 或最优性结论。

# All-visit causal speculation wall 实验

更新日期：2026-09-02

## 目的

旧实验只为严格的 `search -> one LLM -> immediate visit` 三元组创建预测窗口，因而只覆盖
229 个 visit call 中的 116 个、507 个 visit URL 中的 235 个。本实验保留旧 runner 不变，
新增 generalized causal window：每个有后续 LLM 的 search 或 visit 完成后，立即用当时可见的
search-result cache、visited state 和 tool context 预测下一个 visit。

连续调用按下面的因果顺序运行：

```text
search result -> predict visit A during LLM -> visit A result
              -> update visited state
              -> predict visit B during next LLM -> visit B result -> ...
```

候选构造不读取未来 LLM response 或 authority label。rank pattern、trigger-specific probability
calibration 和 service estimate 都使用 whole-session nested OOF。

本轮把上一版 `0.70x` LLM 时长再乘 `0.60`，最终为原 trace 的 `0.42x`。该缩放不是只在
汇总公式中扣除时长：它被物化到新 trace，`total_time_ms`、`inference_time_ms` 和 `rtt_ms`
都缩放到 0.42，当前 LLM completion 及其后全部 event timestamp 同步前移。runner 对新 trace
使用 runtime scale=`1.0`，因此有效 scale 仍是 `0.42`，不会二次缩放。

## 覆盖审计

- 生成 530 个 measurable tool-completion windows：340 search、188 visit、2 other。
- 140 个 search-trigger window 和 89 个 visit-trigger window 的下一个工具是 visit。
- 229/229 个 recorded visit call 都有且只有一个 causal predecessor window。
- 507 个 recorded visit URL 中 499 个满足 runtime HTTP(S) dispatch boundary。
- 499 个 executable URL 的 corrected service 合计 `2483.460 s`。
- eligible visit wall 为 `2483.460 s`，占 `0.42x LLM` full wall `5947.659 s` 的
  `41.76%`。
- 若 oracle 命中全部 executable visit URL，但仍从当前因果边界启动，最多可省
  `1655.287 s`，即 `27.83%` full E2E。这说明扩大范围后 20% E2E 仍在理论可达区间。

## 模型适配

- visit continuation 在 authority visit commit 后先更新 visited LRU，再生成下一轮候选；未访问 URL
  优先，已访问 URL 保留为真实重复访问的 fallback。
- candidate pool 固定为 Top-20，candidate-pool size 与真正发出的 speculative starts 分离。
- 新增 51 维 causal feature：query/title 与任务文本相关性、URL 与 query/任务相关性、上一个
  visit 的 domain/query group/source rank、任意已访问项的对应历史、continuation depth、候选
  position/rank/ordinal、search age、重复出现次数和 group frequency。
- title、query 和 task text 均从启动点当时已经存在的 trace 内容读取；不会读取未来 LLM response。
  snippet 特征也已接入，但这批修复后 trace 的 search result 没有非空 snippet，所以它在本轮
  不贡献区分能力。
- selector 使用 trigger-specific ridge logistic 与 window 内 pairwise logistic ranker 的几何融合。
  rank pattern、两个模型、service estimate 和预算阈值都按 whole-session nested OOF 训练。
- 不沿用旧的 `query_count>=10 && search_streak==2` hard abstain。扩展标签下，该规则会漏掉
  5 个真实 visit window、18 个 URL 和 14 个 candidate-pool hit；generalized runner 改用
  contextual OOF probability 与 expected value admission。

## 本轮模型设置

LLM 时间缩放不改变候选文本、visit label 或 whole-session fold，因此继续使用上一轮选定的
Top-20 pool、rich/pairwise blend 和 previous-rank history。本轮重新拟合了依赖时长的 OOF
service estimate 与 cross-fold allocation threshold；没有把旧 `0.70x` 的 wall 数字混入新表。

## 跨窗口预算优化

固定 per-window W5 会在低价值窗口也发满 5 个 call。新增 cross-fold allocator：训练折只根据
OOF expected-value score 的分布求阈值，验证折不读取 label；允许有把握的窗口借用空闲预算，
同时显式限制单窗口 burst。下面使用平均 W1--W5、burst cap=`2W`：

| Avg. budget | Burst cap | Selected | Hits | Spec precision | Visit reduction | Full E2E reduction | Call amp. |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 244 | 44 | 18.03% | 6.70% | 2.80% | 1.401x |
| 2 | 4 | 602 | 91 | 15.12% | 14.96% | 6.25% | 2.024x |
| 3 | 6 | 1032 | 138 | 13.37% | 21.38% | 8.93% | 2.792x |
| 4 | 8 | 1551 | 172 | 11.09% | 25.68% | 10.72% | 3.764x |
| 5 | 10 | 2093 | 209 | 9.99% | **31.06%** | **12.97%** | **4.776x** |

W5/burst-10 的净节省为 `771.393 s`。虽然缩短 LLM 后可隐藏的绝对 visit 时间少于上一版，
但无优化 full wall 同时由 `7711.503 s` 降为 `5947.659 s`，因此 C1 的端到端降幅从
`11.12%` 提高到 `12.97%`。

## 各任务并发度：相对无优化 baseline

下面固定 W5/burst-10。baseline 与 treatment 都使用同一份 `0.42x` trace；唯一差异是是否
启用 speculative visit。

| Task C | No-opt wall | Optimized wall | E2E reduction | Authority hit rate | Spec precision |
|---:|---:|---:|---:|---:|---:|
| 1 | 5947.659 s | 5176.266 s | 12.97% | 41.88% | 9.99% |
| 8 | 788.621 s | 685.264 s | 13.11% | 41.88% | 9.99% |
| 16 | 436.482 s | 384.926 s | 11.81% | 41.88% | 9.99% |
| 32 | 283.649 s | 267.675 s | 5.63% | 41.88% | 9.99% |
| 64 | 238.750 s | 229.534 s | 3.86% | 41.88% | 9.99% |
| 128 | 230.837 s | 222.483 s | 3.62% | 41.88% | 9.99% |

hit rate 在不同 task concurrency 下相同是预期行为：候选选择在调度前已经固定，并发度只改变
100 个 session 的 closed-loop makespan 排布，不会重新选择或丢弃 speculation。若要让 hit rate
随并发度变化，需要另一个 bounded shared speculative pool/interference 实验，本报告没有假设它。

权威 multi-URL visit 严格按 `unit_duration_s` 串行求和；同一 decision 选中的 speculative visit
在隔离 slot 中并行启动。wrong-call contention 仍未建模。

## Infinite-TTL session URL cache

新增 session 内持久 speculative-result cache，采用本实验批准的离线假设：

- key 为 `session_id + executable URL`，同一 session 内 URL 可跨 decision 复用；
- TTL 无限、无内容过期、磁盘读取成本为 0；
- 已完成结果立即返回，仍在执行的结果 singleflight claim 后只等待剩余尾部；
- 后续 prediction 选择同一 URL 时不重复物理启动；
- 只有 speculative result 写入此 cache，authority miss 的结果不额外写入；
- candidate selection 仍严格因果。未来 corrected service 只作为 trace replay 的 outcome，
  不参与候选选择或阈值训练。

C1 汇总：

| Policy | Policy selections | Physical starts | Deduplicated | Cache hits | Ready / in-flight | Visit reduction | E2E reduction | Call amp. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| budget W5 / burst-10 | 2093 | 1502 | 591 | 277 | 211 / 66 | 46.83% | **19.55%** | 3.455x |
| fixed Top-10 | 5245 | 2706 | 2539 | 359 | 257 / 102 | 58.14% | **24.28%** | 5.703x |

W5 cache 中有 116 次 authority hit 实际由更早 decision 启动的 speculation 服务，其中 68 次是
原 immediate-next replay 完全没有计入的新增命中。Top-10 对应为 159 次 earlier-decision hit 和
62 次新增命中。由于 completed/running URL 都会跨 decision 去重，持久 cache 同时提高收益并降低
物理 starts；W5 的 call amp. 从无 cache 的 `4.776x` 降到 `3.455x`。

相对同一无优化 baseline 的完整并发度结果：

| Task C | W5 cache optimized wall | W5 E2E reduction | Top-10 cache optimized wall | Top-10 E2E reduction |
|---:|---:|---:|---:|---:|
| 1 | 4784.645 s | 19.55% | 4503.712 s | 24.28% |
| 8 | 636.180 s | 19.33% | 599.937 s | 23.93% |
| 16 | 364.099 s | 16.58% | 347.525 s | 20.38% |
| 32 | 263.429 s | 7.13% | 256.287 s | 9.65% |
| 64 | 229.055 s | 4.06% | 223.797 s | 6.26% |
| 128 | 222.468 s | 3.63% | 217.590 s | 5.74% |

这里 C1 的 event wall reduction 与 mean task-flow reduction 相同。高并发下，固定的 session
集合由少数 critical-path tail 决定，因此 makespan reduction 小于 mean-flow；W5 和 Top-10 的
mean-flow reduction 分别保持 `19.55%` 和 `24.28%`。

## 动态预算与 burst cap 快速探索

`cross_fold_budget` 的 W 不是每个 decision 固定发 W 个，而是训练折上的平均预算目标；验证折
按 OOF expected-value threshold 动态发 0--`burst cap` 个。加入 session URL cache 后的 C1
曲线如下：

| Avg. budget | Burst cap | Policy selections | Physical starts | Effective recall | E2E reduction | Call amp. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 244 | 217 | 13.43% | 4.65% | 1.301x |
| 2 | 4 | 602 | 508 | 25.45% | 9.14% | 1.764x |
| 3 | 6 | 1032 | 819 | 37.27% | 13.25% | 2.269x |
| 4 | 8 | 1551 | 1159 | 45.29% | 15.80% | 2.870x |
| 5 | 10 | 2093 | 1502 | 55.51% | 19.55% | 3.455x |
| 6 | 12 | 2687 | 1842 | 61.12% | 21.56% | 4.080x |
| 7 | 14 | 3335 | 2205 | 67.33% | 23.48% | 4.745x |
| 8 | 16 | 3974 | 2526 | 71.94% | **25.16%** | 5.343x |
| 9 | 18 | 4640 | 2836 | 75.95% | 26.44% | 5.924x |
| 10 | 20 | 5297 | 3126 | 77.56% | 26.79% | 6.489x |

W5/cap-10 有 120/530 个 decision 撞到 hard cap，导致 held-out 实际均值只有 3.95 starts。
保持同一 W5 threshold、只把 cap 放宽到 20 后：

| Policy | Mean starts/decision | P95 | At cap | Physical starts | Recall | E2E reduction | Call amp. |
|---|---:|---:|---:|---:|---:|---:|---:|
| W5 / cap-10 | 3.95 | 10 | 120 | 1502 | 55.51% | 19.55% | 3.455x |
| W5 / cap-20 | 5.00 | 18 | 12 | 1881 | 60.52% | **21.81%** | 4.164x |

这表明固定“小上限”会阻止高价值 decision 借用空闲预算，但完全取消 hard cap 也没有必要：
候选池本身只有 20，保留 cap 可以约束异常 fan-out、公平性和未来共享容量下的最坏情况。
推荐两档：

- 保守档：动态 W5、hard cap=20，C1 `21.81%`，call amp. `4.164x`；
- 均衡档：动态 W8、hard cap=16，C1 `25.16%`，call amp. `5.343x`。它同时优于 fixed
  Top-10 cache 的 `24.28% / 5.703x`。

上面的 isolated-slot 曲线只回答候选预算的收益/调用量 Pareto；它不回答共享资源紧张时的
调度问题。下一节用抢占式 shared-capacity replay 单独回答后者。

## 资源紧张：抢占式共享 Visit 池

新的 shared-capacity replay 使用以下语义：

- authority 和 speculation 共用 2/4/8 个 Visit execution slot；
- 真实 Visit 到达时，命中的在途 speculation 直接提升为 authority，并保留已执行进度；
- 若没有命中，则从最低 OOF exact-probability 的在途 speculation 开始即时取消，直到 authority
  可以调度。取消后只计已经消耗的 resource-seconds，剩余执行不再占槽；
- speculation 因而不会排在 authority 前面；authority 仍可能排在其他 authority 后面；
- 同一 authority multi-URL Visit 仍逐 URL 串行；不同 session 共享资源并发；
- session URL cache 仍为无限 TTL、零读取成本、无过期，且只有 speculative result 写 cache。

比较的 candidate policy 是动态 `W5/cap-10` 与 fixed Top-10；调度器是固定半池、固定保留
一个 authority slot，以及可以使用所有实时空闲 slot 的 `adaptive_idle_fill`。后者并不是无界
发起：每个 decision 仍受 W5/cap-10 或 Top-10 candidate hard cap 约束。

必须区分两个 hit-rate：

- **Policy hit**：不施加共享资源限制时，原 selection + session cache 对 499 个 authority URL
  的覆盖率。它没有因调度实验改变，W5 仍为 `277/499 = 55.51%`，Top-10 仍为
  `359/499 = 71.94%`；
- **Realized hit**：受到共享容量限制后，预测实际获得执行并在 authority 到达时可复用的比例。

最初把 8--32 个并发 Agent 全部压进 2/4/8 个绝对全局 slot，会把资源量级一起改变，不能称为
对原实验的“微调”。纠正后的 sweep 按 active Agent 数缩放 pool，使用每个 active Agent
`1.0x / 1.5x / 2.0x` Visit slots。

关键结果（8 个 deterministic session-order repetitions 的均值）：

| Slots / active Agent | Task C | Candidate / scheduler | Policy hit | Realized hit | Policy hits兑现 | E2E reduction | Call amp. |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1.0x | 8 | W5 reserve-one | 55.51% | 29.93% | 53.93% | 8.52% | 2.499x |
| 1.0x | 8 | Top-10 idle-fill | 71.94% | 30.04% | 41.75% | **8.76%** | 2.594x |
| 1.5x | 8 | W5 idle-fill | 55.51% | **44.16%** | **79.56%** | **13.38%** | 3.241x |
| 1.5x | 8 | Top-10 reserve-one | 71.94% | 43.59% | 60.58% | 13.03% | 3.476x |
| 2.0x | 8 | W5 reserve-one | 55.51% | 51.15% | 92.15% | **16.20%** | 3.462x |
| 2.0x | 8 | Top-10 idle-fill | 71.94% | **53.18%** | 73.92% | 16.08% | 4.328x |
| 2.0x | 16 | W5 idle-fill | 55.51% | 52.58% | **94.72%** | 10.75% | 3.479x |
| 2.0x | 16 | Top-10 reserve-one | 71.94% | **54.51%** | 75.77% | **12.45%** | 4.308x |
| 2.0x | 32 | W5 idle-fill | 55.51% | 52.61% | **94.77%** | 4.62% | 3.493x |
| 2.0x | 32 | Top-10 idle-fill | 71.94% | **54.46%** | 75.70% | **8.54%** | 4.333x |

因此，资源紧张的真实 trade-off 是“能兑现多少原本的 policy hit”，而不是 predictor hit rate
突然下降：

1. `1.0x` pool 只能兑现约 42%--61% 的 policy hits，实际 hit 约 25%--34%，但仍有
   约 4%--9% E2E 收益；
2. `1.5x` 时 W5 已可兑现约 80%--83%，实际 hit 约 44%--46%；W5 的 call amplification
   低于 Top-10，综合效率更好；
3. `2.0x` 时 W5 可兑现约 92%--95%，实际 hit 保持 51%--53%；Top-10 实际 hit 为
   53%--55%，E2E 通常更高，但为了追逐其 71.94% policy ceiling 需要更多资源；
4. `adaptive_idle_fill` 与 `reserve-one` 的差距通常只有几个小数点到约 0.4 个百分点，dynamic
   并非压倒性胜出。更稳妥的实现是保留 per-decision hard cap，并按实时空闲量动态调整全局
   quota；在高 load 下至少保留一个 authority slot。

这里的核心结论是：W5 更适合资源紧张档，2.0x 容量已能维持约 52% realized hit；Top-10
适合资源较宽松、追求更高绝对 E2E 的档位。`Policy hit` 始终保持 55.51%/71.94%，不能与
resource-realized hit 混用。

## 固定单策略的 C1--C128 load curve

为避免在每个负载下比较多种策略或事后挑选赢家，最终 load curve 固定使用同一套部署策略：

- fixed Top-10 candidate hard cap；
- infinite-TTL、zero-read-cost、no-expiration session URL cache；
- adaptive idle-fill shared-pool admission；
- authority 到达时立即提升 exact in-flight job，并抢占最低分的错误 speculation；
- 全局 Visit pool 固定为 64 slots，只改变 task concurrency；
- 两份独立 session replica 组成固定 200-task workload，确保 C=128 实际能够同时激活 128 个
  tasks，而不是被原始 session 数量截断。

Top-10 的无资源限制 policy coverage 固定为 `359/499 = 71.94%`。工具侧指标是 authority
可见 Visit 等待（排队 + remaining service）的总和；overall E2E 是完整 0.42x-LLM session
closed-loop makespan reduction。

`Spec calls / auth call` 是实际启动的全部 speculative tool executions 除以 authority tool
calls；`Unused spec calls / auth call` 是其中结果从未被 authority 消费的 executions 使用相同
分母归一化。两者之差不必等于 realized hit，因为同一个成功缓存结果可以服务多个真实调用。

| C | Realized hit | Spec calls / auth call | Unused spec calls / auth call | Tool stall reduction | Overall E2E |
|---:|---:|---:|---:|---:|---:|
| 1 | 71.94% | 5.423 | 4.741 | 58.25% | 24.32% |
| 2 | 71.94% | 5.423 | 4.741 | 58.25% | 24.25% |
| 4 | 71.94% | 5.423 | 4.741 | 58.25% | 24.25% |
| 8 | 71.94% | 5.423 | 4.742 | 58.25% | 23.83% |
| 16 | 70.92% | 5.412 | 4.739 | 56.88% | 22.00% |
| 32 | 55.02% | 3.821 | 3.299 | 42.21% | 15.22% |
| 48 | 43.46% | 2.716 | 2.301 | 32.35% | 9.94% |
| 64 | 35.24% | 2.144 | 1.807 | 25.13% | 6.44% |
| 96 | 22.65% | 1.531 | 1.313 | 14.66% | 5.19% |
| 128 | 14.77% | 1.183 | 1.041 | 10.19% | 5.41% |

这条曲线说明负载上升时首先下降的是 speculative admission 和 realized hit，而不是静态 predictor
coverage。C≤16 时仍兑现约 98%--100% 的 policy hits；C=32 后固定 64-slot pool 开始明显失去空闲
容量。即使 C=128，抢占式策略仍保留 10.19% 工具 stall reduction 和 5.41% overall E2E，且
错误 speculation 不会排在 authority 前面。

为了单独回答 mostly-wrong 的最坏情况，只在相同的最高负载 `C=128` 下加入一组负对照；在
prediction 完成后确定性地将 75% 或 100% 的真实 URL 替换为不在候选集中的 URL，保持候选分数、
admission 和 speculative load 不变：

| C=128 scenario | Realized hit | Spec calls / auth call | Unused spec calls / auth call | Tool stall reduction | Overall E2E |
|---|---:|---:|---:|---:|---:|
| observed | 14.77% | 1.183 | 1.041 | 10.19% | 5.41% |
| 75% prediction corruption | 3.38% | 1.190 | 1.157 | 2.39% | 0.02% |
| all wrong | 0.00% | 1.201 | 1.201 | 0.00% | 0.00% |

全错时，正在运行的 speculation 会在真实 Visit 到达时立即被抢占，因此不提交错误结果，也不让
错误调用排在 authority 前面；延迟收益平滑退化到零而非负值，资源代价被限制为每次真实工具调用
平均 1.201 个无用 speculative calls。

## 当前剩余瓶颈

Top-20 immediate pool 覆盖 361 个 executable authority URL。持久 cache 后，W5 和 Top-10
分别命中 277 与 359 个 authority URL；C1 距 20% 和 25% 分别只差 0.45 和 0.72 个百分点。
若继续提高收益，主要需要：

1. 提高 candidate recall，尤其补进不在历史 search Top-20 中的 continuation URL；
2. 加入真正非空的 snippet/visit-result semantics，当前 trace 无 snippet，visit result 直接链接也只
   覆盖约 5%；
3. 把训练标签从 immediate-next URL 扩展到 TTL/horizon 内 future-use URL，使有限预算直接优化
   cache reuse，而不只是由持久 cache 被动捡回早期预测；
4. 降低高 task-concurrency 下的 critical-path tail。

## 复现

```bash
python reproduction/scripts/scale_trace_llm_timing.py \
  --input-dir traces/my_traces_tool_slo_search_uniform_1_3s_visit_serial_uniform_2_8s \
  --output-dir traces/my_traces_tool_slo_search_uniform_1_3s_visit_serial_uniform_2_8s_llm_x0_42 \
  --duration-scale 0.42

python reproduction/scripts/run_pattern_v2_trace_all_visit_wall.py \
  --traces traces/my_traces_tool_slo_search_uniform_1_3s_visit_serial_uniform_2_8s_llm_x0_42 \
  --candidate-pool-size 20 \
  --selector-model blend \
  --allocation cross_fold_budget \
  --burst-multiplier 2 \
  --per-task-widths 1 2 3 4 5 \
  --concurrencies 1 8 16 32 64 128 \
  --repetitions 32 \
  --llm-duration-scale 1.0 \
  --domain-prior-strength 10 \
  --coordination-cost-ms 1.0 \
  --output-dir reproduction/results/pattern_v2_trace_all_visit_wall_optimized_pool20_budget_w1_5_burst2_c1_128_r32

# budget W5 / burst-10 + infinite-TTL session URL cache
python reproduction/scripts/run_pattern_v2_trace_all_visit_wall.py \
  --traces traces/my_traces_tool_slo_search_uniform_1_3s_visit_serial_uniform_2_8s_llm_x0_42 \
  --candidate-pool-size 20 --selector-model blend \
  --allocation cross_fold_budget --burst-multiplier 2 \
  --cache-scope session_url --per-task-widths 5 \
  --concurrencies 1 8 16 32 64 128 --repetitions 32 \
  --llm-duration-scale 1.0 --domain-prior-strength 10 \
  --coordination-cost-ms 1.0 \
  --output-dir reproduction/results/pattern_v2_trace_all_visit_wall_cache_session_url_budget_w5_burst2_c1_128_r32

# fixed Top-10 + infinite-TTL session URL cache
python reproduction/scripts/run_pattern_v2_trace_all_visit_wall.py \
  --traces traces/my_traces_tool_slo_search_uniform_1_3s_visit_serial_uniform_2_8s_llm_x0_42 \
  --candidate-pool-size 20 --selector-model blend \
  --allocation per_decision --cache-scope session_url \
  --per-task-widths 10 --concurrencies 1 8 16 32 64 128 \
  --repetitions 32 --llm-duration-scale 1.0 \
  --domain-prior-strength 10 --coordination-cost-ms 1.0 \
  --output-dir reproduction/results/pattern_v2_trace_all_visit_wall_cache_session_url_top10_c1_128_r32

# preemptible shared-capacity resource-tight sweep
python reproduction/scripts/run_pattern_v2_trace_all_visit_shared_capacity.py \
  --traces traces/my_traces_tool_slo_search_uniform_1_3s_visit_serial_uniform_2_8s_llm_x0_42 \
  --capacity-ratios 1 1.5 2 --concurrencies 8 16 32 --repetitions 8 \
  --output-dir reproduction/results/pattern_v2_trace_all_visit_shared_capacity_preemptible

# one fixed Top-10/full-cache policy from C=1 through C=128
python reproduction/scripts/run_pattern_v2_all_visit_load_curve.py \
  --visit-capacity 64 \
  --concurrencies 1 2 4 8 16 32 48 64 96 128 \
  --repetitions 16 --workload-replicas 2 \
  --output-dir reproduction/results/pattern_v2_all_visit_top10_load_curve_c1_128
```

Artifacts：

- `reproduction/scripts/run_pattern_v2_trace_all_visit_wall.py`
- `reproduction/scripts/scale_trace_llm_timing.py`
- `reproduction/tests/test_pattern_v2_trace_all_visit_wall.py`
- `traces/my_traces_tool_slo_search_uniform_1_3s_visit_serial_uniform_2_8s_llm_x0_42/LLM_SCALE_MANIFEST.json`
- `reproduction/results/pattern_v2_trace_all_visit_wall_optimized_pool20_budget_w1_5_burst2_c1_128_r32/REPORT.md`
- `reproduction/results/pattern_v2_trace_all_visit_wall_optimized_pool20_budget_w1_5_burst2_c1_128_r32/metrics.json`
- `reproduction/results/pattern_v2_trace_all_visit_wall_cache_session_url_budget_w5_burst2_c1_128_r32/REPORT.md`
- `reproduction/results/pattern_v2_trace_all_visit_wall_cache_session_url_budget_w5_burst2_c1_128_r32/metrics.json`
- `reproduction/results/pattern_v2_trace_all_visit_wall_cache_session_url_top10_c1_128_r32/REPORT.md`
- `reproduction/results/pattern_v2_trace_all_visit_wall_cache_session_url_top10_c1_128_r32/metrics.json`
- `reproduction/results/pattern_v2_trace_all_visit_wall_cache_session_url_dynamic_budget_w1_10_burst2_c1_128_r32/REPORT.md`
- `reproduction/results/pattern_v2_trace_all_visit_wall_cache_session_url_dynamic_budget_w1_10_burst2_c1_128_r32/metrics.json`
- `reproduction/results/pattern_v2_trace_all_visit_wall_cache_session_url_budget_w5_burst4_c1_128_r32/REPORT.md`
- `reproduction/results/pattern_v2_trace_all_visit_wall_cache_session_url_budget_w5_burst4_c1_128_r32/metrics.json`
- `reproduction/scripts/run_pattern_v2_trace_all_visit_shared_capacity.py`
- `reproduction/tests/test_pattern_v2_trace_all_visit_shared_capacity.py`
- `reproduction/results/pattern_v2_trace_all_visit_shared_capacity_preemptible/REPORT.md`
- `reproduction/results/pattern_v2_trace_all_visit_shared_capacity_preemptible/metrics.json`
- `reproduction/scripts/run_pattern_v2_all_visit_load_curve.py`
- `reproduction/results/pattern_v2_all_visit_top10_load_curve_c1_128/REPORT.md`
- `reproduction/results/pattern_v2_all_visit_top10_load_curve_c1_128/metrics.json`

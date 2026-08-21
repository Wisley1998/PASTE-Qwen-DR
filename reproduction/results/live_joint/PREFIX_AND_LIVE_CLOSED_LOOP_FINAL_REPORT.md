# Prefix 与 live tool–queue–LLM 联合闭环探索及最终结论

## 1. 摘要

本轮工作的目标不是只证明某个离线 scheduler 分数更好，而是分别回答两个问题：

1. 在真实 vLLM serving 中，native prefix cache 是否带来可复现、可归因的收益；
2. 在真实 Bing/Jina HTTP、有限 tool worker、共享 tool queue 与真实 LLM waiting 同时存在时，
   speculative tool execution 和 Joint LLM serving 能否形成端到端正增益。

最终得到三层结论：

| 结论层次 | 对比 | 结果 | 判定 |
|---|---|---:|---|
| Native prefix 因果开发实验 | P0 cache off → P1 cache on | Task mean `54.9726→24.3167 s`，降低 `55.7658%` | 通过全部预注册开发 gate |
| 完整联合闭环 | A FCFS+demand-only → F Joint+visit speculation | `161.8274→115.8396 s`，降低 `28.4178%` | 强正向，80/80 source 更快 |
| Joint 内 speculation 的独立增量 | E Joint+demand-only → F Joint+visit speculation | `120.7134→115.8396 s`，降低 `4.0375%` | 稳定且统计显著，但低于预注册 `5%` promotion 门 |

因此可以确认：**prefix 与真实 live tool queue–LLM serving 闭环都获得了有效增益**。
同时必须保留严格口径：formal-v9 为 `39/40` gates 通过，唯一失败项是 E→F
未达到预注册的 `5%` 效果量，所以不能写成“formal promotion 已通过”，也不能把
`28.42%` 四舍五入宣传为 `30%`。

Prefix 与 closed-loop 是两个分别识别的实验；formal 矩阵中四格使用相同 native prefix
条件，因此两者的百分比不能相加。

主要证据：

- [Native prefix strict validation](../../artifacts/live_joint/prefix_native_causal_dev_v2/native-prefix-v2-r1/strict_validation.json)，
  SHA256 `61ef05004edc806d1b304d8a8bff1f9b85248100ead3bc93d518cc2c5f16b670`；
- [v9 development selection](../../artifacts/live_joint/development/v9_screen/v9-screen-r1/strict_development_selection.json)，
  SHA256 `7f7c9de71f341741192de78ab8596b9cb01721fe211ec3faed79ee33bd7dc7cc`；
- [formal-v9 strict aggregate](../../artifacts/live_joint/formal/formal-v9-context10k-live-r1/strict_four_cell_aggregate.json)，
  SHA256 `9a4e6f11fe580f6dd26f63e6ecd1823a74e5aacac04e0a6c306a881142c64869`；
- [formal-v9 completed matrix](../../artifacts/live_joint/formal/formal-v9-context10k-live-r1/completed_matrix.json)，
  SHA256 `54a1541ab62119a97ff51fc6f213f9b44b49fdd078c1ecfe0b5b420b63a6545f`。

## 2. 探索范围与识别口径

正式四格定义如下。四格都启用相同的 vLLM native prefix cache；Joint 中显式
prefix-locality reorder 固定关闭，因此 formal-v9 只识别 serving/tool 两个因素。

| Cell | LLM serving | Tool policy |
|---|---|---|
| A | Native FCFS | Demand only |
| B | Native FCFS | Visit speculation |
| E | Joint physical-KV scheduling | Demand only |
| F | Joint physical-KV scheduling | Visit speculation |

由此：

- A→B 表示 FCFS 下 speculation 的效果；
- A→E 表示 demand-only 下 Joint serving 的效果；
- E→F 是本报告最关键的 **Joint 内 speculative tool queue 增量**；
- A→F 是用户实际部署完整组合时的端到端效果。

实验使用 frozen call graph：Bing search 和 Jina visit 都是真实 HTTP，tool worker、queue、
rate gate、LLM waiting 和依赖关系也都是真实的；但 visit URL 来自冻结 workload 的
`expected_url`。因此这是 **frozen/perfect-URL prediction 下的真实 execution/queue/serving
闭环**，不是 autonomous agent 从实时 Bing 结果自主选 URL 的实验。

## 3. 探索过程中发现并修正的问题

### 3.1 早期 queued ETA bonus 造成反向调度

早期 Joint speculation 将预测到的 `next_tool_wait` 直接作为正向 overlap bonus。Broker
本身大体保持 FIFO，但绝大多数 prediction 在 LLM request 提交时仍为 `queued`；较大的
queue ETA 反而得到更大奖励，造成 LLM confirmation 近乎倒序，并进一步通过 queued→
authoritative promotion 改写 tool 排位。这主要是固定容量下的延迟位置交换，不是吞吐收益。

修正后的 `execution_aware` 信号使用 exact session+invocation identity，并 fail closed：

- `running/completed` exact prediction 才允许 direct readiness bonus；
- `queued/missing/ambiguous/ineligible` 的 readiness gate 为零；
- 原始 `nw/rtw` 继续用于 tail cost 与 return-KV reserve，不伪造 completed ETA；
- 每次 request 记录 prediction state、gate、job/revision/queue position，允许逐请求重算。

这一步切断了“queue 越深→bonus 越大→更早 confirmation→继续跳队”的反馈环。

### 3.2 Snapshot、reservation 与公平性

为了探索 speculative worker reservation，broker 增加了逐 tool 的一债一还机制：一次
reserved speculative 越过同 tool authoritative 后，下一次可启动机会必须偿还给
authoritative。实现同时补齐了：

- snapshot 中 reservation-aware 的 projected queue/rate position；
- 已有 running speculative 时，snapshot 投影期间不错误恢复新的 speculative turn；
- 每个 dispatch 的 lane、reason、running-spec、auth-before、debt before/after 与 ordinal；
- debt 始终位于 `{0,1}`、逐行可重放、最终归零的验证。

开发实验中的 F1（`min_speculative_tool_workers=1`）每块都真实触发 `9 reserve + 9
repayment`，机制安全且可重放；但相对 F0 仅改善 `0.2079%`，bootstrap CI 跨零，而且一块
completed-ready hit 只有 `5`，低于预注册的 `6`。因此最终选择的是 F0（min=0），没有为追求
更好数字强行保留 reservation。

### 3.3 Canary 的保证浪费

旧实验中，明确标记为 prediction-ineligible 的 canary 仍可能在 admission 后进入 speculative
queue，产生保证浪费并占用受 rate gate 限制的 visit start slot。最终实现将 canary skip
提前到 speculative enqueue 之前。formal-v9 每格有 14 个 canary；所有 B/F canary 都满足
pre-enqueue skip，speculative physical record 为零。

### 3.4 HTTP job gate 不等于真实 attempt gate

Broker 的 job start interval 只限制逻辑 job 首启，不能覆盖 executor 内的 retry GET。
因此新增默认显式开启的 shared per-tool monotonic HTTP-attempt gate：首次 GET 与 retry 都经过
同一 gate，search/visit 分锁，并记录每次真实 attempt 的 start、gate wait 与 backoff。

formal-v8 使用 2.1 秒 visit interval 时出现持续 Jina `429→200`：每 cell retry rate
`4.17%–5.00%`，虽然最终无失败，但违反预注册 transport gate。v9 的 development-only
A baseline 先盲选 transport：

1. 首先运行 2.5 秒；
2. 仅当唯一失败是复合 retry gate 时才允许 fresh 3.0 秒 fallback；
3. 不读取任何 treatment 性能来选择 transport。

2.5 秒首格即满足零 retry 和全部 load gate，因此冻结 2.5 秒，3.0 秒未运行。该选择证据为
[selected_transport.json](../../artifacts/live_joint/development/v9_screen/v9-screen-r1/stage-0/selected_transport.json)，
SHA256 `3c44458963c65deb55b35dfa5a2ff888d5e1ec4cb6c0ff350ebe41e53612dc0d`。

### 3.5 输出格式与 token 公平性

长链路实验多次暴露 guided JSON/xgrammar 的边界问题：裸控制字符、语义相同但 JSON 空白
不同、相同可见 bytes 对应不同 completion-token path，以及 plain final 输出远超本地长度
contract。不能通过事后放宽 parser 或 token gate 来“救活”旧运行。

最终 fixed-final contract 使用：

- exact URL 的 xgrammar；
- `min_tokens=max_tokens=192`；
- compact semantic JSON 后接非空 ASCII-space tail；
- strict JSON parse、exact committed-URL binding、raw/semantic/tail SHA 与 tokenizer telemetry；
- tool calls 仍保持独立 guided JSON contract。

formal-v9 的 `960/960` 个 call-2 都精确生成 192 tokens；E/F aggregate completion-token
相对差仅 `0.0671%`，各块都低于 `1%`。这排除了输出长度漂移对主要效果的解释。

### 3.6 自然双队列负载，而不是固定 64 阈值

早期较短 context 不足以稳定同时压住 LLM 与 tool queue；另一风险是误把 Joint 动态 cap
或 long-context cap 当成自然 vLLM queue。最终负载固定为 context padding 10k、
`max_num_seqs=96`、80 个同时 active task，并要求：

- `80 > 64` 且 `80 < 96`；
- A 必须是原生 FCFS、无 Joint patch；
- LLM waiting 必须发生在 running `<96` 时；
- authoritative tool queue 与 LLM waiting 必须同时出现并连续至少 1 秒；
- 0.2 秒 timeline 的相邻 gap 大于 0.5 秒时必须打断连续区间。

formal-v9 三个 A block 的自然压力如下：

| Block | Native LLM wait（running<96） | Auth tool queue | 双队列样本 | 最长连续双压力 |
|---|---:|---:|---:|---:|
| 1 | `34.33%` | `79.74%` | `204` | `36.50 s` |
| 2 | `39.07%` | `79.60%` | `232` | `46.07 s` |
| 3 | `38.48%` | `79.89%` | `228` | `45.92 s` |

六个 E/F server log 中 physical-KV controller 共记录 1,460 个 capacity-write sample：
`native_cap` 始终为 96，动态 effective cap 为 `1..67`、中位数 43、52 个不同值；恰为 64
只有 `64/1460`，并有 186 次高于 64。这不是固定 64 admission 的伪队列。

## 4. Native prefix 因果实验

### 4.1 为什么需要 v2

prefix v1 使用 constrained JSON 常量输出。不同 fresh process/调度状态会选择不同的合法 JSON
空白 token path；甚至相同可见 response bytes 也可能对应不同 completion token 数，因此旧 v1
无法满足 raw/token identity gate，且 completion 总量差超过 1%。该结果只保留为诊断，未事后
放宽 gate。

v2 将三次生成统一改为 pinned tokenizer 预检的 singleton `guided_choice=["A"]`，
`max_tokens=1`。`A` 是唯一普通 token，raw response、completion token 和语义都固定；后续
prompt 仍追加相同 canonical search/visit fixture，因此保留三阶段 10k 长 prompt 与实际
prefix 关系。

### 4.2 设计与结果

实验使用 P0→P1 / P1→P0 两个 reverse block，每个 condition 聚合 `96` tasks、`288`
requests；16 个独立 source 是统计抽样单位。两个 condition 的 prompt tokens 都为
`3,058,156`，completion tokens 都为 `288`。

| 指标 | P0 cache off | P1 cache on | 变化 |
|---|---:|---:|---:|
| Mean task E2E | `54.9726 s` | `24.3167 s` | `-55.7658%` |
| Mean request | `18.3241 s` | `8.1055 s` | `-55.7658%` |
| Task P95 | `65.4460 s` | `25.6025 s` | ratio `0.3912` |
| Prefill time | `158.5928 s` | `86.5087 s` | `-45.4523%` |
| Native prefix hit ratio | `0` | `64.5293%` | `+64.5293 pp` |

16/16 source 更快；paired-source mean saving 为 `30.6559 s`，10,000 次 bootstrap 的
95% CI 为 `[27.7208, 33.5735] s`。所有预注册 gate 通过，包括 reverse-block 方向、prompt/
completion identity、P1 hit ratio、prefill、E2E 与 tail。

这是一项 development-only 的严格因果实验，证明 native prefix cache 在该 10k 三阶段负载中
具有大幅收益；它没有检验 Joint explicit prefix-locality reorder，也没有与 formal-v9 的
tool/serving 百分比相加。

## 5. Live tool 联合闭环的开发选择

v9 development screen 先用 tune workload 做 2.5 秒 transport A-only 筛选，再以两块反序
`E-F0-F1 / F1-F0-E` 比较：

- F0：visit speculation，`min_speculative_tool_workers=0`；
- F1：与 F0 相同，但启用一债一还 reservation，min=1。

F0 的开发结果为：

- E `128.9177 s` → F0 `115.7898 s`，降低 `10.1832%`；
- 16/16 source 更快；relative bootstrap CI `[8.8003%, 12.0087%]`；
- LLM 组件反而慢 `4.6535%`；tool exposed wait 节省 `15.0423 s`，净节省
  `13.1279 s`，解释比 `1.1458×`；
- 960/960 HTTP attempts 全部首次 200，零 retry/fail/waste。

但两块效果分别为 `2.9557%` 与 `16.4275%`。第二块 E 恰好遭遇多个 Jina 成功但极慢的
response，因此 `10.18%` 开发点估计受到 block-correlated transport shock 放大；source
bootstrap 不能把影响全部 source 的 block shock 当作独立噪声。该开发结果只用于按预注册
规则选择 F0，不能代替 untouched formal workload。

F1 相对 F0 仅改善 `0.2079%`，只有 6/16 source 更快，CI 跨零；虽然 reservation 公平机制
本身完全正确，仍被明确拒绝。最终 formal 候选冻结为 F0、visit interval 2.5 秒、min=0。

## 6. Formal 演进：v8 到 v9

formal-v8 首次把以下关键条件同时固定下来：80-source、80-way offered load、fixed-final 192、
三块 fresh reverse matrix、真实 attempt ledger、自然双队列与 exact canary skip。它得到：

- E→F `3.3362%`，70/80 source，CI `[2.8770%, 3.7995%]`；
- A→F `29.8252%`，80/80 source；
- tool-wait 机制分解通过；
- 但 E→F `<5%`，且 2.1 秒 interval 造成 Jina retry rate `4.17%–5.00%`。

v8 因此是有价值的失败诊断，而不是 formal promotion。v9 只采用 prospective development
screen 选出的 2.5 秒和 F0，不根据 formal-v8 的 performance 事后调 scheduler 权重；同时使用
80 个与所有 dev/tune/formal-v2..v8 无重叠的新 source。

v9 相对 v8：

- 将 E→F 从 `3.3362%` 提高到 `4.0375%`；
- 更快 source 从 70/80 提高到 77/80；
- relative CI 上界从 `3.7995%` 提高到 `4.6453%`；
- 将所有 HTTP retry 从非零降为精确零；
- formal 失败项从两个降为一个。

## 7. Formal-v9 设计与完整性

正式矩阵使用 80 个 untouched source、r00、三块顺序：

1. `A-B-E-F`；
2. `B-A-F-E`；
3. `A-B-F-E`。

每个 cell 都启动 fresh vLLM server、空 cache 和空 broker；效果先按 source 跨三块取均值，
再以 80 个 source 为单位做 10,000 次 paired bootstrap（seed `20260817`）。

完整性证据：

- `960/960` tasks 成功；
- `2,880/2,880` LLM requests exactly once、HTTP 200；
- `1,920/1,920` authoritative tool commits；
- `960/960` final outputs 精确 192 completion tokens；
- 12 个互不相同的 fresh server instance；
- 所有 manifest/evidence SHA 与 run-plan binding 一致；
- 零 LLM retry、preemption、guided recovery、broker leak 或 server error。

真实 transport 共 1,920 次 GET：960 Bing + 960 Jina。逐 attempt 重算得到：

- `1,920/1,920` 都是 actual transport、attempt 1、HTTP 200；
- 0 retry、429、5xx、exception、cancel、physical failure；
- 948 个 cell 内相邻 visit start gap 的最小值为 `2.500080687 s`，无一低于 2.5 秒；
- Jina service 有成功长尾，最大 `34.376 s`，但不是 F 特异；E/F 的物理 HTTP service
  不平衡只相当于每 task `0.0831 s`，约为净 E2E saving 的 1.7%。

## 8. Formal-v9 最终结果

### 8.1 效果量

| 对比 | Mean E2E（baseline→candidate） | 降低 | 95% relative CI | 更快 source | 三块均正 |
|---|---:|---:|---:|---:|---:|
| A→B | `161.8274→139.8429 s` | `13.5852%` | `[10.7620%, 15.9513%]` | `55/80` | 是 |
| A→E | `161.8274→120.7134 s` | `25.4061%` | `[23.4222%, 27.6957%]` | `80/80` | 是 |
| E→F | `120.7134→115.8396 s` | `4.0375%` | `[3.4924%, 4.6453%]` | `77/80` | 是 |
| A→F | `161.8274→115.8396 s` | `28.4178%` | `[26.3101%, 30.7937%]` | `80/80` | 是 |

E→F 三块分别降低：

- block 1：`3.6755%`，70/80 更快；
- block 2：`3.9890%`，75/80 更快；
- block 3：`4.4454%`，73/80 更快。

E→F task P95 为 `204.4502→199.8522 s`，mean makespan 为
`218.5661→212.9858 s`；canary mean/p95 ratio 为 `0.9237/0.9414`，tail 与 canary
都没有回退。

### 8.2 机制分解

| 每 task 均值分量 | E | F | E−F（正数为节省） |
|---|---:|---:|---:|
| LLM | `42.9922 s` | `44.0700 s` | `-1.0778 s` |
| Tool exposed wait | `77.5882 s` | `71.6376 s` | `+5.9506 s` |
| 其中 visit exposed wait | `77.3722 s` | `71.4229 s` | `+5.9493 s` |
| Residual | `0.1330 s` | `0.1320 s` | `+0.0010 s` |
| E2E | `120.7134 s` | `115.8396 s` | `+4.8738 s` |

F 的 LLM 组件反而慢 `2.5069%`，tool exposed-wait saving 是净 saving 的 `1.2209×`。
因此正增益来自真实 tool wait 被隐藏，而不是更短输出、更快 LLM 或 residual 偏差。

### 8.3 Speculation 实际做了什么

F 的 198 个 exact hits 分解为：

- `189` queued promotion；
- `1` running/inflight promotion；
- `8` completed reuse。

只有 `9/198` 在 authoritative confirmation 前真正启动 HTTP service；spec worker time
`10.8446 s`，waste 为零。B/F 合计 396 个预测 visit 则为 `336 queued + 2 running + 58
completed reuse`，60 个以 speculative 身份真实开跑，全部最终 commit。

因此 F 的主机制不是“大量提前执行完网络请求”，而是 **预测 job 提前进入真实共享队列，
在 authoritative 到来时 promotion，降低后续 exposed queue wait**。这正是 live tool queue
与 LLM serving 的联合闭环，但应准确称为 queue pre-positioning / promotion。

### 8.4 Formal 判定

formal-v9 共 40 个预注册 gate，39 个通过。唯一失败项：

```text
E_to_F_mean_reduction:
  observed = 4.037517%
  required >= 5%
```

其他关键门全部通过：三块方向、77/80 source、bootstrap lower>0、P95、makespan、canary、
token、LLM-not-faster、tool-wait decomposition、自然双队列、80>64<96、zero retry、zero
waste、attempt interval 和全部完整性门。

所以最终口径是：

- **有效性**：E→F 是稳定、统计显著、机制闭合的 `4.04%` 正增益；
- **部署组合**：A→F 得到 `28.42%`，80/80 source 更快；
- **严格晋级**：由于 E→F 未达到事先规定的 `5%`，`formal_promotion_passed=false`；
- **不可声称**：不能写“所有 formal gates 通过”“达到 30%”“已经最优”。

## 9. 外推边界

### 9.1 Frozen URL，不是 autonomous search selection

formal-v9 的 Bing 与 Jina 都是真实网络请求，但实时 Bing 结果包含冻结 expected URL 的比例
仅为 `412/960 = 42.9167%`。selected URL 始终来自 frozen workload。因此可以支持：

> 在真实 Bing/Jina、有限共享 tool queue、真实 LLM serving 与 frozen/perfect URL prediction
> 下，visit prediction 的 queue promotion 能稳定降低 exposed tool wait。

不能支持：

> autonomous model 能从实时 Bing results 以相同准确率选择 URL，并保持同样的 hit rate 或
> 4.04% 收益。

要完成后一层外推，需要独立的 autonomous live-search experiment，以 raw Bing rank、
prediction abstention、mismatch cancellation 与真实 attempt ledger 为前提。

### 9.2 Prefix 与 closed-loop 不可直接相加

Prefix 实验比较 cache off/on；formal-v9 四格则都使用相同的 native cache，且显式 prefix
locality 关闭。`55.77%` 与 `28.42%` 来自不同 estimand，不能相加，也不能称为 `84%`
总收益。formal-v9 证明的是在统一 prefix 基线上的 serving/tool 因素。

### 9.3 网络长尾仍然存在

2.5 秒 gate 消除了 429/retry，但没有消除成功响应的 Jina 长尾。formal-v9 中 Jina mean
`0.8963 s`、P95 `2.8824 s`、max `34.3759 s`；这些尾部会增加 block 噪声。三块反序与
source folding 能缓和但不能完全建模所有 block-correlated remote-state shock。因此报告效果
区间时应保留三块结果和 bootstrap，而不只给单一 point estimate。

## 10. 最终工程决策

1. **保留 native prefix cache**。在该三阶段 10k context 负载中，它有大幅且严格可归因的
   收益。
2. **保留 execution-aware exact signal 与 F0 visit speculation**。queued readiness bonus
   必须为零；prediction 仍可提前进入 broker queue，并在 authoritative 到来时 promotion。
3. **保留 2.5 秒 shared per-tool HTTP-attempt gate**。它覆盖首次 GET 与 retry，并在 v9
   产生 1,920/1,920 首次成功。
4. **不启用 min=1 reservation 作为默认候选**。公平机制正确，但开发增量只有 0.21%，
   未通过增量 gate。
5. **保留 canary pre-enqueue skip、fixed-final 192 与完整 attempt/dispatch telemetry**。
6. **对外报告两个不同结论**：完整组合 A→F 为 `28.42%`；Joint 内 speculation 独立增量
   为 `4.04%`，后者未过 5% formal promotion 门。
7. 下一阶段若继续，应优先验证 autonomous live-search selection 和动态、请求等待期间可更新的
   broker→scheduler side channel，而不是继续事后扫 rate、降低 gate 或把 reservation 打开来追数字。

## 11. 证据索引

- [Native prefix protocol](NATIVE_PREFIX_CAUSAL_DEV_PROTOCOL.md)
- [Native prefix strict result](../../artifacts/live_joint/prefix_native_causal_dev_v2/native-prefix-v2-r1/strict_validation.json)
- [Live tool–LLM protocol](LIVE_TOOL_LLM_PROTOCOL.md)
- [v9 development protocol](V9_DEVELOPMENT_SCREEN_PROTOCOL.md)
- [v9 selected transport](../../artifacts/live_joint/development/v9_screen/v9-screen-r1/stage-0/selected_transport.json)
- [v9 strict development selection](../../artifacts/live_joint/development/v9_screen/v9-screen-r1/strict_development_selection.json)
- [v8 diagnostic report](FORMAL_V8_DIAGNOSTIC_REPORT.md)
- [v9 formal protocol](V9_FORMAL_MATRIX_PROTOCOL.md)
- [v9 completed matrix](../../artifacts/live_joint/formal/formal-v9-context10k-live-r1/completed_matrix.json)
- [v9 strict formal aggregate](../../artifacts/live_joint/formal/formal-v9-context10k-live-r1/strict_four_cell_aggregate.json)


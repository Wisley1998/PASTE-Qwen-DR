# 多 Session Speculative Tool Execution 探索总结

更新日期：2026-09-02

本文总结本轮约 10 小时的研究探索：为什么开始优化 speculative tool execution，依次尝试了
哪些调度与执行机制，每种方案解决了什么问题、又暴露了什么新问题，以及目前能支持和不能
支持哪些论文结论。详细设计与逐 artifact 数据分别见：

- [安全设计](./SAFE_SPECULATION_DESIGN.md)
- [sidecar scheduler 综合分析](./results/pattern_v2_sidecar_scheduler_analysis/REPORT.md)
- [最初的 Pattern-v2 逐 request 报告](./results/pattern_v2_per_trace/REPORT.md)

## 0. 一页结论

这轮探索尚未得到一个可以正式声称“正 speculative budget 在多并发下既显著加速、又不拖慢
authority”的 operating point，因此工作不能被描述为已经完成性能验收。不过，问题的根因、
失败方案以及最小可行设计已经基本收敛。

最重要的结论有六条：

1. **主要瓶颈不是 predictor accuracy，也不是 tool worker 数量。** 当前 20 ms 轻量 synthetic
   tool 下，更敏感的是 parent event loop 上的 IPC、pickle、Future callback、result bridge、
   claim/merge、cleanup、timer 同相唤醒等固定串行成本。tool 越轻，这些固定成本占比越高。
2. **“机器资源很多”不等于 authority-safe capacity 很多。** CPU affinity 不是 reservation，
   不同逻辑 CPU 仍可能共享 LLC、内存、内核调度和功耗域；独立本地 worker 也不代表有独立的
   connection、rate token 或远端 backend quota。
3. **不能为了隔离 speculation 而迁移原始 authority 路径。** 把 authority 搬到独立进程虽然
   让 child 内执行更快，却被 IPC/result handoff 税反转，端到端 authority wall 明确变差。
4. **只把 prefetch 移出 event loop 仍不够。** 旧 off-loop 版本在 C16/R32 下同时获得了 logical
   benefit 和显著 authority regression。最新版 silent raw prefetch 删除了 cutoff callback，
   但 current-source C1/R8 仍出现 `+0.192 ms/target` authority regression；正 hit 的 merge/
   notification 和宿主 phase 仍是未完全消除的干扰源。
5. **策略应采用字典序目标。** 第一层只决定本次是否存在可证明的安全预算；第二层才在该预算
   内最大化全局收益。推荐形式是
   `K_eff=min(K_requested,K_cap,K_safe)`，然后跨所有 sessions 做 global net-value Top-K。
6. **字面上的“任何情况下绝不更慢”只有两个可靠答案：完全 K0，或真正独立的 interference
   domain。** 只要 speculative hit 必须在同一个 parent loop 提前通知 agent，就必然存在一次
   treatment-only merge；共享运行时上最多证明预注册 margin 下的 statistical non-inferiority，
   不能证明每次运行、每个 request 都逐点不变慢。

当前推荐的 strict 默认仍是 fail closed：缺 exclusive resource entitlement、缺同
concurrency/placement 的 matched A/A + all-wrong 校准、或任一 runtime certificate 失败时，
`K_eff=0`，且不创建 sidecar。

这里描述的是**目标规范**，不是当前 runner 已经完整实现的 public certificate。当前代码已有
静态资源和 runtime mechanism gate，但尚未读取并 exact-match `K_safe` calibration manifest；
这一缺口在第 7 节单独列出。因此目前任何正 K artifact 都不能称为 strict-safe。

## 1. 探索目的

### 1.1 起点

Pattern-v2 已经能够从 session-local search/history/visited 状态生成下一步 `visit(url)` 候选，
并通过离散 count table 给出 exact-match probability。最初的 100-trace grouped-OOF 结果为：

- Exact Recall@1：约 `21.2%–21.3%`；
- Exact Recall@5：约 `56.8%–57.0%`；
- 实际 overlap-producing runtime hit：`4.3%`；
- 初始逐 request 共享 broker 重放的 overall factor：`0.913x`，即整体慢 `9.59%`；
- 每次 replay 约 35 个 wrong speculative starts，physical-call amplification `1.149x`。

这说明“候选集中存在正确 URL”与“实际系统能获得净加速”之间相差很远。用户提出的核心判断
也是正确的：下一阶段不应继续主要优化 pattern accuracy，而应解决多 session 下的全局资源
协调和 authority interference。

### 1.2 字典序目标

本轮采用两个严格有先后次序的目标：

1. demand-only authority 在受声明的资源域和 workload 范围内不得发生不可接受的回归；
2. 只有第一层通过后，才把有限 speculative 资源分配给期望收益最大的候选。

这与把 latency、hit rate、waste 和 fairness 混成一个加权 reward 不同。第一层不允许用更多
logical benefit 抵消 authority regression；一个方案即使 agent-visible result 更快，只要
authority safety gate 失败，也不能成为默认方案。

### 1.3 成功标准

主要指标包括：

- authority scheduled-to-terminal latency；
- authority observed latency，包括返回 parent/control loop 的 handoff；
- authority wall；
- agent-visible logical latency 与 logical wall；
- selected、started、visible hit、physical-call amplification 和 wrong speculative work；
- all-wrong negative control 与 identical K0/K0 A/A；
- runtime safety invariants，例如 backend arm 顺序、bridge 时机、prefetch deadline、CPU affinity、
  quiet gap、latch 和资源上界。

当前实验使用 paired AB/BA repeat 作为推断单位，而不是把数百个 targets 当成独立样本。实用
no-regression margin 为 `0.10 ms/target` 和 `0.1% authority wall`；`inconclusive` 不等于 pass。
R1 只验证机制，R8 是探索性统计，旧版 R32 才能支持更稳定的正负判断。

## 2. 实验边界与术语

- `C`：同时活跃的 source-session streams 或 lockstep batch width；本轮代表性值为 1、16、64。
- `K`：同一资源域跨所有 sessions 的 speculative start 上限，不是每个 session 一份 K。
- `K0`：完全不运行 speculation 的 demand-only 路径。
- `R`：paired AB/BA repetitions。
- `observed`：使用 trace 中真实 exact hit/miss 标签。
- `all-wrong`：保持候选、时序和负载不变，但强制真实 target 不可能命中，用于测纯干扰。
- `strict shadow authority`：即使 speculative hit，每个真实 target 仍执行一条 authority call，
  便于直接测量 speculation 对原始 demand path 的影响。
- `unsafe-positive-ablation`：为了分析机制而强制打开正 K，但没有完整外部资源证书；不能用于
  no-harm 论文 claim。

主要 sidecar 实验使用固定 20 ms synthetic service、40 ms prediction lead、64 个 authority
workers/tool capacity，并把 predictor scoring/selection 预计算在 timed replay 之外。因此这些
结果测的是 online scheduling/control path，不是完整 predictor、LLM 或真实网络工具的端到端
性能。

### 2.1 源码与证据谱系

结果横跨多个实现版本，不能跨 source hash 合并统计：

| Source family | 识别信息 | 用途 |
|:---|:---|:---|
| 历史 `S0` off-loop | runner `b77b665b…` | 证明 helper 中 pull 仍不足以保护 authority |
| 当前 `S1` silent | runner `6774fbd9…`、sidecar `1bf2ccbd…`、policy `8bbaaeae…` | 当前机制和性能判断的主要依据 |
| 更早的 shared/lazy/eager/process 版本 | 以各 artifact 内 hash 为准 | 解释设计演化和定位瓶颈，不与 S1 合并推断 |

当前工作区源码与最新 S1 C1/R8 artifact 的上述 hash 一致。证据读取优先级为
`metrics.json + source hash` > artifact 自带 `REPORT.md` > 跨 artifact 综合报告 > 目标设计文档。
综合分析报告生成于 `2026-09-02 01:56 UTC`，最新 C1/R8 metrics 生成于 `02:06 UTC`，所以本文
第 4.12 节使用后者，而不是较早综合报告中的 C1 数字。

## 3. 探索过程总览

| 阶段 | 方案 | 主要动机 | 结果与结论 | 当前状态 |
|:---|:---|:---|:---|:---|
| 0 | 直接 Top-k、共享 broker | 验证候选 recall 能否直接转化为收益 | wrong starts 和 cleanup tail 远大于 hit 收益，初始 overall `0.913x` | 淘汰 |
| 1 | authority-first queue + reserve | 给真实调用保留 slot | 能限制 queued speculation，但不能抢占已运行的错误调用 | 保留为基础防线，不是零干扰证明 |
| 2 | global confidence/utility selector | 避免每个 session 各发 Top-k | 显著降低 amplification，高负载能 graceful K0；仍没有稳定正收益 | 保留并继续演化 |
| 3 | batch admission、bounded pending、atomic cancel | 删除 O(N) admission/cleanup 和 sibling 启动竞态 | 控制工作量有界，正确性更稳；不消除运行中 non-preemptive work | 保留 |
| 4 | thread/process sidecar、CPU pin、SCHED_IDLE | 隔离 speculative event loop、GIL 和 run queue | process sidecar 优于同进程共享控制面，但 affinity 不是 exclusive entitlement | 保留 process sidecar；证书边界收紧 |
| 5 | strict shadow-debt barrier | 防止 hit 让下一批 authority 提前压入 broker | 消除 closed-loop arrival 内生性，但牺牲部分连续链路 speedup | strict 实验保留 |
| 6 | dedicated authority thread/process | 进一步保护 authority | 迁移 authority 自身的 IPC/handoff 税大于执行收益 | 仅作诊断，默认淘汰 |
| 7 | latest-safe-start、precompletion、guard | 让 speculation 在 authority 前完成 | 减少时间重叠；无法终止超时的不可抢占远端调用 | strict admission fence 保留 |
| 8 | finite lease、延后 cleanup | 删除 confirmation 附近的 tombstone/scan | all-wrong hot path 不再发 cleanup packet | 保留 |
| 9 | lazy claim | 只在真实 hit 时传结果 | wrong-result traffic 小，但 hit 需要 child round-trip 和 parent wakeup | 仅作消融 |
| 10 | eager parent staging | 把 hit result 提前放入 parent map | hit delivery 明显改善，但所有 wrong result 都唤醒/占用 parent | 默认淘汰，仅作消融 |
| 11 | claim-time pull mailbox | 不运行持续 result bridge | 避免 full eager traffic，但 confirmation 时的 recv/decode 仍在敏感路径 | 被 sealed pull 取代 |
| 12 | guard-sealed pull | authority 前 bounded pull 并 seal epoch | transport/claim invariant 干净，但旧版仍在 cutoff 观察 Future | 被 silent 版本取代 |
| 13 | off-loop prefetch | 删除 parent 上的 pull CPU | C16/R32 仍显著拖慢 authority，说明“off-loop”本身不足 | 淘汰 |
| 14 | silent raw prefetch + independent release clock | 删除 cutoff asyncio callback 和 release 依赖 | runtime safety 全通过；current C1/R8 仍有 authority regression | 当前机制候选，性能尚未验收 |
| 15 | coordination-cost sweep + global Top-K | 只运行净收益足够高的请求 | 干扰随 starts 减少，但可测 benefit 同时消失 | 策略保留，参数需正式校准 |
| 16 | burst latch / structural K0 | 高并发或证书异常时硬退化 | C64 selected=started=0、amp `1.000x` | 已验证的安全 fallback |
| 17 | matched A/A、all-wrong、`K_cap/K_safe` | 把容量和残余干扰分开 | 概念已收敛；代码尚缺强制绑定 calibration manifest | 必须补齐 |

## 4. 各类优化方案的详细结论

### 4.1 从 per-session Top-k 改为全局收益分配

早期 shared-pool sweep 表明，直接发送 Rank-5 在 C1 即有 `3.31x` 左右 call amplification，
drained wall 变差约 `33.2%`；到 C8/C32/C64，虽然 started wrong calls 会被容量限制，authority
仍会受到明显排队与 cleanup 影响。

随后依次加入：

1. 每个 tool 的 authority reserve；
2. confidence threshold；
3. forecast pressure；
4. expected utility/density；
5. 跨所有 sessions 的单一 global start budget。

在最初的 `utility_global_risk_limited` 策略中，C1 call amplification 已从多个倍数降到约
`1.15x`，但整体仍慢 `9.6%`；C8 只启动极少 speculation，C128 完全 K0。结论是：全局预算和
abstention 是正确方向，能够把灾难性资源放大变成 bounded harm/graceful fallback，但共享
non-preemptive pool 仍不能提供 no-regression。

最终保留的 policy 形式为：

```text
K_cap  = 独立 CPU/worker/backend entitlement 给出的容量
K_safe = 同 concurrency、tool class、interference domain 的干扰校准预算
K_eff  = min(K_requested, K_cap, K_safe)

V_i = w_i * p_i * min(service_i, lead_i)
U_i = V_i - coordination_cost_i
```

先过滤 `U_i<=0` 和不可预完成的候选，再按 `U_i` 做 global Top-`K_eff`；service 不同时可用
`U_i/service_i` 排 queued work。每个 `(session_id, decision_id)` 最多一个 exact candidate。

### 4.2 authority-first priority 与 reserve

authority-first 只对**尚未开始**的 queued work 有效。错误 speculation 一旦进入不可抢占 tool，
后到 authority 仍可能等待。authority reserve 可以确保 speculation 不占满所有 broker-tracked
slots，但不能保留共享 start-rate token，也不能终止 detached/remote physical work。

因此该方案的结论是：

- 作为 correctness、bounded occupancy 和普通负载保护机制应保留；
- 不能写成“authority 可以抢占 speculation”；
- 不能单独支持“正 K 无干扰”；
- strict positive K 仍需要独立 capacity 或真正的 hard preemption。

### 4.3 process sidecar、CPU affinity 与 SCHED_IDLE

把 speculation 放入 process sidecar 后，可以隔离它的 event loop、GIL、worker counter 和主要
control plane。固定 arrival、forced all-wrong 的 supplemental experiment 中，C1/R8 authority
点差为 `+0.0085 ms/target`，90% CI `[-0.0819,+0.0988]`，在预注册 margin 下通过；C16/C64
因区间很宽仍 inconclusive。

这一结果说明 sidecar isolation 有价值，但不能外推成物理零干扰：

- `taskset`/affinity 只限制本进程运行位置，不阻止其他 workload 使用该 CPU；
- 不同 core 仍可能共享 LLC、NUMA memory、kernel timers 和功耗域；
- 本地进程隔离不覆盖远端 connection/rate/concurrency quota；
- 当前宿主在实验期间负载较高且没有 exclusive cpuset。

因此 process sidecar 被保留，thread sidecar 与共享 broker 主要作为历史/诊断实现；严格证书必须
另有外部 resource entitlement，并记录 CPU/LLC/interference-domain placement。

### 4.4 不迁移 authority：dedicated thread/process 负控

为了保护 authority，曾尝试把原始 demand-only broker 放入专用 thread 或 process。最明确的
C16/R16 K0 负控为：

- child 内 scheduled-to-terminal 改善 `0.488 ms/target`；
- 回到 parent 后的 observed latency 回归 `0.158 ms/target`；
- authority wall 回归 `1.303%`，90% CI `[1.031%,1.576%]`；
- 将 submit/result 改为同步 batch 后更差，observed `+1.922 ms/target`，wall 回归 `2.493%`。

这证明执行 CPU 不是唯一问题。Pipe send、pickle、child result、parent bridge 和 Future/event-loop
handoff 的固定税可以完全反转 child 侧收益；批量 callback 还会制造 coherent stall。

结论：**原始 in-process authority 拓扑必须保持不变。** Process-authority lane 仅用于把 child
execution 与 parent handoff 分解开，不能作为推荐部署路径或 baseline 替代物。

### 4.5 strict shadow barrier

在 closed-loop agent 流程中，speculative hit 会让逻辑结果提前。如果系统立即让 session 发出
下一批真实调用，treatment 的 authority arrival trace 就比 baseline 更激进，可能让其他 session
变慢。strict shadow-debt barrier 要求本批 shadow authority 全部 drain 后，下一批才进入受保护
broker。

该方案：

- 成功把“本次 hit 加速”和“下一批 authority 提前到达”拆开；
- 保证负载比较更可解释；
- 会牺牲连续 tool chain 的一部分端到端加速；
- 若生产环境希望保留提前 arrival，必须给它另一份 certified surplus capacity，不能借用 baseline
  authority lane。

所以 barrier 在 strict evaluation 中保留，在生产策略中应明确它代表的速度/隔离 trade-off。

### 4.6 latest-start、precompletion 与 temporal guard

严格 admission 使用：

```text
latest_start = planned_confirmation - service_upper_bound - guard
```

sidecar 在实际进入 executor 前再次检查 deadline；结果是否可见使用固定 planned-confirmation
cutoff，不能因为 authority 自己变慢而扩大 hit window。guard 还把 result transport 与 authority
backend 的活跃区间分开。

该方案能减少实际重叠，但不是 hard cancellation：若远端调用超过 service upper bound，又不能
被 sandbox 强制终止，它仍可能跨入 authority interval。因此 precompletion 被保留为 admission
fence，却不能替代独立 backend quota 或 hard timeout。

可选 phase guard 曾尝试错开跨 epoch timer phase，但跨版本实验没有稳定 dose-response，只保留
为诊断消融，不作为安全证书。

### 4.7 lease、cleanup 与 bounded state

早期 authority confirmation 会触发 tombstone、retire、cleanup scan，导致 all-wrong 请求也在
最敏感路径上做 treatment-only 工作。后来改为：

- 有界 registry/pending/result capacity；
- `lease_end=start_deadline+claim_grace`；
- 未命中项在下一次 admission 或最终 drain 时懒回收；
- process miss path 不发送 parent cleanup packet；
- payload、packet、单次 pull 数量都有硬上限。

该优化删除了明显的 all-wrong hot-path cleanup traffic，属于应长期保留的机制。但它只减少
control tax，不解决 running tool/backend contention。

### 4.8 lazy claim

lazy 路径只在真实 exact hit 时通知 child 序列化并返回结果，因此 all-wrong 不搬 result。然而
hit 会在 authority 已提交后触发 child claim、bridge wakeup 和 `call_soon_threadsafe`，正好落在
confirmation 附近。

代表性 C16/R16 all-wrong 结果约为：authority `+0.166 ms/target`、wall `+0.173%`、call amp
`1.054x`，仍不足以证明 no-harm。lazy 的优点是 wrong-result traffic 小，缺点是命中交付路径
长；当前仅作为结果路径消融，不是推荐 strict default。

### 4.9 eager parent-local staging

eager staging 让 child 完成后立即把结果送进 parent 私有 map；exact confirmation 只做本地
try-lock/O(1) lookup，不再发 child claim packet。hit-conditioned authority penalty 从 lazy 的
约 `+0.820` 降到 `+0.290 ms/target`，证明 claim-time round-trip 确实是一个真实瓶颈。

但 eager 对所有 selected results 都做 bridge decode/staging，包括错误预测：

- C16 all-wrong wall 从 lazy 的约 `+0.173%` 增至 eager 的 `+0.288%`；
- observed authority latency `+0.561 ms/target`、wall 约 `+0.650%`；
- overall logical benefit 为负。

结论：parent-local hit 是对的方向，但持续 eager bridge 不是。eager 保留为机制证据，不作为默认
实现。

### 4.10 threshold 提高与 net-value abstention

单纯将 threshold 提高到约 `.26`，曾把 call amplification 降到 `1.011x`，authority latency
点差降到 `+0.084 ms/target`；但 wall 仍回归 `+0.346%`。因此 threshold 只能提高收益密度，
不能把共享控制税变成零。

current silent source 的 C16/R8 coordination-cost sweep 更清楚：

| 设置 | Started | Amp | Observed logical benefit | All-wrong authority 点差 | 结论 |
|:---|---:|---:|---:|---:|:---|
| K1, cost 3.5 ms | 196 | `1.104x` | `+0.250 ms/target` | `+0.261 ms/target` | 有收益点估计，但 authority/wall 失败 |
| K1, cost 4.8 ms | 63–64 | `1.034x` | `-0.024 ms/target` | `+0.140 ms/target` | 干扰降低，收益消失 |
| K1, cost 5.2 ms | 22 | `1.012x` | `-0.044 ms/target` | `+0.020 ms/target` | 点估计接近零但统计 inconclusive |
| K4, cost 5.2 ms | 24 | `1.013x` | `-0.001 ms/target` | `+0.255 ms/target` | K 增大几乎没有增加合格候选，也无总体收益 |

这个 sweep 支持两点：

1. coordination cost 应显式进入 admission，不能只看 probability；
2. 在当前 predictor/service/control-cost 组合下，C16 的安全/收益可行区很窄，更保守会同时消灭
   interference 和可测 benefit。

### 4.11 claim-time pull、sealed pull 与 off-loop pull

为避免 eager 的持续 bridge，child 将 completed result best-effort 写入有界
`SOCK_SEQPACKET` mailbox，parent 只在需要时 pull。

第一版 claim-time pull 在 C1/R8 有小幅 logical improvement：`+0.0849 ms/target`，authority
`-0.0089 ms/target`；但 wall 上界刚超过 gate。C16 all-wrong authority 达
`+0.3897 ms/target`、wall `+0.522%`，明确失败。说明把 recv/decode 放在 confirmation 本身仍
过于敏感。

guard-sealed pull 随后把 bounded pull 移到 authority 前并 seal epoch，exact confirmation 只查
parent-local map。它成功实现：无 claim packet、authority 前无 bridge、late packet 不进入当前
epoch、state/payload 有界。不过旧实现仍在 parent cutoff 观察 prefetch，或让 prefetch 与
authority release 共用 worker。

把 pull CPU 移到 helper 后，历史 `S0` C16/R32 仍得到：

- K0 A/A authority `-0.0364 ms/target`；
- K1 observed authority `+0.2405`，logical benefit `+0.1387`，wall `+0.3063%`；
- K1 all-wrong authority `+0.2510`，wall `+0.3390%`。

observed 虽然显著更快地返回 logical result，但 authority 同时显著回归，所以该版本被淘汰。
它证明“off-loop”不是充分条件，cutoff wakeup、Future 状态观察和 timer phase 仍会影响 parent。

### 4.12 silent raw prefetch

current source `S1` 做了更彻底的时序拆分：

1. baseline 和 treatment 使用同一个只负责 absolute confirmation 的 release clock；
2. prefetch 使用 pin 到 sidecar CPU 的独立 raw `concurrent.futures.Future`；
3. raw Future 不被 asyncio wrap，不注册 parent-loop callback；
4. worker 在 guard 前半段 bounded drain/seal，后半段保持 quiet；
5. confirmation 时先创建所有 authority tasks，并用两个 event-loop turns 让 backend/service await
   全部 arm；
6. 之后才 `Future.done()` non-blocking poll 和 parent-local exact claim；
7. late/error/unsealed、affinity/quiet/arm failure 都抑制当前 claim并关闭后续 latch；
8. zero-target prefetch 也必须被观察；teardown 必须 join-before-close。

针对 race、late Future、zero-target 和 teardown 的测试全部通过。C1 R1 smoke 中 raw prefetch
p95 约 `0.234 ms`，最小 quiet gap `12.43 ms`，没有 late/error/bad claim，并得到约
`+3.98 ms/target` logical benefit；但 R1 不能做性能推断。

最新 C1/R8 current-source 探索性结果为：

| 场景 | Selected / started / visible | Amp | Authority（90% CI） | Logical benefit（90% CI） | Authority wall（90% CI） | 判读 |
|:---|:---:|---:|:---|:---|:---|:---|
| observed | 2512 / 2505 / 375 | `2.332x` | `+0.192 [+0.131,+0.254] ms` | `+3.908 [+3.809,+4.008] ms` | `+0.061% [-0.043%,+0.166%]` | logical improvement，但 authority regression |
| all-wrong | 2512 / 2505 / 0 | `2.332x` | `+0.076 [-0.052,+0.204] ms` | `-0.076 [-0.204,+0.052] ms` | `+0.056% [-0.066%,+0.179%]` | inconclusive |

所有 mechanism/order safety invariants 仍通过，`strict_positive_budget_certificate=false`，因为
这是未获外部资源证书的 ablation。这个结果意味着：silent prefetch 已删除一条明确 hazard，
却仍未证明整个 positive-hit merge path 对 authority 无害。observed 中 hit 与 non-hit shadow
authority 都有同向波动，宿主 phase 与 parent-side claim/race/notification 需要进一步拆分。

在同一 source 上补跑的 C1 K0/K0 A/A 因会话被中断，没有产生 artifact，因此不能用历史 A/A
校正这次 C1/R8 结果。

### 4.13 burst latch 与 C64 structural fallback

当一个 synchronized authority batch 超过配置的 `Bmax=32` 时，runtime latch 将后续预算保持为
K0，不反复开关。current C64/R8 artifact 中：

- selected = started = 0；
- physical-call amplification = `1.000x`；
- sidecar 从未激活。

这严格证明了结构 fallback 行为，但 treatment/baseline 的非零 latency/wall 点差只能是 K0
宿主/运行次序噪声，不能被解释为 speculative gain 或 harm。由于当前 Bmax 仍是手工配置而非
正式 calibration manifest 的 lookup，它验证的是机制，不是完整 public certificate。

### 4.14 A/A calibration 与统计方法

多个 identical K0/K0 实验出现了非零甚至表面“显著”的差值。例如一个 C16/R16 A/A 得到
`-0.155 ms/target` 的假 improvement；current S1 C16/R8 K0 A/A 又得到
`+0.1937 ms/target`、wall `+0.2438%`。这些结果证明亚毫秒 effect 与宿主 phase/noise 同量级。

因此正式顺序必须是：

1. 冻结 runner、policy、broker、sidecar、trace 和 placement；
2. 在同一 interleaved block 先运行 identical A/A；
3. A/A resolution 通过后运行 forced all-wrong；
4. all-wrong 相对 A/A 的一侧 UCB 同时满足 latency/observed/wall margin，才授予该 operating point
   的 `K_safe`；
5. 最后才看 observed logical benefit。

不同 source hash、不同时间窗口或不同 CPU/LLC placement 的 A/A 只能用于诊断，不能从正 K
结果中直接相减后称作正式 DiD。

## 5. 根因模型

综合所有实验，authority slowdown 可分成五层。

### 5.1 不可抢占执行资源

如果 speculation 与 authority 共享一个不可抢占 slot，只要未来 authority burst 可能需要全部
slots，就不存在逐 arrival-trace no-harm。queue priority 只能阻止 queued speculation，不能收回
running work。解决方法只有独立 quota、可证明的 hard preemption 或 K0。

### 5.2 authority topology migration

把 authority 放入新 thread/process 会增加 submit、serialization、return bridge 和 Future handoff。
这属于 baseline path 本身的改变，不应伪装成 speculation isolation。当前结论是保持原始 direct
authority，专用 authority lane 只作负控。

### 5.3 parent 串行 control path

即使 sidecar 的 CPU/tool capacity 完全独立，以下工作仍可能在 parent loop 串行：

- exact-key lookup 与 claim bookkeeping；
- Future state/lock；
- logical race 和 notification；
- result decode/staging；
- callback、cleanup、snapshot 或 bridge 生命周期；
- timer ready-queue phase。

这是“资源充裕但 authority 仍变慢”的核心解释。增加 workers 只增大平行执行容量，不会加宽单
event loop 的串行部分。

### 5.4 result delivery trade-off

- lazy：wrong path 安静，hit path round-trip 重；
- eager：hit path短，所有 wrong results 都打扰 parent；
- pull：避免持续 bridge，但 pull 时机敏感；
- silent sealed pull：目前最干净，但正 hit 最终仍需一次 parent merge。

不存在同时“零 parent work”和“提前把结果交给同一 parent agent”的魔法接口。后续优化只能
继续缩小、有界化或隔离 merge，而不能声称它不存在。

### 5.5 宿主与测量噪声

CPU affinity 不是独占，实验宿主负载高且共享 LLC/内存/内核调度。20 ms asyncio synthetic
service 还会把 event-loop timer stretch 记入 service。A/A 的非零差值说明 raw point estimate
不能直接归因；必须使用同 block calibration 和足够 R。

## 6. 当前收敛的最小设计

### 6.1 执行面

```text
unchanged direct authority
    |
    +-- baseline-identical absolute release clock
    +-- authority tasks always submitted and never cancelled
    +-- backend arm before treatment lookup

isolated process sidecar
    |
    +-- independent CPU/worker/backend entitlement
    +-- SCHED_IDLE + verified affinity
    +-- bounded ingress/registry/result mailbox
    +-- latest-start + precompletion
    +-- silent raw prefetch + half-guard quiet interval
    +-- sealed parent-local result map
    +-- finite lease and post-authority cleanup
```

任一 transport、deadline、affinity、seal 或 arm 问题都只降低 hit coverage，不允许 authority 等待。
strict shadow barrier 防止 early hit 改变下一批 authority arrival。

### 6.2 目标 Policy 面

安全 gate 与收益 ranking 分离：

```text
缺 resource certificate                 -> K_cap = 0
缺 matched interference calibration      -> K_safe = 0
burst > Bmax 或 runtime certificate 失败 -> K_eff = 0，latch 不自动重开
否则                                     -> K_eff=min(K_req,K_cap,K_safe)

eligible_i = exact/idempotent
             AND precompletion feasible
             AND p_i >= threshold
             AND p_i*min(S_i,L_i)-c_i > 0

globally select at most K_eff candidates by net value
```

该方案不需要 online bandit、AIMD 或复杂 per-tenant fairness controller。研究版只需一张离线、
fail-closed 的 operating-point calibration table；key 必须精确包含 source/config、tool class、
concurrency、CPU/LLC placement 和 resource domain。该 manifest lookup 是收敛后的设计要求，
尚不是当前 runner 已完成的能力。

## 7. 实现状态

已经实现并测试的部分：

- `SafeGlobalBenefitPolicy`：全局 net-value selection 和 per-decision cap；
- process sidecar：bounded batch ingress、finite leases、pull mailbox、SCHED_IDLE、affinity certificate；
- unchanged direct authority 与 two-turn backend-arm invariant；
- baseline-identical release clock 与 silent raw prefetch worker；
- deadline/quiet-gap/late/error/zero-target/runtime latch；
- strict shadow barrier；
- join-before-close 与 cancellation-safe teardown；
- structural K0 fallback；
- paired-repeat inference 和 authority/logical/wall telemetry。

相关测试最近一次为 `97 passed, 4 subtests passed`。

独立审计发现一个尚未补齐的关键接口：代码中的 `strict_positive_budget_certificate` 目前仍主要
依赖 `--certified-exclusive-resources`、手填 Bmax/cost 和 runtime mechanism invariants，尚未强制
绑定一份同 source/config/concurrency/LLC 的 `K_safe` calibration manifest；而且 strict certificate
还需显式排除 `unsafe_positive_ablation`。因此：

- 当前所有正 K 结果必须继续标为 ablation；
- 当前 public strict positive certificate 不应被论文引用；
- 下一实现步骤是 exact-match manifest gate，0 个或多个匹配 entry 都退化 K0；
- CLI K/Bmax/cost 不能覆盖 manifest，只能请求不超过已批准 operating point 的预算。

## 8. 当前能支持与不能支持的论文结论

### 8.1 可以支持

- 初始 Pattern-v2 的主要失败不是 top-k recall 为零，而是 wrong starts、cleanup 和共享控制成本
  超过 hit overlap 收益；
- per-session Top-k 会让 speculative demand 随 session 数扩张，global budget/abstention 是必要的；
- authority-first queue 不能抢占 running non-preemptive work；
- process sidecar 能隔离部分 GIL/event-loop/tool-capacity 干扰，但 affinity 不等于 reservation；
- 迁移 authority topology 会被 IPC/return handoff 税反转；
- eager staging 能缩短 hit delivery，却会让 wrong-result traffic 占用 parent；
- off-loop prefetch 仍可能通过 cutoff wakeup/timer phase 拖慢 authority；
- silent raw prefetch 已删除这条明确 hazard，并通过 mechanism/order/race tests；
- coordination-cost gate 能随 starts 降低干扰面，但当前 C16 的可测 benefit 同时消失；
- C64 configured burst guard 能结构性退化到 K0；
- A/A resolution 和 all-wrong gate 必须先于 observed benefit。

### 8.2 目前不能支持

- “资源很多，所以正 speculation 一定不会影响 authority”；
- “轻量工具适合无条件多发 speculation”；
- “当前 silent source 已经实现正 K no-harm”；
- “C16 cost5.2 的 `+0.020 ms` 点估计等于 equivalence pass”；
- “旧 source 的 C1 正结果可以替代 current-source 验证”；
- “R1/R8 或不同时间窗口的 A/A 足以支持 sub-ms formal claim”；
- “不同逻辑 CPU 已经隔离 LLC、内存、功耗或远端 backend quota”；
- “lockstep C1/C16/C64 已代表所有 staggered、异步、长期 multi-session arrivals”；
- “正 K 在每次运行、每个 target 上逐点不比 baseline 慢”。

### 8.3 外部有效性边界

- 当前 tool 是 20 ms synthetic `asyncio.sleep`，固定协调税占比不能直接外推到长网络调用；
- predictor/scoring 在 timed replay 前预计算，结果不包含在线预测开销；
- 当前 host 非独占，affinity 没有隔离其他租户、LLC、内存、功耗域或内核 timer；
- 尚无真实独立 connection、rate-limit token 和远端 backend quota 证书；
- `C` 是 lockstep burst，不代表 randomized phase、staggered async 或长期 serving；
- 当前 OOF 和 R8 均属于 development evidence，不是冻结方案后的新 confirmatory holdout；
- sub-ms effect 与 A/A phase noise 同量级，不能只凭点估计判定 pass。

## 9. 当前状态与下一步

当前 stop condition 尚未达到：没有一个 current-source positive K cell 同时通过完整 strict
resource certificate、matched A/A resolution、all-wrong no-harm 和显著 observed benefit。

接下来的最小路径应是：

1. 修复 certificate 接口，引入冻结的 calibration manifest，并让缺失/错配自动 K0；
2. 不再做大范围盲扫，先在 current source 上完成同 block C1 K0/K0 A/A；
3. 针对 positive hit 的 parent merge/notification 做一次最小消融，避免继续重复优化已经安静的
   wrong-result prefetch；
4. 冻结 source 后，以预注册 R（建议至少 32）依次运行 A/A、all-wrong、observed；
5. C16 只在对应 operating point 的 `K_safe` 通过时打开正 K；否则正式策略报告 K0；
6. C64 保持结构性 fallback，除非有独立的 C64 calibration entry；
7. 若要扩展 paper claim，再补 randomized phase-offset workload；无需先实现复杂线上公平控制器。

## 10. 代表性 artifacts

### 初始 predictor 与共享 broker

- `results/pattern_v2_per_trace/`
- `results/pattern_v2_load_robustness/`
- `results/pattern_v2_adaptive_load/`
- `results/pattern_v2_open_loop_stress/`

### authority topology 负控

- `results/authority_process_percall_nobarrier_individual_vs_direct_k0_lead40_c16_w64c64_cpu10_11_12_r16/`
- `results/authority_process_batched_nobarrier_vs_direct_k0_lead40_c16_w64c64_cpu10_11_12_r16/`

### lazy/eager/pull 路径

- `results/pattern_v2_sidecar_direct_postauthority_retirement_guard10_allwrong_c16_w64c64_cpu57_60_r16/`
- `results/pattern_v2_sidecar_eager_staging_guard10_allwrong_c16_w64c64_cpu57_60_r16/`
- `results/pattern_v2_sidecar_eager_staging_guard10_observed_c16_w64c64_cpu57_60_r16/`
- `results/pattern_v2_direct_pull_netvalue_c5p2_guard10_c1_c16_c64_w64_cpu57_60_r8/`

### 历史 sealed/off-loop 结果

- `results/pattern_v2_direct_sealed_pull_k4_c1_c16_c64_w64_cpu57_60_r8/`
- `results/pattern_v2_offloop_guard15_k0_aa_c16_r32/`
- `results/pattern_v2_direct_offloop_pull_k1_guard15_c16_r32/`

### current silent source

- `results/pattern_v2_silent_k4_guard15_c1_cpu8_12_r8/`
- `results/pattern_v2_silent_net35_k0_aa_c16_cpu8_12_r8/`
- `results/pattern_v2_silent_net35_k1_guard15_c16_cpu8_12_r8/`
- `results/pattern_v2_silent_net48_k1_guard15_c16_cpu8_12_r8/`
- `results/pattern_v2_silent_net52_k1_guard15_c16_cpu8_12_r8/`
- `results/pattern_v2_silent_net52_k4_guard15_c16_cpu8_12_r8/`
- `results/pattern_v2_interference_envelope_c64_k0_fallback_cpu8_12_r8/`

这些 artifact 跨越多个 source revision。比较时必须先检查 `source_sha256` 和 canonical payload
hash；历史结果用于解释设计演化，不能与 current source 直接拼接成正式因果估计。

# 多 Session speculative scheduler：current-source 综合分析

## 结论先行

当前最该优化的不是 predictor accuracy，也不是继续增加 worker，而是避免在 authority 的串行
控制路径上制造固定税。对这里约 20 ms 的轻量 tool，worker/backend slot 并未表现为首要瓶颈；
更敏感的是 IPC、序列化、result bridge、Future callback、timer 同相唤醒，以及 confirmation
附近的 claim/cleanup。多 session 把这些几十至几百微秒的成本聚成 burst，最终可能让没有命中
speculation 的真实 authority call 也变慢。

因此 current-source 对论文最合适的设计结论是：

1. 保留原始 in-process demand-only authority，不为 speculation 迁移 authority 拓扑；
2. strict positive `K` 使用原始 direct authority、baseline-identical release clock、silent raw
   prefetch、两轮 authority-first 与 precompletion；同时必须有外部 exclusive CPU/backend
   entitlement 和 matched all-wrong/A/A interference calibration，任一缺失就结构性退化为 `K=0`；
3. process-authority 只保留为瓶颈分解的诊断臂，不作为推荐部署路径；
4. `K_eff=min(K_requested,K_cap,K_safe)` 是跨所有 sessions 的全局预算，按扣除协调税后的 net
   value 排序；对 service 很短、预期收益不足以覆盖协调税的 tool，正确动作就是不投机；
5. 当前宿主没有 exclusive entitlement，故本报告中的 lazy/eager/pull 正预算结果都只是 ablation，
   不能支持“正投机在任何情况下 no-worse”的 claim。

这个结论并不否定 speculation：eager staging 明显改善了命中结果的交付路径，但全量、持续的
parent staging 会被低 precision 放大。current-source 因而改为 silent kernel mailbox，由 pin 到
sidecar CPU 的 raw-Future worker 在 authority 前的 guard window 做有界 prefetch 并 seal epoch，
完成时不唤醒 parent loop。exact confirmation 先 arm authority，再只查 parent-local map，不读
socket、不做 pickle，也不唤醒 child。

## 1. 根因：资源总量充裕，不等于 authority 路径无征用

### 1.1 执行核不是当前主要瓶颈，跨进程 handoff 才是

最直接的负控是在 `C=16, R=16, K=0` 下比较原始 direct authority 与 process authority。
没有任何 speculative call，因此差值完全来自 authority 拓扑迁移：

- child 内部 scheduled-to-terminal latency **减少 0.488 ms/target**；
- 回到 parent/control loop 的 observed latency 却 **增加 0.158 ms/target**；
- authority wall **回归 1.303%**，90% CI 为 **[1.031%, 1.576%]**。

也就是说，独立 CPU/GIL 确实让 child 执行更快，但同步 Pipe submit、pickle、child send、parent
bridge 解码和 Future/event-loop handoff 的固定成本更大。把 authority 移到独立进程并没有
保护 authority，反而改变并拉长了原本最短的 demand-only 路径。

进一步把 authority submit/result 做成批量同步 burst 后，authority wall 回归扩大到
**2.493%**（90% CI `[2.118%, 2.871%]`），observed latency 增加 `1.922 ms/target`。批量减少
packet 数并不等于减少关键路径成本：同一批 service timers 和 result callbacks 会在 parent
event loop 上形成 coherent stall。因而 current-source 恢复 per-call、individual result 的
诊断实现，但推荐策略是完全保留原始 direct authority。

对应 artifacts：

- `../authority_process_percall_nobarrier_individual_vs_direct_k0_lead40_c16_w64c64_cpu10_11_12_r16/`
- `../authority_process_batched_nobarrier_vs_direct_k0_lead40_c16_w64c64_cpu10_11_12_r16/`

### 1.2 轻量 tool 会放大固定协调税

这里 synthetic service 是 20 ms。一次本地 claim、一次 bridge wakeup 或一次 timer stall 的绝对
时间虽然很小，但它们会作用于每个 decision epoch，而且无法靠增加 tool workers 消除。tool
越轻，`coordination_cost / service_time` 越大；多 session 的同步 arrival 又会把 callback 排队
成本集中到同一个 event-loop turn。因此“这些工具很轻量”恰恰是更应该保守 admission 的原因，
而不是可以忽略控制面的理由。

### 1.3 96 个 CPU 不是 reservation certificate

本轮诊断期间机器有 96 个逻辑 CPU，但 load average 约 73，CPU PSI `some avg10` 约 13%，且
一秒采样中 pin 给 sidecar 的 CPU11 曾约 87% busy。`taskset`/affinity 只限制本进程可以在哪些
CPU 上运行，不能阻止同宿主其他 workload 使用这些 CPU，也不隔离 LLC、内存带宽、kernel
timer、功耗域或 backend quota。

因此“不相交 CPU mask + SCHED_IDLE”是有用的运行时检查，但不是 exclusive entitlement。
strict positive `K` 仍需来自外部环境的 exclusive cpuset/isolated host，以及 sidecar 独立的
connection pool、rate token 和 backend concurrency quota。CLI 的
`--certified-exclusive-resources` 是调用方 attestation，不是 runner 能自行证明的物理事实。

### 1.4 当前 C16 回归主要是 timer/order stretch，不是 tool slot 饱和

旧 pull formal 的 C16 all-wrong cell 表面上有 `+0.3897 ms/target` authority delta。逐段分解为：

- authority 第一个 event-loop turn：`+0.1032 ms`（约 26.5%）；
- broker queue：`+0.0086 ms`（约 2.2%）；
- broker dispatch 到 executor completion：`+0.2821 ms`（约 72.4%）；
- broker 其余部分与 return 后残差合计约 `-0.0042 ms`。

这里的第三项包含 synthetic executor 的 `asyncio.sleep(20ms)` timer stretch，不是 20 ms backend
本体变慢。该 cell 的 authority 最大并发仅 `23.625 < 64`、call amplification 仅 `1.0117x`，而且
authority interval 内 pull/claim/result packet 均为 0、parent bridge 未启动、sidecar 与 parent
分别 pin 在 CPU60/57。因此数据不支持“投机占满 tool worker”或“wrong-result decode 拖慢
authority”；更像是 authority 首次调度顺序、coherent timer wakeup 与共享宿主 phase 的组合。

这个判断还得到 K0 A/A 的支持：同样 C16 下 AB repeats 平均约 `+0.393 ms`，BA repeats 约
`-0.002 ms`，与 treatment 同量级。由于这组 A/A 与旧 pull 的 source hash 不同，只能用于根因
诊断，不能作为正式因果校正。current-source 因而同时修正执行顺序，并要求冻结源码后的 matched
A/A；不能用一组 raw A/B 点差直接归罪于 speculation。

## 2. 推荐的最小 policy

策略采用字典序目标：先保护 demand-only authority，再在已认证的 surplus 中最大化收益。无需
引入复杂 bandit 或 per-tenant optimizer。

```text
if strict_resource_certificate is invalid:
    effective_K = 0
else:
    effective_K = min(requested_K, certified_K)

for candidate i from every active session:
    require exact context/arguments and safe, idempotent tool
    require p_i >= threshold
    require predicted_start + S_upper_i + guard <= planned_confirmation_i

    gross_i = w_i * p_i * min(S_i, L_i)
    net_i   = gross_i - coordination_tax_i
    keep only net_i > 0

globally select at most effective_K candidates by net_i
order queued candidates by net_i / S_i with a stable identity tie-break
```

`coordination_tax_i` 不需要复杂建模；研究版用对应 topology/concurrency 的 all-wrong
microbenchmark 给出 admission、staging、claim 的保守上界即可。matched A/A 用来判断当前测量
窗口能否分辨这个 effect，不应把偶然 host phase 全部硬算成每个 candidate 的因果成本。这样
predictor 只回答“可能命中多少”，resource policy 负责回答“这次命中的价值是否足以覆盖协调
税”。对轻量 tool，若 net value 不为正，哪怕 CPU 看起来空闲也取 `K=0`。

`K` 必须是同一资源域内所有 sessions 共享的全局预算，而不是每个 session 各有一个 `K`。
每个 `(session_id, decision_id)` 最多给一个 exact candidate；pending queue 有界，高价值新候选
只能替换尚未运行的低价值候选，不能抢占已经运行的 tool。若一个 authority burst 超过经过
校准的 `Bmax`，latch 在本 replay 内关闭且不自动重开，避免高并发附近反复启停。

## 3. authority-safe 执行路径

### 3.1 原始 authority 永远先提交，也永远不取消

每个真实 target 都先无条件提交原始 demand-only authority call。speculation 只参加
`first(valid speculative result, authority terminal result)`，identity 不匹配、未 ready、过期、
失败或 transport busy 时立即 miss，并回退到已经存在的 authority future；不会再创建第三次
调用，也不会取消 shadow authority。

strict shadow-debt barrier 阻止一个 batch 因 speculative early return 而提前把下一 batch 的
真实 authority calls 压进 baseline broker。它牺牲一部分连续链路 speedup，但避免一个 session
的命中把其他 sessions 的 demand-only 权益变成隐性资源。

多 session 下，“创建了 authority task”还不等于 backend 已经启动。current-source 在每个
confirmation batch 中固定使用两个 authority-first event-loop turn：第一轮让 authority task
进入 broker 并创建 runner，第二轮让 broker runner 进入 executor、arm 第一个 service await；
然后才允许 treatment 做 exact lookup/claim。executor 入口计数器记录每个 authority backend
是否已经 arm，任何 violation 都使 safety certificate 失败。这个两轮顺序直接针对旧实现中
`handle_by_target.pop()` 聚集在第一个 yield 之前的 control burst，又不需要引入新的 scheduler。

### 3.2 precompletion 是 admission fence，不是均值启发式

strict 模式使用：

```text
latest_start = planned_confirmation - calibrated_service_upper_bound - guard
```

child 在真正进入 executor 的下一 event-loop turn 再检查一次 deadline；迟到任务不调用
executor。结果可见性使用固定的 planned-confirmation cutoff，而不是可能已被拖晚的实际
confirmation。这样不会因为 authority 本身变慢而反向扩大 speculative hit window。

silent sealed pull 把最后一个允许 result transport 的时点固定为
`planned_confirmation - guard`。独立、pin 到 speculative CPU 的 worker 通过 raw
`concurrent.futures.Future` 做 bounded prefetch，将已完成包移到 parent-local map 并 seal
epoch；它不包装成 asyncio Future，完成时不会唤醒 parent loop，也不和 authority release clock
共用 worker。guard 的后半段是显式 quiet gap。planned confirmation 到达后，parent 先按上面的
两轮顺序提交并 arm authority，随后才 non-blocking poll raw Future。未完成、deadline miss、
decode error 或 seal 失败都抑制当前 claim、关闭后续正预算 latch，而不会让 authority 等待。

这仍不能终止一个超出 service upper bound 的不可抢占远端调用，所以 strict claim 还必须依赖
独立 backend entitlement，或一个可以强制停止的 hard sandbox deadline。

### 3.3 延后 lease 回收，把 cleanup 移出 authority confirmation

每个 speculative handle 使用有限 lease：

```text
lease_end  = latest_start + claim_grace
claim_grace = service_upper_bound + guard + 10 ms
```

未 claim 工作到期后由 child 静默 tombstone，不向 parent authority path 发送 terminal/cleanup
packet。parent 不在 confirmation 时扫描或 retire，而是在**下一次 submit/admission** 前懒回收
过期 registry；若没有下一次 submit，则由最终 lifecycle drain/close 清理。exact claim 只做
O(1) key lookup 和 lease check。这个“延后回收”机制删除了 all-wrong authority hot path 上的
retire packet、bridge wakeup 和 cleanup scan，同时仍由有限 queue、lease 与最终 drain 保证
状态有界。

对应 lazy all-wrong current-source artifact：
`../pattern_v2_sidecar_direct_postauthority_retirement_guard10_allwrong_c16_w64c64_cpu57_60_r16/`。

### 3.4 sealed pull 是推荐 strict direct 路径，lazy/eager 只作消融

lazy claim 在真实 authority 已提交后才通知 child 序列化结果，并经 bridge 唤醒 parent loop。
这正好把跨进程 round-trip 放在最敏感的 confirmation 附近。eager staging 则把完成值预先传入
parent 私有有界 map；真实 arguments 到达后只做本地 try-lock/O(1) exact claim，未 ready 立即
miss，不发 child claim packet。

它的 trade-off 很清楚：eager 改善 hit delivery，但会在 parent 解码所有 selected 结果，包括
错误预测。sealed pull 仍由 child best-effort 写入 bounded `SOCK_SEQPACKET` mailbox，却没有
timed parent reader；只有独立 raw-Future worker 在 guard window 内最多读取 `max_pending` 个
datagram，并在返回前 seal epoch。worker 完成不注册 asyncio callback；authority 使用另一条
baseline-identical release clock，并在 backend 已 arm 后才观察 Future。之后 exact `try_claim`
只做 parent-local non-blocking registry/staging try-lock 和 O(1) key lookup；锁忙、not-ready、
identity 不匹配或过期都立即 miss。迟到包留在 socket，不能重新进入这个 epoch 的 authority
claim。

这里的“non-blocking”特指没有 transport wait，且 registry acquisition 用 try-lock。Future、
handle 与统计计数内部仍有本地锁；strict runner 通过“无 bridge、无并发 submit/snapshot/close”
使它们在封闭协议内无竞争，但这不是任意外部线程下的 lock-free/wait-free API 保证。

inner result 限为 4 KiB，outer packet 也按 result cap 加固定 envelope headroom 截断；超限或
malformed packet 在 outer pickle 前丢弃。setup-only certificate handshake 不创建 bridge，普通
snapshot/close 只在 authority drain 后启动 lifecycle bridge。因此推荐 strict direct 路径是
sealed pull；lazy/eager 只用于分解 round-trip 与 full-staging 税。

prefetch 也不是 hard real-time primitive：deadline 在每个 packet 之前检查，随后 bounded recv/
decode 仍可能越界；runner 会把 deadline miss 记为证书失败并 fail closed。current-source 的
4 KiB、`max_pending=8` C1 R1 smoke 中，prefetch p95 为 `0.234 ms`，最小 worker-body quiet gap
为 `12.43 ms`，但正式 no-regression 仍以 matched repeated experiment 判定，不能由该单次
timing 推出。

## 4. 当前实验结果

所有推断以 paired AB/BA repeat 为独立单元，而不是把单个 target 当独立样本。no-regression
门限是 `0.10 ms/target` 与 `0.1% authority wall`；`inconclusive` 不等于 pass。下表的正预算
实验均使用 `--unsafe-positive-ablation` 或等价的未认证配置，因为当前 host 没有 external
exclusive certificate。

| 实验 | C/R | Authority latency 差值 | Authority wall 差值 | 额外诊断 | 判读 |
|:---|:---:|:---|:---|:---|:---|
| process K0 vs direct K0，per-call/individual | 16/16 | child `-0.488`，observed `+0.158` ms/target | `+1.303%`，CI `[+1.031%,+1.576%]` | call amp `1.000x` | topology migration 明确回归 |
| process K0 vs direct K0，batched submit/result | 16/16 | child `+0.523`，observed `+1.922` ms/target | `+2.493%` | call amp `1.000x` | coherent callback burst 更差 |
| direct lazy，t=.22，all-wrong，CPU57/60 | 16/16 | `+0.166 ms/target` | `+0.173%` | call amp `1.054x` | inconclusive，不是安全证据 |
| direct eager，t=.22，all-wrong，CPU57/60 | 16/16 | `+0.159 ms/target` | `+0.288%` | call amp `1.054x` | wall regression |
| direct eager，t=.22，observed，CPU57/60 | 16/16 | `+0.561 ms/target` | 约 `+0.650%` | logical benefit `-0.287 ms/target`，call amp `1.053x` | authority regression；全局收益被抵消 |
| direct lazy，t=.26，all-wrong，CPU57/60 | 16/16 | `+0.084 ms/target` | `+0.346%` | call amp `1.011x` | 降低工作量仍未通过 wall gate |
| identical direct K0 A/A，CPU57/60 | 16/16 | `-0.155 ms/target` | 约 `-0.132%` | 两臂物理行为相同 | 竟被判为 “improvement”，暴露 sub-ms noise |

相关 artifacts：

- lazy t=.22 all-wrong：
  `../pattern_v2_sidecar_direct_postauthority_retirement_guard10_allwrong_c16_w64c64_cpu57_60_r16/`
- lazy t=.22 observed/hit decomposition：
  `../pattern_v2_sidecar_direct_lateststart_guard10_bmax32_c16_w64c64_cpu10_11_r16/`
- eager t=.22 all-wrong：
  `../pattern_v2_sidecar_eager_staging_guard10_allwrong_c16_w64c64_cpu57_60_r16/`
- eager t=.22 observed：
  `../pattern_v2_sidecar_eager_staging_guard10_observed_c16_w64c64_cpu57_60_r16/`
- lazy t=.26 all-wrong：
  `../pattern_v2_sidecar_lazy_guard10_t026_allwrong_c16_w64c64_cpu57_60_r16/`
- identical K0 A/A：
  `../pattern_v2_direct_identical_k0_aa_guard10_c16_w64c64_cpu57_60_r16/`

### 4.1 eager 确实改善命中交付，但没有改善全局结果

对 treatment 中最终由 sidecar 可见返回的 target，按 target id 与 paired baseline authority
对齐后，pooled hit-conditioned authority delta 从 lazy 的约 `+0.820 ms/target` 降到 eager 的
约 `+0.290 ms/target`。这支持“confirmation 附近的 child claim/result round-trip 是一个真实
串行点”，也说明 parent-local staging 的方向正确。

但该分解是诊断量，不是预注册的全局 success metric，而且两次 run 位于不同测量窗口。全量
指标仍然失败：eager all-wrong authority wall 从 lazy 的约 `+0.173%` 增至 `+0.288%`；observed
authority latency 为 `+0.561 ms/target`。当前约 24% started precision 下，大多数 staged value
不会命中，wrong-result traffic 与宿主噪声足以吃掉命中交付改善。因此正确修复不是无条件开启
eager，而是 `eager + net-value abstention + exclusive entitlement`。

### 4.2 提高 threshold 只减少调用，不会自动形成 no-regression

把 threshold 提高到 `.26` 后，physical call amplification 已降到 `1.011x`，authority latency
点差也只有 `+0.084 ms/target`；然而 authority wall 仍回归 `+0.346%`。这说明 threshold 是收益
密度控制，不是资源隔离。固定 staging/setup/timer 税与 host phase noise 不会随预测数量同比
消失，轻量 tool 仍可能应直接 `K=0`。

### 4.3 A/A 噪声校准必须前置

两臂都为 direct K0、没有 speculative execution 的 identical A/A，竟得到
`-0.155 ms/target` 的“显著 improvement”（90% CI `[-0.309,-0.001]`）。这不是 K0 的算法收益，
只能来自运行次序、宿主 phase 或 sub-ms 测量噪声。它证明仅有 paired AB/BA 和 nominal CI 仍
不足以支撑这里的亚毫秒 claim。

因此每个 concurrency/topology 的正式 A/B 前必须先做 matched A/A，并把其经验偏差/波动并入
equivalence margin 或 coordination tax。若 A/A 自身没有通过预注册的等价性校准，对应 A/B
无论点估计多好都只能标为 inconclusive/ablation。

## 5. 三臂实验设计

后续实验应固定为三个可解释的臂：

- **A — Direct K0**：原始 in-process demand-only authority；这是论文和部署基线。
- **B — Direct K>0**：authority 与 A 完全相同，只增加 pull process sidecar；A/B 才回答
  “speculation 是否在不拖慢 authority 的情况下带来净收益”。
- **C — Process-authority diagnostic**：把 authority 移入独立进程，用 K0 与 K>0 做 IPC/GIL/
  handoff 分解。C 不能替代 A，也不能把 C 内部的增量改善写成相对原始 no-spec 的收益。

实验顺序也应固定：先 identical A/A 校准噪声，再做 A/C 的 K0 topology negative control，随后
做 A/B all-wrong safety gate，最后才做 A/B observed benefit。只有 all-wrong 同时通过 authority
latency 与 wall no-regression，observed 才有资格讨论 speculative gain。

current direct-eager、threshold `.26` 的 `C=1/16/64, R=8` 最终矩阵已经完成：

`../pattern_v2_direct_eager_netvalue_t026_guard10_c1_c16_c64_w64_cpu57_60_r8/`

| Scenario | C | Selected / visible hit | Authority delta ms/target (90% CI) | Logical benefit ms/target (90% CI) | Authority wall delta (90% CI) | 判读 |
|:---|---:|:---:|:---|:---|:---|:---|
| observed | 1 | 24 / 8 | `+0.0046` `[-0.1042,+0.1134]` | `+0.0821` `[-0.0267,+0.1909]` | `+0.0011%` `[-0.0610%,+0.0633%]` | wall pass；其余 inconclusive |
| observed | 16 | 22 / 4 | `+0.0791` `[-0.1441,+0.3022]` | `-0.0340` `[-0.2721,+0.2042]` | `+0.1823%` `[-0.1180%,+0.4835%]` | inconclusive |
| observed | 64 | 0 / 0 | `-0.6765` `[-1.3456,-0.0075]` | `+0.6765` `[+0.0075,+1.3456]` | `-0.4013%` `[-1.1440%,+0.3471%]` | burst guard K0；差值是噪声 |
| all-wrong | 1 | 24 / 0 | `+0.1226` `[+0.0845,+0.1607]` | `-0.1226` `[-0.1607,-0.0845]` | `+0.0838%` `[-0.0004%,+0.1680%]` | logical regression |
| all-wrong | 16 | 22 / 0 | `+0.2130` `[+0.0196,+0.4065]` | `-0.2130` `[-0.4065,-0.0196]` | `+0.3370%` `[+0.0896%,+0.5850%]` | logical regression；no-worse 未通过 |
| all-wrong | 64 | 0 / 0 | `-0.3217` `[-0.8480,+0.2046]` | `+0.3217` `[-0.2046,+0.8480]` | `-0.3281%` `[-0.8740%,+0.2208%]` | burst guard K0；inconclusive |

六个 cell 的 child claim packet 都是 `0`，所以 eager 确实删除了 claim round-trip；C1 observed
authority 点差接近零也支持这一机制。但 all-wrong C1/C16 仍有显著的 logical regression，说明
剩余瓶颈是所有 completed guesses 都触发的 eager result packet、bridge decode/staging 与 parent
GIL/lock 活动，而不是 exact-hit claim。本 run 没有使用后来加入的显式
`--coordination-cost-ms`，`.26` threshold 只是其前身；不能把目录名中的 `netvalue` 解读为已
校准的 coordination tax 实验。

该 run 仍是 `unsafe-positive-ablation`，`strict_positive_budget_certificate=false`；即使某个
点估计良好也不能替代 exclusive-host 复测。artifact 的 canonical payload SHA256 为
`8851bf748181c89db0c2850c586658a6ac975eb68ffce36151d99528fa5aa8cf`。

### 5.1 claim-time pull formal：有 C1 收益，但旧路径与测量噪声尚未过关

在 sealed-prefetch 之前，claim-time pull 已完成 `C=1/16/64, R=8`：

`../pattern_v2_direct_pull_netvalue_c5p2_guard10_c1_c16_c64_w64_cpu57_60_r8/`

| Scenario | C | Selected / started / hit | Authority delta ms/target (90% CI) | Logical benefit ms/target (90% CI) | Authority wall delta (90% CI) | 判读 |
|:---|---:|:---:|:---|:---|:---|:---|
| observed | 1 | 24 / 24 / 7 | `-0.0089` `[-0.0591,+0.0412]` | `+0.0849` `[+0.0293,+0.1406]` | `-0.0016%` `[-0.1048%,+0.1015%]` | logical improvement；wall 上界刚越 gate |
| observed | 16 | 22 / 22 / 5 | `+0.1259` `[-0.0374,+0.2892]` | `-0.0702` `[-0.2324,+0.0921]` | `+0.1096%` `[-0.1401%,+0.3594%]` | inconclusive |
| all-wrong | 1 | 24 / 22 / 0 | `+0.0833` `[+0.0049,+0.1618]` | `-0.0833` `[-0.1618,-0.0049]` | `+0.0694%` `[-0.0307%,+0.1696%]` | no-regression 未证实 |
| all-wrong | 16 | 22 / 22 / 0 | `+0.3897` `[+0.1552,+0.6242]` | `-0.3897` `[-0.6242,-0.1552]` | `+0.5220%` `[+0.2158%,+0.8281%]` | raw regression |

C64 由 `Bmax=32` 结构性退化为 K0。这个 run 证明 C1 确有可测的命中收益，但不能证明总体
no-worse。更重要的是，它的 payload/source hash 为
`4c47778f383013112c20c41a4bf6d16fed0ccc9be202e60f88e227e57e65b045`，而当时 K0 A/A 的 hash
不同；后者 C16 又有明显 AB/BA phase bias。因此不能把两者直接相减成正式 DiD，也不能把 raw
C16 回归全解释成 speculative contention。

`coordination_cost=5.2 ms` 还使 effect size 过小：C16 全矩阵只有 22 starts/5 hits，policy 预测
总 net 仅约 `6.74 ms / 1880 targets = 0.0036 ms/target`，远低于当前宿主的分辨率。它适合作为
保守 abstention 点，却不适合检验“资源充裕时机制本身能否带来收益”。

### 5.2 仅把 prefetch 移出 loop 仍不够：历史 S0 R32

历史 offloop source `S0` 已完成 C16/R32。它把 bounded pull 移到 helper，但仍在
authority release/arm 附近观察 prefetch，因此没有删掉所有 treatment-only 时序依赖。

| Cell | Selected / started / visible | Authority ms/target（90% CI） | Logical benefit（90% CI） | Authority wall（90% CI） | 判读 |
|:---|:---:|:---|:---|:---|:---|
| K0 A/A | 0 / 0 / 0 | `-0.0364 [-0.1037,+0.0308]` | `+0.0364 [-0.0308,+0.1037]` | `+0.0147% [-0.0781%,+0.1076%]` | latency pass，wall 仍 inconclusive |
| K1 observed | 773 / 767 / 134 | `+0.2405 [+0.1506,+0.3303]` | `+0.1387 [+0.0371,+0.2403]` | `+0.3063% [+0.1487%,+0.4641%]` | 有 logical gain，但 authority 明确回归 |
| K1 all-wrong | 773 / 761 / 0 | `+0.2510 [+0.1605,+0.3414]` | `-0.2510 [-0.3414,-0.1605]` | `+0.3390% [+0.2154%,+0.4627%]` | 明确回归 |

S0 runner/payload 分别是 `b77b665b...`、K0 `419db037...` 与 K1 `28e0d081...`。这组结果
证明“worker 数多、pull 已 off-loop”并不充分；它不能用来评价下一节的新 S1 silent
机制。

### 5.3 current-source S1：silent prefetch 与 interference envelope

S1 runner hash 为
`6774fbd9b87df6ecfe12857825664ca992424072ff2adb9f78bdfc606999843f`。相比 S0，它使用两个
完全独立的 helper：baseline/treatment 共同的 authority release clock 只 sleep 到 confirmation；
speculative CPU 上的 raw-Future worker 在此前 drain 与 seal，不产生 asyncio callback。parent
被 authority clock 唤醒后先提交并 arm 全部 authority backend，然后才 poll Future 与尝试
parent-local claim。late/error 会抑制当前 claim 并永久关闭这次 replay 的正预算 latch。

| Cell | Selected / started / visible | Amp | Authority ms/target | Logical benefit | Authority wall | 判读 |
|:---|:---:|---:|---:|---:|---:|:---|
| C1 K4 cost1 observed, R1 | 314 / 314 / 47 | `2.336x` | `+0.1396` | `+3.9767` | `-0.0925%` | 只是 mechanism smoke |
| C1 K4 cost1 all-wrong, R1 | 314 / 314 / 0 | `2.336x` | `+0.0539` | `-0.0539` | `-0.0023%` | 只是 mechanism smoke |
| C16 K0 A/A, R8 | 0 / 0 / 0 | `1.000x` | `+0.1937` | `-0.1937` | `+0.2438%` | 基线自身漂移，统计分辨率不足 |
| C16 K1 cost3.5 observed, R8 | 196 / 196 / 42 | `1.104x` | `+0.2226` | `+0.2504` | `+0.3755%` | wall regression |
| C16 K1 cost3.5 all-wrong, R8 | 196 / 196 / 0 | `1.104x` | `+0.2611` | `-0.2611` | `+0.3158%` | latency/logical regression |
| C16 K1 cost4.8 observed, R8 | 64 / 64 / 13 | `1.034x` | `+0.1689` | `-0.0235` | `+0.1682%` | inconclusive |
| C16 K1 cost4.8 all-wrong, R8 | 64 / 63 / 0 | `1.034x` | `+0.1402` | `-0.1402` | `+0.2717%` | inconclusive |
| C16 K1 cost5.2 observed, R8 | 22 / 22 / 6 | `1.012x` | `+0.1108` | `-0.0437` | `+0.2352%` | inconclusive |
| C16 K1 cost5.2 all-wrong, R8 | 22 / 22 / 0 | `1.012x` | `+0.0202` | `-0.0202` | `-0.0191%` | inconclusive，不是 equivalence pass |
| C16 K4 cost5.2 observed, R8 | 24 / 24 / 8 | `1.013x` | `+0.0907` | `-0.0014` | `+0.2460%` | inconclusive |
| C16 K4 cost5.2 all-wrong, R8 | 24 / 24 / 0 | `1.013x` | `+0.2550` | `-0.2550` | `+0.1672%` | inconclusive/logical regression |
| C64 burst fallback, R8 | 0 / 0 / 0 | `1.000x` | — | — | — | 所有 repeat 结构性 K0 |

当 coordination cost 从 `3.5 -> 4.8 -> 5.2 ms` 提高时，C16 K1 starts 从
`196 -> 64 -> 22`、amplification 从 `1.104x -> 1.034x -> 1.012x`，all-wrong authority 点差从
`+0.261 -> +0.140 -> +0.020 ms/target`。这支持 net-value abstention 确实缩小干扰面，但也使
observed logical benefit 从 `+0.250` 变成 `-0.024/-0.044 ms/target`。当前 C16 尚未找到同时
具有可测收益和 no-harm 证据的 operating point。cost5.2 下 K1 增至 K4 只多启动 2 个
候选，说明此时限制来自 value filter，不是 slot 数。

C1 smoke 的 314 个 prefetch 均 off-parent-loop，p95 `0.234 ms`，最小 worker-body quiet gap
`12.43 ms`，late/error、backend-arm violation、unready claim 都为 0。C64 因 authority burst
超过 `Bmax=32` 而从第一个 batch 起保持 K0，sidecar 完全未激活。C64 的非零 A/B latency/
wall 点差因此只能是 host/order noise，不能归因于 speculation。

这些 S1 正预算实验都是 `unsafe-positive-ablation`，因当前 host 没有 external exclusive
resource entitlement，`strict_positive_budget_certificate=false`。R1 不支持性能推断，R8 也没有
建立统计等价性；同源 C16 K0 A/A 自身就有 `+0.1937 ms/target` 偏移。所以现阶段最
稳妥的 paper 结论是“新机制删掉了可见的 control-path hazard，但正 K 的 formal no-harm 尚未
证实”，而不是把 cost5.2 的近零点估计写成 pass。对应 artifacts 是：

- `../pattern_v2_silent_k4_guard15_c1_cpu8_12_smoke_r1/`
- `../pattern_v2_silent_net35_k0_aa_c16_cpu8_12_r8/`
- `../pattern_v2_silent_net35_k1_guard15_c16_cpu8_12_r8/`
- `../pattern_v2_silent_net48_k1_guard15_c16_cpu8_12_r8/`
- `../pattern_v2_silent_net52_k1_guard15_c16_cpu8_12_r8/`
- `../pattern_v2_silent_net52_k4_guard15_c16_cpu8_12_r8/`
- `../pattern_v2_interference_envelope_c64_k0_fallback_cpu8_12_r8/`

## 6. 多 Session 结论的边界

当前 runner 的 `C` 是 lockstep batch width：多个 session 在一个 epoch 内形成同步 burst，并在
strict shadow barrier 后推进。它很好地暴露 coherent timer/callback stall，也验证全局 `K` 而非
per-session `K` 的必要性；但它不是任意相位、随机到达、长期存活的异步 multi-session 服务。

因此当前结果不能证明：

- 任意 staggered arrival 下的 tail latency 与公平性；
- session churn、不同 tool service distribution 下的稳定性；
- 共享真实 backend/rate limiter 时仍无征用。

论文可以把 lockstep C1/C16/C64 定位为受控的同步压力实验。若要扩展一般化 claim，只需再加入
一个 randomized phase-offset workload；不需要为当前 paper 过度设计在线 fairness scheduler。

## 7. 可支持与不可支持的论文 claim

当前证据可以支持：

- 多 session 下的首要风险是共享串行 control/result path，而不只是 predictor accuracy 或 tool
  slot 数量；
- process authority 的 child 执行收益会被 parent handoff 税反转，authority batching 还会制造
  coherent stall；
- 两轮 authority-first 可以把 treatment lookup/claim 放到 authority backend 已 arm 之后，并由
  runtime counter 检验顺序；
- 延后 lease 回收删除了 all-wrong confirmation path 的 cleanup traffic；
- eager parent-local staging 能降低 hit-conditioned delivery penalty，但必须把 wrong-result
  staging 计入 net value；
- silent sealed pull 已在 mechanism smoke 中做到 raw prefetch 无 asyncio callback、worker 与
  authority clock 独立、post-arm 才 poll、所有 hit 的 claim packet 为 0、authority 前无 bridge、
  all-wrong 无 claim；
- `K_eff=min(K_requested,K_cap,K_safe)` 的全局、fail-closed gate 比 per-session 贪心投机更符合
  no-regression 第一优先级；资源充裕只能增大 `K_cap`，不能绕过同 concurrency/interference
  domain 的 all-wrong/A/A 校准所给出的 `K_safe`。

当前证据不能支持：

- “96 CPUs/轻量工具意味着正投机不会拖慢 authority”；
- 未经 external entitlement 的正 `K` 具有逐 trace no-worse 保证；
- 把 `inconclusive`、A/A 假 improvement 或命中子集改善当成全局收益；
- lockstep C 已代表所有异步 multi-session arrival。

在获得独占 host/backend 证书并完成 matched all-wrong/A/A 校准之前，唯一可诚实写成 strict
default 的行为仍是：任一证书不足则 `K=0`；二者充分时才用原始 authority + baseline-identical
release clock + silent raw prefetch + 两轮 authority-first + precompletion，按 global net value
在 `K_eff` 内分配资源。

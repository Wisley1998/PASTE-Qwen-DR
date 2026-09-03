# 多 Session 下的 authority-safe speculative execution

## 1. 目标

策略采用字典序目标：

1. 先保护 demand-only authority 的资源权益和执行路径；
2. 只在独立的剩余资源中，最大化所有 active sessions 的期望关键路径收益。

这比把“收益”和“回归风险”混进一个加权分数更容易解释和审计。若独立资源证书无效，或
对应 interference-domain 的干扰校准缺失，安全动作就是 `K=0`，直接退化到 demand-only。

“任何情况下不更差”必须限定资源域、载荷模型与宿主假设。本文能给出的硬保证是：在固定的
exogenous authority arrival trace、已声明的 payload/service 上界和已认证的隔离域内，
speculation 不占用 authority worker、tool capacity、CPU run queue、start-rate token 或
backend quota。不同逻辑 CPU、不同物理核只证明局部 run-queue 分离；若 authority 与 sidecar
仍共享 LLC、内存控制器、内核调度或功耗域，就不能把它们称为不同的 interference domain。
远端限流、网络拥塞以及任意恶意反序列化时间也不能凭本地实验自动纳入保证。缺少外部
exclusive CPU/backend certificate，或缺少与当前 concurrency/interference domain 匹配的
all-wrong/A/A 校准时，默认策略始终是结构性 `K=0`，而不是根据“机器看起来很空”推测可安全
开启。

## 2. 根因：共享 idle capacity 不是安全容量

设一个不可抢占工具的共享容量为 `C`。如果 wrong speculation 可以占用其中任意一个
slot，那么在它运行期间到达 `C` 个 authority calls 时，至少有一个 authority 必须等待。
authority-first queue priority 只能优先尚未开始的请求，无法抢占已经运行的调用。因此，
无论预测精度多高、平均资源多空闲，只要 authority 和 speculation 共享不可抢占的物理
slot，就不存在逐 arrival-trace 的 no-regression 保证。

早期的同进程双 lane 方案虽然保留了 authority worker/tool cap，仍共享：

- Python event loop 和 GIL；
- dispatcher、lock、cleanup scan 与 wakeup；
- 每个候选的 submit/tombstone IPC；
- 下游 connection pool、rate-limit token 或 backend quota。

轻量工具执行时间短时，控制面固定成本占比反而更大。高并发又会把几十微秒的 admission、
清理和 timer rounding 放大为可见的 queue/wall regression，所以“机器总体资源充裕”不能
排除这些局部串行点。

当前源码的对照更直接地支持这个定位。C16/R16 在完全不运行 speculation 的 K0 条件下，把
原始 in-process authority 搬到独立进程后，child 内 scheduled-to-terminal 反而快
`0.488 ms/target`，但回到 agent/control loop 的 observed latency 慢 `0.158 ms/target`，
authority wall 明确回归 `1.303%`（90% CI `[1.031%,1.576%]`）。执行核不是瓶颈；同步
Pipe submit、child pickle/send、parent bridge 解码和 Future/event-loop handoff 的固定税把
执行收益全部吃掉。把 submit/result 合成 synchronized burst 更差，wall 回归约 `2.493%`，
因为相同 service timer 与批量 Future callback 在单 event loop 上形成 coherent stall。

这对轻量工具尤其重要：tool 本身越短，固定 IPC、GIL 和 callback 成本占比越大。“机器有
96 CPUs”也不是反证。本轮实验期间 load average 约 73、CPU pressure `some avg10` 约 13.5%；
静态 pin 的 sidecar CPU11 在一秒样本中曾有约 87% busy。affinity 只限制本进程去哪一个核，
并不阻止其他租户使用该核，更不是 backend quota entitlement。

hit-conditioned 分解还找到了第二个串行点：lazy claim 会在真实 authority 已提交后才要求 child
序列化结果，并经 parent bridge 与 `call_soon_threadsafe` 唤醒 control loop。新的 eager-staging
消融把完成值先放进 parent 私有有界 map；exact confirmation 只做本地 try-lock/O(1) claim，
未 ready 立即 miss，且不发 child claim packet。C16 中 hit targets 的 authority 点差从 lazy
约 `+0.820` 降到 eager 约 `+0.290 ms/target`。但 eager 会搬运所有 selected 结果；当前约 24%
precision 下，wrong-result traffic 使 all-wrong wall 点差从 lazy 约 `+0.173%` 增到 eager
`+0.288%`。因此 eager 证明了“去掉 claim round-trip”有价值，却不是最终默认。current-source
新增 pull mailbox：child 将 completion best-effort 写入有界 kernel socket，但不运行持续 parent
reader。早期 sealed-pull 虽把 bounded decode 移到 authority 前，仍让 parent event loop 在 cutoff
处等待/观察 prefetch，或者让 prefetch 与 authority release 共用一个 worker；这会制造一条
treatment-only 的 cutoff wakeup，轻量工具下仍足以改变后续 timer phase。最新路径改为
**silent raw prefetch**：baseline 与 treatment 使用同一条仅负责 confirmation 的 release clock；
prefetch 由 pin 到 sidecar CPU 的独立 worker 通过 raw `concurrent.futures.Future` 执行，既不
`await`，也不包装为 asyncio Future，因此 cutoff 完成时没有 callback 唤醒 parent loop。worker
在 guard 前半段最多 non-blocking drain `max_pending` 个 datagram 并 seal epoch，guard 后半段
保持 quiet；parent 只在 planned confirmation 被 baseline-identical clock 唤醒，先提交并 arm
全部 authority backend，随后才 non-blocking poll raw Future。mailbox 满、结果未 ready、worker
late/error 或 seal 失败都会禁止本 epoch claim 并关闭后续正预算 latch，而不是等待 prefetch。

现有 timing fields 可把 slowdown 分成四类，避免把所有正点差都误写成“资源不够”：

1. **真实执行资源竞争不是当前主因。** direct-pull C16 all-wrong 的 physical call
   amplification 仅 `1.0117x`，authority 最大并发远低于 `64` 的 worker/tool cap，broker queue
   只增加约 `0.009 ms/target`；sidecar 也使用独立进程与逻辑 CPU。
2. **authority/control burst 是可定位的固定税。** 旧 pull artifact 的 C16 all-wrong
   `+0.390 ms/target` 中，first-run lag 约 `+0.103 ms`；lazy/eager/pull 的 C16 first-run 点差都
   约为 `+0.055` 至 `+0.103 ms`。根因是 treatment-specific exact lookup 曾位于 authority
   backend arm 之前，而且旧 guard prefetch 仍能在 cutoff 唤醒 parent loop。因此 current-source
   同时采用下文的 two-turn authority-first 与 silent raw prefetch。
3. **lazy result handoff 是明确串行点。** process-authority K0 的 child service 虽更快，parent
   return handoff 仍约 `0.646 ms/target`；hit-conditioned lazy/eager authority 点差约为
   `+0.820/+0.290 ms`。silent sealed pull 删除 claim-time child round-trip，把 bounded decode
   移到 authority 前的 guard，并避免 prefetch completion 产生 asyncio callback。
4. **宿主噪声与剩余点差同量级。** identical K0/K0 在 C16 已产生约
   `+0.196 ms/target`、`+0.302% wall` 的伪回归，且 AB/BA 方向不同。因此 service/timer
   stretch 不能自动归因于 speculation；必须先通过同源 A/A resolution gate。

这个分解也解释了为什么增加 worker 没有直接解决问题：高并发放大的是同一个 parent loop 上的
串行工作与 timer phase，而不是当前实验中的 tool-slot queue。

## 3. 最小安全架构

### 3.1 原始 Authority 快路径保持不变

推荐部署路径继续在原 parent loop 使用未修改的 demand-only `LiveToolBroker`。speculation 不应
为了“隔离”而把 authority 搬到另一个进程：最新 K0 对照已经证明该迁移本身会增加 observed
handoff 与 wall。`ProcessAuthorityLane` 仍保留为诊断臂，用来区分 child execution、IPC return
和 sidecar 增量成本，但不是系统默认。

因此完整实验至少有三个臂：A=原始 in-process K0，B=process-authority K0，C=与 B 相同的
process-authority+K1。`C-B` 只估计 sidecar 增量，`B-A` 估计 topology migration，`C-A` 才是
真实部署差。当前 `--direct-authority-baseline` 只是 A/C 两臂快捷诊断，不等于同一 block 内完整
三臂；正式 paper run 应在同一 seed block 内 counterbalance A/B/C。B/C 都属于
`unsafe-positive-ablation` 的瓶颈诊断，不能取得 public strict certificate，也不能替代 direct
authority 的部署比较。

shadow-evaluation 模式中，每个真实 target 都先无条件提交一条 authority call。direct 模式使用
两个 authority-first event-loop turns：第一轮完成 broker admission/runner 创建，第二轮让 runner
进入 executor 并 arm backend/service await；只有此后才做 treatment-specific exact lookup。
runner 将“每次 speculative claim 前 authority backend 均已 arm”作为逐 target invariant，而不是
只观察 task 已被 `create_task`。这避免把 exact-key hash/lookup 的时间从 `first_run_lag` 简单搬到
尚未启动的 broker service。
process 诊断模式只要求请求已不可撤销写入有序 Pipe。随后只允许 exact-key、try-lock、
transport-non-blocking claim；不等待 child round-trip。锁忙、
未 ready 或 transport 满时立即 miss。命中时 logical result 是：

```text
first(valid speculative success, authority terminal result)
```

Authority task 不被取消。Speculation reject、过期、失败或结果 identity 不匹配时，仅回退到
已经存在的 authority task，不创建第三次调用。因而 logical result 不会晚于其 shadow
authority terminal result；代价是命中后 shadow authority 仍会自然 drain。

保留原始 authority 的推荐严格正预算档使用 silent sealed pull staging。sidecar 在 modeled
confirmation guard 前把 terminal outcome best-effort 放入有界 kernel mailbox，parent 不启动
持续 bridge。runner 维护两个互不依赖的单线程 clock：authority release clock 在 baseline 与
treatment 中完全相同，只睡到 planned confirmation；silent prefetch clock 只在 treatment 有
submitted handle 时工作，并 pin 到 sidecar CPU，而非 authority/control CPU。prefetch 使用
`ThreadPoolExecutor.submit()` 返回的 raw concurrent Future；它从不被
`asyncio.wrap_future()`/`run_in_executor()` 包装，也不注册 loop callback。因此 prefetch 在
cutoff 完成不会把 parent loop 加入 ready queue，更不会决定 authority 何时 release。

对于 guard 长度 `G`，silent worker 在 `confirmation-G` 开始 bounded `MSG_DONTWAIT` drain，必须
在 `confirmation-G/2` 前 seal；剩余半个 guard 是显式 quiet gap。到 planned confirmation 后，
parent 仅由 baseline-identical release clock 唤醒，先创建所有真实 authority calls，再用两个
authority-first turns 让 broker runner 进入 executor/backend await。只有 backend-arm invariant
通过后，parent 才用 `Future.done()` non-blocking poll prefetch outcome，并做 request-id/key 的
parent-local O(1) lookup。poll 时未完成、deadline/error、未 seal 或 backend 未 arm，都会令当前
epoch 的 claim 关闭，并永久关闭该 replay 的正预算 latch；authority 不等它、不重试。未 ready、
receiver busy、malformed、过期或 mailbox 丢包同样立即 miss 且不发 child claim packet。

零真实 target 的 false-positive decision 也不能绕过协议：若该 decision 已提交 prefetch job，
runner 会在进入下一 batch 前 non-blocking poll；not-ready/late/error 同样关闭 latch，避免孤儿
worker 与后续 authority epoch 重叠。最终 teardown 则先等待所有 authority terminal，再
`shutdown(wait=True)` join silent worker，之后才允许 snapshot/lifecycle bridge/sidecar close
接触同一个 result socket（join-before-close）。eager parent-local staging 也不需要 claim
round-trip，但会持续唤醒 parent 搬运 wrong result，只保留为机制消融；lazy delivery 命中时才
搬结果，却需要 claim-time child/bridge round-trip，同样不是推荐 strict direct 路径。

completion race 使用 single-notification：speculative payload 的 identity 在 bridge/concurrent-
future callback 中先验证，control loop 只接收一个仍可能获胜的通知；authority child 的 raw
future 已 terminal 时，即使 control proxy 尚未醒来也由 authority 赢 tie。observer 的取消用
shield 隔离，不能反向取消真实 authority。报告同时保留 child `lane_terminal_at` 与 control
`observed_at`，避免把 authority IPC 延迟误算成 speculative benefit。

严格模式增加一个 shadow-debt barrier：有 speculative early return 的 batch 可以立即把结果
交给 agent 做模型计算，但下一 batch 的真实 authority calls 在本 batch 的 shadow backups
全部 terminal 前不能进入 protected broker。它阻止“上一批未 drain 的 authority + 被命中
提前释放的下一批 authority”叠加，从而保护其他 sessions。最后一个 batch 不需要等待 barrier，
仍可保留最终 agent-visible early completion。该 guard 很简单，但会牺牲连续 tool-chain 的
一部分 wall speedup；若要保留这部分收益，只能让提前到达的真实调用使用另一份 certified
surplus capacity，而不能提前占 baseline lane。

### 3.2 Speculation 放进独立 sidecar

Speculative executor 和 scheduler 运行在 fork child 中，使用独立 event loop、GIL、worker
和 tool capacity。parent/child 通过 bounded `AF_UNIX/SOCK_SEQPACKET` 传输，避免
`multiprocessing.Queue` feeder thread 常驻 parent。

CPU helper 读取 sysfs topology，依次选择“同 NUMA、不同物理核”、“同 socket、不同物理
核”，最后才考虑其他不相交逻辑 CPU，并尽量避开 SMT sibling。三角色模式稳定地选择
`authority / control / sidecar`，正式 pilot 用 `taskset -c 10,11,12`；sidecar child 同时设置
Linux `SCHED_IDLE`。silent prefetch worker 与 sidecar 使用同一 CPU mask，使 prefetch decode
不会占 authority/control run queue；它不是第四份资源，也不能偷偷 pin 回 parent CPU。
这只是 affinity，不是 core reservation：本机实验期间 load average 约
`65–71/96 CPUs`，一秒窗口中候选核可从低负载快速变为高负载。独立 run queue 能去掉本任务
内部的直接 CPU slot 竞争，却不能隔离其他租户、LLC、内存带宽、内核 timer、功耗域和宿主
调度。严格证书必须来自 exclusive cpuset/isolated host，而不是只看 mask 不相交。真实部署还
必须给 sidecar 独立的连接池、rate token 和 backend quota。

物理核证书与 interference-domain 证书必须分开。若要声明 cache-level no-interference，sysfs
中的 LLC/cache sharing set 也必须不相交（通常意味着把 sidecar 放到不同 LLC domain，必要时
不同 NUMA/socket），并同时约束内存带宽与功耗共享；“同 socket、不同 core”不能支持这项强
claim。若为了 NUMA locality 选择共享 LLC，则只能在**同一 LLC placement、同一 concurrency、
同一 tool class** 的 matched all-wrong/A/A block 上估计干扰预算 `K_safe`；结果不能外推到另一
socket、另一 host 或更大的 burst。LLC domain 未记录、placement 改变或校准缺失时，严格预算
回到 `K=0`。这条限制比继续寻找瞬时 idle core 更直接，也更适合 paper 的可审计边界。

专用 authority **线程**只隔离 event-loop ready queue，仍与 parent bridge/control 共享 GIL；
专用 authority **进程**虽然改善 child execution，却引入更大的 return-handoff 税。因此推荐
最小拓扑是“原始 authority + 一个 process sidecar”的两角色路径；三角色、两个 child process
只用于机制分解。无论两角色还是三角色，静态 affinity 都必须与外部 exclusive cpuset/backend
entitlement 分开记录。

### 3.3 Authority 附近的控制路径必须 bounded、non-blocking、fail-open

一个 decision epoch 的候选只发送一个 bounded datagram，parent registry 采用全有或全无
注册。发送锁忙、socket 满、packet 过大、状态异常或 priority 非正时，整批立即返回 miss，
authority 不重试、不等待。

每个候选持有有限 lease：

```text
lease_end = start_deadline + claim_grace
claim_grace = service_upper + completion_guard + 10 ms   # precompletion profile
```

普通模式默认 `claim_grace=10 ms`。precompletion runner 使用上式，使 exact confirmation 有固定
调度余量；parent 只在 shadow authority 已 drain 后的下一次 submit
时懒回收 registry。lazy authority miss path 不发送 tombstone/terminal packet。fixed-arrival
all-wrong runner 在 authority 完成前不启动 result bridge；随后为读取完整 snapshot/关闭 child
可能启动它，这不属于 timed authority path。observed runner 若 sidecar 确实会运行，则在计时前
为 lazy/eager 消融启动一个阻塞于内核读的 bridge，避免第一次 hit 在 authority hot path 创建
线程；pull 模式则明确禁止 timed bridge。若默认的 no-certificate gate 全程关闭，则 process、
CPU reservation、preload、bridge 和 IPC 都不创建，是真正的 K0 fallback。

lazy 成功命中仍需一次 claim-time 跨进程通知。eager/pull 都由 child 提前传输 terminal outcome，
parent registry/map 受 `result_capacity`、lease 与 TTL 三重约束；live 或 intra-batch duplicate
exact key 整批拒绝，取消竞态用 `Future.set_running_or_notify_cancel()` fail-open。exact claim 前
public Future 不完成，ready claim 与 not-ready miss 都不发 child packet。eager 的 bridge 立即
解码所有 outcome；sealed pull 只在 pre-authority guard 内执行有界 try-lock/receive，封存后
claim 不再接触 transport。成功 payload 上限为 4 KiB，外层 datagram 也有固定 envelope 上限；
receiver 最多读取“4 KiB inner result + 固定 metadata/envelope”对应的 outer cap，而不是任意大的
`max_packet_bytes`；异常 packet、锁忙或 decode error 都 drop/miss。这个字节上界提供有界工作量，
但若允许任意不可信 pickle reducer，仍不能推出绝对时间上界，因此 strict profile 只接受受控的
primitive/tool-result schema。两者的最小研究实现有 record 时都在 child 侧
1 ms polling，但该 polling 在隔离进程/CPU 内，并由 precompletion cutoff 与 authority 分相。

“transport-non-blocking”不等于任意线程下 lock-free/wait-free。registry acquisition 是 fail-open
try-lock，但本地 Future、handle、epoch flag 与统计计数仍有短锁；strict runner 通过禁止并发
bridge/submit/snapshot/close 使这些锁在封闭协议内无竞争。外部调用者若破坏该协议，不能继续
引用此时间上界。prefetch deadline 也在每个 packet 前检查，随后受字节上限约束的 decode 仍可能
跨过 deadline；这会使当前 claim 被抑制并 trip runtime latch，而不是被事后解释成安全成功。

当前 positive runner 先 fork child，再启动两个 runner-owned clock thread，避免从多线程 parent
继续 fork。pull 模式用 setup-only `startup_snapshot()` 直接读取 child affinity/`SCHED_IDLE`
证书，并明确断言 timed authority 完成前 `bridge_started=false`。authority release clock 只执行
absolute sleep，baseline/treatment 一致；silent prefetch clock 只执行 bounded pull/seal，raw
Future 没有 asyncio callback 且其实际 affinity 必须等于 sidecar mask。普通 `snapshot/close` 只在
authority 全部 drain 且 silent worker join 后启动 lifecycle bridge；任何取消/异常清理也遵循
join-before-close。eager/lazy 消融仍可在计时前预启动 bridge。candidate selection 与进程启动均
在 timed authority path 外；一个 epoch 的 ingress 合并为一个 bounded packet。

可选的 phase guard 只把 epoch 2+ 的 speculative release 推迟到上一个 modeled authority
boundary 之后，authority arrivals 不变。跨开发版本的 guard runs 没有呈现稳定
dose-response；它们只支持继续把 guard 当作减少 timer 同相唤醒的 ablation，不能形成严格的
同版本 sweep 结论，更不是安全证书。

## 4. 多 Session 的全局调度

`K` 是一个资源域跨所有 sessions 的全局运行预算，不能让每个 session 各自拥有 `K`。资源
“有多少”与 authority-safe 地“能用多少”是两个不同上界：

```text
K_cap  = 独立 sidecar worker/backend entitlement 给出的物理容量上界
K_safe = matched all-wrong/A/A 干扰校准允许的并发上界
K_eff  = min(K_requested, K_cap, K_safe)
```

`K_cap` 只能来自 exclusive worker/CPU/backend quota certificate；机器的 idle CPU 数、sidecar
queue 长度或 predictor 命中率都不能提高它。对每个已冻结的
`(tool_class, concurrency_bucket, interference_domain)`，先在 `k=0..K_cap` 运行 matched A/A 与
all-wrong negative control，定义：

```text
I_lat^UCB(k)  = UCB95(Delta_allwrong_lat(k)  - Delta_AA_lat)
I_wall^UCB(k) = UCB95(Delta_allwrong_wall(k) - Delta_AA_wall)

K_safe = max k such that, for every j <= k, both UCBs(j) are within
         preregistered no-harm margins
```

若论文采用字面零回归，margin 就设为 0；若采用 practical equivalence，则必须预注册并同时
报告 `0.10 ms/target`、`0.1% wall` 等 margin。A/A 没有足够 resolution、all-wrong CI 不通过、
placement/LLC/burst 超出校准格，或根本没有校准时，`K_safe=0`。这样“资源充裕”最多增大
`K_cap`，不会绕过 authority interference gate。为避免把一个 R8 点估计硬编码成线上 controller，
研究版只使用离线表格与 fail-closed lookup，不引入 AIMD/bandit。

每个 `(session_id, decision_id)` 最多提交一个候选，exact identity 为：

```text
(session_id, decision_id, tool_name, canonical_arguments, context_token)
```

对候选 `i` 定义：

```text
p_i = 校准后的 exact-match probability
S_i = 预计 tool service time
L_i = authority confirmation 前可覆盖的 lead
w_i = task weight，默认 1

V_i = w_i * p_i * min(S_i, L_i)
c_i = 同一校准格中 all-wrong 相对 A/A 的保守 per-start 协调税 UCB
U_i = V_i - c_i
D_i = U_i / S_i
```

`c_i` 与 `K_safe` 来自同一 matched block，但作用不同：`K_safe` 限制一个 epoch 的总干扰，
`c_i` 防止单个轻量请求的预期收益小于 admission/prefetch/claim 固定税。最小估计可以把
`I_lat^UCB(k)` 的保守增量按实际 starts 摊为 tool-class 常数、下限截为 0，并加已测的 bounded
admission/prefetch CPU time；无需拟合复杂模型。若相应 per-start calibration 缺失，令
`c_i=+inf`，候选
自然 abstain。不能用 observed hit test set 反向调小这个成本。

先过滤 `p_i < threshold` 或 `U_i <= 0`，再按 `U_i` 做 epoch 内 global Top-`K_eff`；scheduler 用
`D_i` 排 queued work。代码用 `--coordination-cost-ms` 显式提供固定的 `c_i`；默认 `0` 只为
兼容旧实验，不能用于正式正预算 claim。单一模拟工具使用一个常数；多工具场景可按 tool class
查表，不需要引入 per-session controller。service time 相同时，扣除同一常数不改变已通过净收益
门的候选次序。pending 集合有固定上限，后来更高收益的候选只可替换 queued work，不能把
running work 伪装成可抢占。

最小研究实现不在 online hot path 重新估计这些量：实验 harness 从预注册 calibration manifest
读取 `K_cap/K_safe/c_i/Bmax`，把 `min(K_cap,K_safe)` 作为 runner 的正 start budget，并把 `c_i`
传给已有 net-value selector。manifest key 不匹配就传 K0。这样只增加一个离线表 lookup，不需要
实现长期自治 controller；artifact 必须记录 calibration source/hash，避免手工把某个方向好看的
R8 点估计当成正式预算。

严格档再加一个简单的“可预完成”门，不把 prediction accuracy 与资源安全混在一起：

```text
predicted_start + S_i + completion_guard <= authority_confirmation
```

不满足时 strict effective budget 为 0；允许其与 authority 重叠的版本只能标为 best-effort。
这里的 `S_i` 在 strict 档必须是经过校准的 service upper bound，而不是均值。runner 把 child
的 latest-start fence 设为 `planned_confirmation - S_i - guard`，并在真正进入 executor 的
下一 event-loop turn 再检查一次；过期任务不调用 executor。结果交付还用固定的 planned
confirmation cutoff（不能用可能被拖晚的实际 confirmation）检查真实 terminal timestamp；
晚结果不能成为 visible hit。claim lease 与 latest-start 分离，并把未命中回收推到 modeled
authority interval 之后，避免“为了等真实 arguments”而让 cleanup 回到保护窗口。

这条门减少 wrong speculative execution 与真实 authority 的时间重叠，但不能凭估计值停止一个
已经运行且不可抢占、又超过 `S_i` 的远端调用；严格模式仍要求独立 backend entitlement，或
可由 sandbox 强制终止的 hard deadline。precompletion 只是资源/时序门，不是零控制成本证明。

`threshold=0.20, K=4` 是早期离线 sweep 的收益/physical-call knee，但不是安全默认。当前
严格默认是 `K_cap=K_safe=0`：即使 CLI 请求正 `K`，没有显式资源证书与 matched 干扰校准也不
启动 sidecar。host/backend certificate 只能把 `K_cap` 打开；all-wrong/A/A UCB 只验证到
`K=1` 时，`K_safe` 仍只能是 1，不能因为另有三条空闲 worker 就擅自升到 4。提高到
`threshold=0.25` 可减少工作量，但它只改善收益密度，不能把共享控制成本变成零。若不同任务有
重要性差异，可只通过有界 `w_i` 表示；本文不再加入复杂的在线 bandit 或 per-tenant fairness
optimizer。

### 4.1 简单的 fail-closed 资源证书

研究 policy 的状态只有 `CLOSED` 和 `OPEN(K_eff)`：

```text
缺任一 strict-profile 前置条件                 -> CLOSED, effective K = 0
K_cap/K_safe/cost 校准完整且 burst <= Bmax     -> OPEN,   K_eff=min(K_req,K_cap,K_safe)
prefetch late/error/unsealed 或 backend 未 arm  -> CLOSED，当前 epoch 不 claim，replay 不自动重开
任一 epoch 的 burst > Bmax                     -> CLOSED，当前 replay/session 不再自动重开
```

完整静态前置条件是：process sidecar、CPU isolation、strict shadow barrier、precompletion、
显式 `Bmax`、调用方的 exclusive CPU/backend attestation、对应 interference-domain 的
`K_safe/c_i` calibration，以及不会在 confirmation 热路径等待 child 的结果交付方式。推荐且
唯一能取得 public strict certificate 的路径是“原始 direct authority + baseline-identical release
clock + silent raw prefetch + two-turn backend arm + post-arm poll”。silent epoch 必须在 authority
前完成 bounded prefetch/seal，并留下至少半个 guard 的 quiet gap；authority 期间不得启动
bridge、读取 result socket 或解码 payload。prefetch Future 不能注册 asyncio callback，其 worker
必须位于 sidecar CPU；late/error/unsealed 时 current claim 必须被抑制，后续 latch 保持 K0，且
teardown 必须 join-before-close。inner/outer byte cap、zero-target handling 与 backend-arm invariant
也都必须通过。process authority + return bridge 只能用 `unsafe-positive-ablation` 做
topology-migration 诊断。runner 从 sysfs 验证涉及的逻辑 CPU 是可证明的不同物理核；未知拓扑或
SMT sibling 均 fail-closed。若 claim 涉及 LLC-level isolation，还必须记录并验证 cache-sharing
domain；共享 LLC 的结果只能依赖该 placement 的经验 `K_safe`，不能引用“不同物理核”作为强
隔离证书。仅传 `Bmax` 不再打开正预算。需要复现旧设计时必须显式传
`unsafe-positive-ablation`，artifact 会记录该状态，不能用于 no-regression claim。

sidecar fork 后、计时前还有一次 ready snapshot handshake，验证 child affinity 和
`SCHED_IDLE`；非-pull 模式还验证 bridge affinity。失败会关闭 child、清空 selection，再进入
timed K0 path。这个 fallback
只保证计时路径结构 K0，不是 lifecycle-identical K0，因为 setup 中曾短暂 fork。latch 一旦关闭
就保持关闭，避免高并发附近反复启停和 phase oscillation。单次 transport admission 失败仍
non-blocking miss，不会反向关闭整个 replay latch。

完整部署证书还应覆盖五个互相独立的域：sidecar CPU/worker，tool/backend quota，parent
control plane 的 `Bmax`，结果/网络连接配额，以及 LLC/内存/功耗 interference domain。前四类
容量只决定 `K_cap`；同 placement 的 all-wrong/A/A UCB 再决定 `K_safe` 和 `c_i`。transport、
affinity、quota、event-loop health 或 silent-prefetch runtime certificate 异常时，应在当前 claim
点 fail-open 并关闭后续预算。只看到“很多 idle CPU”或“tool capacity=64”不足以打开 gate。
`--certified-exclusive-resources` 是外部 attestation，不是 runner 对远端 quota 的自动探测；本文
停在可复现的静态 profile、离线校准表、启动握手与 replay 内 latch，不再加入 AIMD/bandit 等
工程控制器。

## 5. 可证明的性质与边界

在以下假设下：

- sidecar 的 worker、tool、CPU、connection、rate token 和 backend quota 均独立；共享 LLC/
  内存/功耗域的剩余干扰已经在同 placement 的 `K_safe` 校准中保守覆盖，否则这些域也必须独立；
- `K_eff=min(K_requested,K_cap,K_safe)` 是资源域内跨所有 sessions 的单一全局额度；
- executor 退出前一直持有其物理 slot，不存在未计数的后台工作；
- direct authority 的 backend/service await 在任何 speculative exact lookup 前已经 arm；
- silent-pull prefetch、payload decode 与 epoch seal 在 guard 前半段完成，后半段保持 quiet；raw
  Future 无 asyncio callback，authority release clock 不依赖 prefetch，authority interval 内没有
  result transport/bridge 活动；
- authority backend arm 后才 poll prefetch；late/error/unsealed、zero-target orphan 或 arm violation
  均抑制 claim 并关闭 latch，所有 teardown 都先 join raw prefetch worker 再 snapshot/close；
- authority/control CPU 的外部宿主负载、共享 cache/内存/功耗域干扰不超过证书声明的边界；
- speculative tool 只读/幂等，或副作用已隔离、去重；
- 提前产生的结果在 authority confirmation 时仍有效；
- exact identity 包含 tenant、credential、tool/version、数据快照和授权上下文；本实验的空
  `context_token` 只适用于单租户受控 replay；
- 讨论固定 exogenous arrival trace，且调用方不取消正常 race；

可以声明：

1. wrong speculation 不占 baseline authority broker slot；
2. sidecar 同时运行数不超过全局 `K_eff`，pending 不随 session 数按 `N*K` 增长；
3. transport 饱和或证书失效会 fail-open 到 demand-only result path；
4. exact claim 才能交付结果，失败只回退到已有 authority call；
5. 在同一 target 上，logical completion 不晚于该 treatment 的 shadow-authority completion。

这些是条件性保证，不是对任意宿主、任意 payload、任意远端服务和任意内生 arrival 的绝对
“零微秒 slowdown”承诺。任一资源域、载荷或宿主假设不能验证时，唯一严格动作是不开 sidecar
且取 `K=0`；正 `K` 只能标为 uncertified/best-effort ablation。

“隔离”在此特指模型中的执行容量和 cleanup 的相位分离；它不表示进程之间没有任何内核、
内存或结果通知开销。lazy all-wrong 可以避免 result traffic，eager staging 会在 parent 搬运
wrong result，旧 pull 会在 parent cutoff 产生 treatment-only wakeup，silent sealed pull 则由
sidecar CPU 上无 loop callback 的 raw worker 有界搬运后封存；所有路径都必须把各自剩余的协调
税计入 `U_i`，并由 `K_safe` 覆盖聚合干扰。

不能据此声明 closed-loop 中每一条后续 authority call 的 response latency 都下降。一次 hit
会让 session 更早产生下一次 authority arrival；它可能在前一条 shadow backup 尚未 drain
时排队。这是内生 arrival / coordinated-omission 效应：相对 sojourn latency 可以增加，
同时绝对完成时间和 agent-visible latency 改善。

因此实验必须分开回答三个问题：

- `all-wrong closed-loop`：same fixed workload/no-hit 的负控，但不是固定时间戳 replay；
- `open-loop all-wrong`：预先生成绝对 arrival schedule，检验直接资源征用；
- `observed strict-barrier`：检验 agent-visible 收益，同时禁止 unresolved shadow debt 传播到
  下一批 protected authority。

当前 closed-loop runner 的 `C` 是同步 decision-batch 宽度：同一批 sessions 一起 admission、
confirmation 和 gather；它是 authority burst stress，并不覆盖 session A 的 result callback 与
session B 的任意 staggered authority arrival。fixed-arrival supplement 会预创建全部 authority
tasks，但仍沿用 epoch 化 source-session cadence。论文应把当前结论限定为 burst/epoch workload，
把随机相位的 staggered multi-session replay 作为后续外部有效性实验，而不能从 `C=64` 直接推出
任意异步 serving traffic 的硬保证。

结构性资源不变量和有噪声的 wall-clock estimate 也必须分开报告。若实验的 A/A 噪声大于
预设 equivalence margin，统计结论只能是 inconclusive，不能把“未显著回归”写成“证明
零回归”。

## 6. 实验与验收

主实验在 Task concurrency `C=1/16/64` 下，对 observed nested OOF 和 deterministic
all-wrong 各做 8 次 paired AB/BA，并启用 strict shadow barrier。正式 block 先做 matched
K0/K0 A/A；其噪声上界若已超过预注册 margin，该 block 不具备判断 sub-ms no-regression 的
分辨率，不继续解释正 K。对每个打算启用的 `k<=K_cap`，all-wrong 相对 A/A 的 UCB 同时生成
`K_safe` 与保守 coordination cost；缺格、跨 concurrency bucket、跨 LLC placement 或 burst 超出
`Bmax` 都按 K0 处理。涉及 authority topology 时应在同一 seed block counterbalance
A=original K0、B=process-authority K0、C=process-authority+K1，而不是只比较 C/A。另用固定
绝对 schedule 的 open-loop all-wrong runner 作为 direct no-interference supplemental。每个
repetition 是独立统计单元；logical benefit 用 repeat-level one-sided 95% / equivalent two-sided
90% CI 判定。

“matched A/A”要求 A/A 与正 K artifact 使用完全相同的 runner、policy、broker、sidecar source
hash，相同 Python/runtime 配置、CPU/cpuset、payload、seed block 和 concurrency，并在同一个
冻结测量窗口内交错执行。只要代码在两次 run 之间改变，旧 A/A 只能解释历史噪声，不能从正 K
点差中作差后称为正式校准；该 cell 必须重跑同源 A/A。A/A 自身未通过 equivalence/resolution
gate 时，A/B 无论方向多好都只能标为 inconclusive。

每格至少报告：

- selected、started、exact/visible hits、coverage、precision；
- physical call amplification 和 wrong work；
- agent-visible logical latency 与 global logical wall；
- authority scheduled latency、first-run lag、broker wait 和 authority wall；
- admission/retire/confirmation lateness；
- `K_cap/K_safe/K_eff`、coordination-cost UCB 与 burst lookup 命中/abstain 原因；
- worker/tool cap、CPU affinity、LLC/interference-domain placement、scheduler policy、transport、
  raw-prefetch quiet gap、latch、zero-target、join-before-close 与 drain invariants；
- matched K=0/K=0 A/A noise calibration。

### 6.1 旧 sealed-pull 路径 smoke（只验证 transport 机制）

pre-silent revision 的 R1 artifact：
`results/pattern_v2_direct_sealed_pull_k4_c1_c16_smoke_r1/`，canonical payload SHA256 为
`8c7213d36e6a7c1b7b408e38e0d7a5d47b1da538ccb74039dd1e76f8af1a063b`。它刻意使用
`K=4, coordination_cost=1 ms` 扩大 selected/hit 数，使稀有 race 和 transport invariant 在一次
smoke 中可见；这不是正式 policy 推荐、不是 cost 校准，也不能用于性能推断。

- observed C1：`selected/started/visible-hit = 314/313/49`；
- observed C16：`100/100/15`；
- 所有 visible hit 的 child claim packet 为 `0`，`not-ready=0`；
- prefetch epoch 均在 guard 内 sealed，authority 完成前 bridge 始终未启动；
- 每次 claim 前 authority backend 已 arm，violations 为 `0`；
- all-wrong C1/C16 的 claim attempts/packets 均为 `0`；
- inner result cap 为 `4096 B`，所有 safety invariants 通过。

R1 点估计只说明完整路径能运行且不变量可观测。它既没有 paired-repeat inference，也没有同源
A/A resolution，因此不能据 C1/C16 的 latency 或 wall 点差宣称 no-regression 或 speedup。该
revision 仍在 parent cutoff 观察 prefetch，并已被 silent raw prefetch 取代；不能把它的 transport
通过误写成最新版 authority-path 证据。

### 6.2 Silent raw prefetch 与 interference-budgeted Top-K

旧 C16/R32、`K=1` guard-sealed revision 的 all-wrong authority 点差为约
`+0.251 ms/target`，而对应 A/A 点差约 `-0.036 ms/target`。二者方向相反，说明旧路径的正点差
不能只归为宿主共同漂移；它促成了“prefetch 绝不唤醒 parent cutoff”的 silent raw Future
修改。R32 只增加了对旧机制问题的置信度，不应被混入新版效果估计。

current silent 路径的 C1/K4 R1 mechanism smoke 位于
`results/pattern_v2_silent_k4_guard15_c1_cpu8_12_smoke_r1/`：observed 有 `314` starts、`47`
visible hits，authority 点差 `+0.1396 ms/target`，logical benefit `+3.9767 ms/target`；all-wrong
authority 点差 `+0.05385 ms/target`。报告的 treatment wall regression 分别为
`-0.0925%/-0.00227%`（observed logical wall speedup `+0.0955%`）。raw prefetch p95 为
`0.234 ms`，最小 realized quiet gap 为 `12.43 ms`，late/error/bad-claim 均为 0；off-parent-loop、
sidecar-affinity、post-arm poll、latch 与 join-before-close 等 safety invariants 全部通过。这仍是
**R1 insufficient**：大 logical point benefit 只说明 C1 命中路径仍有潜力，不能宣称统计 speedup
或 no-regression。

在 C16/R8 使用 `coordination_cost=5.2 ms`、`K_cap=1` 的 current silent 路径中，1880 个真实
targets 只启动 22 个 speculative calls，physical amplification 为 `1.0117x`；all-wrong
authority 点差降到 `+0.020 ms/target`。代价是 predicted net value 很少，observed logical benefit
没有显著性。这个 cell 很好地展示了双预算的用途：容量允许 K1 不等于每个 epoch 都应启动 K1，
net-value gate 会主动 abstain；但 R8 CI 仍是 inconclusive，不能把 `+0.020` 的点估计描述为
“已经 no-harm”。artifact 为
`results/pattern_v2_silent_net52_k1_guard15_c16_cpu8_12_r8/`。

C64 的同步 burst 超过已校准 `Bmax`，因此结构性 `K_eff=0`、不启动 sidecar。该 cell 只验证
fail-closed 行为；任何 treatment/baseline wall 点差都是 K0 宿主/次序噪声，不能当成 speculative
收益或损失。这也是 `K_safe` 必须按 concurrency/burst lookup、不能从 C16 外推到 C64 的原因。

安全验收是字典序的：容量证书失败时 `K_cap=0`；matched all-wrong/A/A UCB 缺失或未通过时
`K_safe=0`；二者通过后才以 global net value 选择不超过 `K_eff` 的候选。只有 no-regression
通过后才评估 observed benefit。`inconclusive`——包括上面的 R8——不能当作安全通过。线上若
authority event-loop lag、远端 rate-limit、raw prefetch late/error 或证书状态异常，当前 claim
fail-open 且 replay latch 在下一安全点令 `K=0`；如何 cooldown/reopen 不属于本文最小实现。

## 7. 实现与当前结论

- sidecar：`paste_repro/speculation_sidecar.py`
- process authority lane：`paste_repro/authority_process_lane.py`
- observed/closed-loop runner：`scripts/run_pattern_v2_sidecar_load.py`
- fixed-arrival negative-control runner：
  `scripts/run_pattern_v2_sidecar_open_loop_no_interference.py`
- tests：`tests/test_authority_process_lane.py`、`tests/test_speculation_sidecar.py`、
  `tests/test_pattern_v2_sidecar_load.py`、
  `tests/test_pattern_v2_sidecar_open_loop_no_interference.py`
- 综合结论：`results/pattern_v2_sidecar_scheduler_analysis/REPORT.md`

当前数据仍没有找到一个正 `K` 配置同时**统计通过**严格 no-worse 与显著 observed benefit，
但设计已经收敛到两个足够简洁的部件。

第一，执行路径使用 unchanged direct authority 与 silent raw prefetch。历史 negative controls
说明这是必要的：C16/R16 把 authority 迁到 child 后，child scheduled latency 虽改善
`0.488 ms/target`，parent-observed 却增加 `0.158 ms/target`，authority wall 明确回归
`1.303%`；batched return 更差，约 `2.493%`。eager staging 又证明，删除 exact-claim
round-trip 能降低 hit-conditioned delivery 税（约 `+0.820 -> +0.290 ms/target`），但持续搬运
wrong result 会反向占用 parent。silent pull 保留 parent-local hit 的优点，同时把 wrong-result
decode 限定到 sidecar CPU 上的 guard worker，并彻底删除 cutoff asyncio wakeup。

第二，调度使用 interference-budgeted global Top-K。独立资源只授予 `K_cap`；同 source、同
concurrency、同 LLC placement 的 all-wrong/A/A UCB 决定 `K_safe` 与 coordination cost。C16
`cost=5.2 ms` 的 R8 只启动 `22/1880` 个 extra calls、amplification `1.0117x`，all-wrong 点差降至
`+0.020 ms/target`，却没有显著 logical benefit。这不是失败的 predictor，而是说明轻量 tool 的
安全/收益可行区可能很窄：更保守的 cost 会降低干扰，也会让可测收益消失。论文应诚实报告这个
Pareto trade-off，不用复杂在线 controller 掩盖它。

旧 C16/R32 K1 all-wrong `+0.251 ms/target` 与 A/A `-0.036 ms/target` 支持继续删除
treatment-only wakeup；current C1/K4 R1 则验证了 raw Future、sidecar affinity、quiet gap、
post-arm poll、zero-target/latch 和 join-before-close 不变量，并保留了 `+3.9767 ms/target` 的
logical point benefit。但 R1 与 R8 都不足以作最终性能 claim；下一轮正式结果必须在冻结 source
下先通过 matched A/A resolution，再用预注册 R（建议至少 32）验证 all-wrong UCB，最后才看
observed benefit。

实验宿主不独占仍是主要外部限制：affinity 不能阻止其他进程使用同核，也不隔离 LLC、内存或
功耗域。因此 paper 的严格默认必须 fail-closed。default-no-certificate 在 `C=1/16/64` 均应为
`selected=started=0`、`sidecar_activated=false`、physical amplification `=1.0`；C64 burst 超过
校准范围时也结构性 K0。若论文另报正预算，应明确其 `K_cap/K_safe`、placement 与校准范围，
不能把 `inconclusive` 写成 no-harm，也不能把 R1/R8 方向好看的点估计写成 speedup。

当前 runner 的 predictor/scoring 在 timed replay 前预计算，因此实验只隔离和测量 online
scheduling/control path；不能用该 wall time 声称离线特征生成也具有零开销。

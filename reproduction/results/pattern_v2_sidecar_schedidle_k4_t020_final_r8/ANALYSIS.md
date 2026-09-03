# 多 Session speculative execution：瓶颈、设计与最终实验

## 结论

原系统的主要问题不是 Pattern-v2 的识别率，而是 speculative work 与 authority
共享了不可抢占的 tool capacity、event loop/GIL、dispatcher/cleanup，以及逐候选 IPC
热路径。资源总量看似充裕时，这些控制面串行点仍会把错误投机的微小管理开销按 decision
batch 累积，并在高并发 authority queue 中放大。

最终实现把安全性和收益做成字典序目标：先给 authority 保留完整的 demand-only
资源包络，再在独立 sidecar 中按全局期望收益选择候选。配置为全局 `K=4`、概率阈值
`0.20`。在 observed nested OOF 上，C=1/16/64 的 agent-visible tool latency 分别减少
`0.307 / 0.823 / 2.347 ms/target`；三个 repeat-level 90% CI 的下界都大于零，因此都达到
`improvement`。这是 predictor/selection 已在 timed wall 前预计算条件下的 tool-call latency
收益，不是包含 predictor CPU 的端到端净收益。all-wrong 下没有一格确认 authority
regression，但三格也都没有通过 equivalence gate，正式结论均为 `inconclusive`。

不能把结果扩大为“实验已经证明逐请求 wall-clock 严格零回归”。0.1 ms/0.1% 的
equivalence margin 小于当前 asyncio/VM 环境的 R=8 分辨率，因此 all-wrong 的统计结论
仍是 `inconclusive`。这里能严格审计的是资源和控制路径不变量；延迟实验只能说没有确认
practical regression，不能证明零回归。

## 根因

1. 共享 broker 的 authority-first 只对仍在 queue 中的 work 有效。已经运行且不可抢占的
   wrong speculation 仍会占用 baseline tool slot。
2. 即使增加 worker，shared event loop、GIL、全局 lock、dispatcher wakeup 和 cleanup scan
   仍是同一个串行资源。旧实验在 C=64 每次只启动约 9 个 prediction，却会产生数百次
   cleanup/dispatcher 操作。
3. 每个候选一次 submit、每个 miss 一次 tombstone，会把 pickle、锁和 syscall 放进
   authority confirmation 附近的热路径。
4. `asyncio.sleep(lead - admission)` 经 epoll 毫秒取整后，会把本应被 lead 吸收的几十微秒
   重新加到每个 batch；这不是 tool capacity 不足，却会形成累计 wall regression。
5. 常驻 parent result bridge 即使没有 hit 也留下一个额外线程/GIL 参与者。对轻量工具而言，
   这个控制面成本可能比 tool 本身的 CPU 成本更明显。

## 最小设计

### 1. Authority 资源证书

Authority 保持原始 demand-only broker：4 workers、`visit` capacity 2。Speculation 使用
独立 fork process、独立 tool/executor capacity、独立 CPU mask；本机 authority 固定在
CPU 0，sidecar 固定在 CPU 95。真实部署还必须给 sidecar 独立的 connection pool、rate
token 和 backend quota；只隔离本地 worker 不能证明下游服务不被征用。

sidecar child 使用 Linux `SCHED_IDLE`（snapshot 中实际 policy 为 5）。独立 CPU 空闲时
它正常执行；一旦物理调度器出现竞争，它自动让位给普通 authority work。资源证书无效、
状态过期或 `K=0` 时必须 fail closed 到 demand-only。

本次主机的 sysfs topology 显示 CPU 0 与 95 位于不同 core/socket/NUMA node，且均无 SMT
sibling；这比只看 logical affinity 更强，但 runner 在其他主机上选择首/末 CPU 并不会自动
产生同样的物理证书，复现时应重新核对 topology。

### 2. 全局收益调度

所有 active sessions 共享同一个 `K`，而不是每个 session 各拿 K 个 slot。候选先过滤
`p_exact < 0.20`，再按校准概率、可重叠 lead、service time 形成 expected saved latency / 
service priority。每个 `(session, decision)` 最多一个候选，所有 session 进入同一个全局
benefit heap；queued-only replacement 允许后来更高收益的候选替换低收益项，running work
不伪装成可抢占。

离线 sweep 显示 K=4 是收益与放大率的 knee；本实验的 physical call amplification 在
C=1/16/64 为 `1.190x / 1.188x / 1.112x`，没有随 session 数按 `N*K` 膨胀。

### 3. Authority 热路径 O(1) 且 fail-open

- 一个 decision epoch 的全部候选只发一个 bounded `SOCK_SEQPACKET` datagram；发送锁忙、
  packet 过大或 socket EAGAIN 时整批返回 miss，parent registry 原子回滚。
- finite `start_deadline` 同时建立 lease，默认 claim grace 10 ms。未 claim 候选在 child
  自行 tombstone，parent 在下一次 submit 懒回收；authority miss 不发送 cleanup packet。
- result bridge 只在首次 exact claim 时懒启动。all-wrong timed path 没有 parent bridge、
  result event、terminal event 或 tombstone packet。
- authority call 总是先提交并获得一次 event-loop 调度，再检查 exact handle；authority
  从不等待 sidecar admission、cleanup 或 drain。Spec failure 只回退到已经存在的 authority
  task，不创建第三次调用。

成功 hit 仍不可避免地需要一次 parent 通知：首次 claim 会创建 lazy bridge，结果要跨 IPC
并完成 future。当前设计把这项成本从所有 miss 移到少量 hit path，而不是声称 parent 与
sidecar 完全 wait-free。每批的参数 canonicalization、pickle 和 registry 也仍是同步 bounded
工作；`K=4` 使它不随 session 数线性增长。

### 4. 独立 confirmation clock

Synthetic agent confirmation 使用 baseline/treatment 相同的绝对 monotonic deadline，由
一个 sleeping timer thread 唤醒。它避免 epoll 对 `9.x ms` timeout 的向上取整把 admission
耗时误算成 agent 变慢。Admission 仍计入总 wall；若它真的超过 lead，会记录 deadline
overrun 并由 child fail closed。

## 主实验结果

每格 8 次 paired AB/BA；235 authority targets/repetition。sample summaries、repeat vectors、
snapshot 和 source hashes 在 `metrics.json`；per-target raw rows 没有保存在 aggregate artifact。

| Scenario | C | Visible hits | Started precision | Logical benefit ms/target (90% CI) | Logical wall | Authority wall point | Call amp. |
|---|---:|---:|---:|---:|---:|---:|---:|
| observed | 1 | 75 | 20.9% | +0.307 `[+0.172,+0.443]` | +0.96% | -0.95% | 1.190x |
| observed | 16 | 80 | 22.8% | +0.823 `[+0.535,+1.111]` | -0.22% | +0.24% | 1.187x |
| observed | 64 | 55 | 25.7% | +2.347 `[+1.328,+3.366]` | +0.06% | -0.01% | 1.114x |
| all-wrong | 1 | 0 | 0% | -0.043 (inconclusive) | -0.10% | +0.10% | 1.190x |
| all-wrong | 16 | 0 | 0% | -0.219 (inconclusive) | -0.13% | +0.13% | 1.188x |
| all-wrong | 64 | 0 | 0% | +0.350 (inconclusive) | -0.02% | +0.02% | 1.112x |

Observed 三格的平均 logical benefit 都是 repeat-level 显著正收益。C=16/64 的 global
logical wall 几乎不变，是因为当前 closed-loop runner 对同一 batch 做 barrier：一个
session 的 hit 不能让整个 batch 越过仍在等待的其他 sessions。它会低估独立 agent
session 的 task-level throughput benefit，因此高并发下主要报告平均 agent-visible tool
latency，而不把单个 global makespan 当作唯一收益指标。

主实验把 immutable predictor/selection plan 在 parent CPU 上、`wall_started` 前预计算。
冻结统计约为 feature `148.589 ms/run`、lookup `5.804 ms/run`，合计约
`0.657 ms/authority target`；它没有计入上表 logical benefit。特别是 C=1 的
`0.307 ms/target` 不应外推为包含同步 predictor 的端到端净收益。

## Observed authority regression：已确认现象，机制尚未识别

Observed C=16/64 的 scheduled-to-terminal authority latency 分别增加约 0.60/1.31 ms，
90% CI 为 `[+0.269,+0.924] / [+0.698,+1.927] ms`，其 lower bound 均高于 0.10 ms margin；
因此旧 runner 的 formal gate 正确标记为 `regression`，不能用机制解释将其降格。

一种相容机制是 exact hit 让 closed-loop session 更早进入下一步，使下一批 real authority
call 与尚未 drain 的 shadow backup 重叠。但 observed 每格也有 55--80 次 claim/result，
首次命中会启动 parent bridge，因此 hit-path IPC/callback/GIL 也可能贡献回归。artifact 没有
保存 batch arrival timestamp、arrival shift、命中时 outstanding backups 或 per-target absolute
completion，不能区分这两种机制，也不能声称 absolute completion 已改善。

all-wrong 没有 hit，因此是 same fixed workload、same batch sequence、no-hit 的 closed-loop
negative control；它仍在每个 batch gather 后启动下一批，不是固定 timestamp arrival replay。
3 个 cells、24 个 all-wrong treatment runs 中：

- `transport_claims = 0`
- `transport_results = 0`
- `transport_tombstone_packets = 0`
- `transport_terminal = 0`
- timed authority 前 `bridge_started = false`
- 所有 worker/tool cap、authority attempts/commits/state、CPU affinity、SCHED_IDLE、drain
  invariants 通过。

这些不变量证明 all-wrong miss path 没有 result/cleanup bridge traffic，但不能排除 observed
hit path 的 parent 开销，也不能替代真正预生成绝对 schedule 的 open-loop no-interference
实验。

## A/A 噪声与 no-regression 边界

同 runner 立即执行但未保存 raw artifact 的 K=0/K=0 R8 A/A 描述性结果为：

| C | Authority ms/target 90% CI | Authority wall 90% CI |
|---:|---:|---:|
| 1 | `[-0.127,+0.142]` | `[-0.135%,+0.406%]` |
| 16 | `[-0.327,+0.419]` | `[-0.270%,+0.244%]` |
| 64 | `[-1.571,+0.695]` | `[-0.443%,+0.319%]` |

all-wrong 的 authority wall 点差 `+0.100% / +0.133% / +0.024%` 与这次单独 A/A run 的
量级相近。这只是描述性比较：一个 treatment point 落在另一个实验的 mean CI 内不是正式
等价或差异检验，而且 A/A 缺少 raw provenance。开发阶段 SCHED_IDLE 前后点估计方向有变化，
也只能说与 scheduling variability 相容。正式状态保持 `inconclusive`。

## 论文中可以和不可以声称什么

可以声称：

- modeled worker/tool/CPU execution capacity 隔离，错误投机不会占 baseline broker slot；
- miss path 没有 authority-side cleanup/bridge traffic，transport saturation fail-open；
- observed 平均 logical latency 在三个代表性并发度都有显著改善；
- all-wrong 没有确认 practical authority regression，但三格仍为 inconclusive。

不可以声称：

- R8 已经证明逐请求 wall-clock 在 0.1 ms/0.1% 内统计等价；
- 本地 CPU/worker 隔离自动等于远端 API quota 隔离；
- closed-loop 中每个后续 authority call 的相对 sojourn latency 必须下降；
- global batch makespan 等价于每个独立 agent session 的 completion time。

## 推荐默认策略

论文主配置可用 `threshold=0.20, K=4`。部署顺序保持简单：

1. 没有完整独立资源证书时 `K=0`。
2. 有证书时按全局 benefit Top-K 分配，不做 per-session Top-K。
3. sidecar 使用独立 process/CPU/backend quota 和低 OS priority；所有热路径操作 bounded、
   non-blocking、fail-open。
4. authority event-loop lag、远端 rate-limit 或证书状态异常时立即打开 circuit breaker，
   cooldown 期间退化到 demand-only。

这套设计没有增加复杂的在线优化器；核心只有资源证书、全局 Top-K、lease cleanup 和 lazy
claim bridge，足以解决多 session 下真正的资源协调问题。

## 仍然存在但不应过度设计的瓶颈

- 已运行的错误投机不可抢占；长尾调用会占满 K，降低后续投机收益，但在资源证书成立时不
  阻塞 authority。研究版保留 queued-only replacement，不引入复杂 preemption。
- sidecar 的 K 个 async workers 都绑定在一个 CPU 上，适合本实验的轻量 I/O service；对
  CPU-bound tool，必须重新分配 CPU certificate，不能把 K 直接解释成 K 个 CPU cores。
- predictor/Top-K、进程启动和 immutable selection plan 在 timed replay 前完成，所以结论
  依赖长期驻留、异步预测；不外推到每次请求都冷启动的系统。
- synthetic executor 使用 `asyncio.sleep`，不能代表大 payload、重序列化、共享网络或真实
  backend 限流。生产 exact key 还必须包含 tenant、credential、tool/version、数据 snapshot
  与 authorization context；空 `context_token` 只适合此受控 replay。

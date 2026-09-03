# 固定 PASTE setup 下的 Murakkab M-only 实验

实验日期：2026-08-31

## 结论

在固定模型、固定四卡部署、固定工作流和保留 ResNet 共载的条件下，Murakkab
的配置空间退化为一个候选。因此本实验中的 M cell 是一个
**A-equivalent 的 constrained Murakkab-style emulation**：计时外做一次 typed-DAG
校验和唯一候选选择，计时内使用 native FCFS 和 demand-only 工具调用。

三次 clean repetition 的时间加权吞吐为 **0.296089 tasks/s（17.7653
tasks/min）**；按 80 个独立 source 先跨 repetition 求均值后，E2E mean/p50/p95/p99
为 **185.178/179.001/256.609/263.417 s**；每次完成 80 个任务的 makespan
均值为 **270.189 s**。240/240 个任务成功，720/720 次 LLM 调用成功，480/480
次物理工具请求完成，观察到 0 retry、0 speculation。

这组结果没有显示 Murakkab 在该固定 setup 下获得了额外调度收益。它仍能达到
不错的绝对性能，主要来自 vLLM 原生 continuous batching、prefix cache、80 个并发
任务，以及按依赖就绪执行 DAG；这些不是 Murakkab 在本实验中额外创造的优化。

## 完全固定的实验条件

- 模型：`Alibaba-NLP/Tongyi-DeepResearch-30B-A3B`，revision
  `4b0ac5767427a55d08a254f0367e2934976598e0`。
- 服务：vLLM 0.10.1、BF16、TP=4、单 replica，GPU 4--7 上的
  4×A100-SXM4-40GB；16K context、`gpu-memory-utilization=0.86`、
  `max-num-batched-tokens=2048`、`max-num-seqs=96`、原生 prefix cache 开启。
- 工作负载：同一份 frozen formal-v9 的 80 个 source，offered concurrency=80，
  每任务 10K private padding、固定 3 次 LLM 调用、final completion 192 tokens。
- 工作流：`LLM -> search -> LLM -> visit -> LLM`；真实 Bing/Jina transport；
  4 个物理工具 worker、search capacity=3、visit capacity=2、visit start interval=2.5 s。
- Murakkab 可选 workflow/model/hardware/parallelism/replica/SLO tier 均只有 1 个；
  禁止换模型、换卡、改变 TP/replica、autoscaling、workflow pruning、质量降级、
  多 workflow multiplexing、合成 SLO mix 和 speculation。
- ResNet PID 2298 没有被关闭。每次实验在 GPU 4--7 的 before/after 证据中均验证：
  同一 PID、进程 start time、boot ID、executable、cwd、argv、脚本 SHA，以及每卡
  恰好一条该 ResNet application；不允许额外 selected-GPU application。

ResNet 证据只证明端点的进程和代码身份相同。由于没有加入连续的共载强度监控，
不能声称不同 repetition 或历史 PASTE 实验中的 GPU utilization、功率和训练强度
严格相同。

## 实验方法

先做 2-task smoke（不计入性能结果），然后计划取得 3 次 clean repetition。每次均
启动 fresh vLLM server、fresh broker 和空 result cache，并完整执行 80 个任务。
吞吐分母是 runner experiment start（包含 metrics/client setup）到最后一个任务完成；
另行保留 release-window throughput。统计单位是 80 个 source，而不是把三次重复的
240 条观测错误地当作 240 个独立样本。

原始第 2 次完成后，硬件 sidecar 和独立进程时间戳证明另一会话的 vLLM 在 GPU 0--3
上与计时窗口重叠：API overlap 104.556 s，worker overlap 70.836 s。即使它未占用
GPU 4--7，也无法排除同主机 CPU、内存、I/O、互联、功率或温度干扰。因此该次按
时间戳规则做 post-hoc operational exclusion，完整保留为 supplementary；排除规则
没有使用 latency/throughput 阈值，但性能值在决定前已被查看，这一点也已披露。
随后补跑 clean replacement，最终主统计仍包含 3 次 clean repetition，总共披露
4 次完成的 performance attempt。

第 1 次实验后，原仓库的 `live_executor.py` 被另一会话修改。为了不覆盖用户改动、
也不让 repetition 间代码漂移，第 2 次 replacement 和第 3 次在 commit
`83e018557566c78e5d499dae5bfd1a877b66eef2` 的 frozen worktree 中运行；绑定的
`live_executor.py` SHA 与第 1 次完全相同。所有 run 同时绑定配置、runner、runtime、
workload、DAG registry、run plan、模型 revision 和硬件 sidecar 的 SHA。

## M-only 实测结果

| Clean repetition | 吞吐（tasks/min） | 80-task makespan（s） | E2E mean（s） | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1 | 16.0606 | 298.868 | 232.986 | 224.604 | 284.411 | 290.087 | 290.967 |
| r2 clean replacement | 18.5724 | 258.448 | 163.060 | 158.941 | 245.931 | 252.945 | 253.004 |
| r3 | 18.9535 | 253.251 | 159.487 | 153.462 | 241.189 | 247.219 | 248.634 |
| 三次 aggregate | **17.7653**（时间加权） | **270.189**（均值） | **185.178** | **179.001** | **256.609** | **263.417** | **264.202** |

最后一行的 latency 是“每个 source 跨三次求均值后”的 80-source 分布；max 也按这一
定义。它与直接 pooling 240 条 task observation 的 p50/max 不同。

三次聚合的每任务平均时间分解为：LLM 108.596 s、search exposed wait 0.183 s、
visit exposed wait 76.228 s、未归因 residual 0.171 s。

### 为什么 r1 明显更慢

r1 不能作为异常值删除，因为它通过了与 r2/r3 相同的代码、输入、GPU 和 ResNet
身份校验，也没有 HTTP retry。主要差异出现在 LLM serving：

| Repetition | 每任务 LLM 时间（s） | vLLM 累计 queue time（s） | vLLM 累计 inference time（s） | visit exposed wait（s） |
|---|---:|---:|---:|---:|
| r1 | 167.668 | 9526.369 | 3361.596 | 64.863 |
| r2 | 77.868 | 4798.268 | 1060.866 | 84.884 |
| r3 | 80.252 | 4969.936 | 1130.549 | 78.938 |

三次 prompt-token 总量范围只有约 0.031%，completion-token 范围约 0.098%，所以
不是输出长度漂移。r1 的 LLM 更慢同时把 visit 到达摊开，反而降低了 tool queue
压力。这说明系统存在明显的 block-level serving/host-load 波动；现有端点证据不足以
确定其单一原因，不能事后只保留较快的两次。

## 与历史 PASTE 的关系

| 数据 | LLM scheduler | 工具策略 | Mean E2E（s） | Mean makespan（s） |
|---|---|---|---:|---:|
| 当前 M（3 clean reps） | native FCFS | demand only | 185.178 | 270.189 |
| 历史 formal-v9 A | native FCFS | demand only | 161.8274 | 未在摘要中报告 |
| 历史 formal-v9 E | Joint | demand only | 120.7134 | 218.5661 |
| 历史 formal-v9 F | Joint | visit speculation | 115.8396 | 212.9858 |

当前 M 的 timed semantics 与历史 A 等价。r2/r3 的 mean E2E 相对历史 A 分别为
+0.76% 和 -1.45%，可以作为实现 sanity check；但三次 M 的均值因 r1 升到
185.178 s，比历史 A 高 14.43%。这恰好说明跨时段波动不能忽略。

历史 formal-v9 在它自己的同期、配对矩阵内部报告了 A->E 降低 25.4061%、A->F
降低 28.4178%、E->F 降低 4.0375%（低于预注册的 5% promotion gate）。这些结果
说明 PASTE 的优势来自动态队列下的 Joint physical-KV 调度和 speculative tool
overlap；它们不是 Murakkab 在当前单候选配置空间中能够使用的机制。

不能把当前 M 与历史 A/E/F 的差直接称为 Murakkab-vs-PASTE speedup：两者不是同期
随机化或配对实验，使用的是已经观察过的 workload，Bing/Jina、host load、ResNet
强度和 GPU 状态均可能随时间变化。历史数值在这里只是描述性背景。

## 之前 setup 的错误

之前的概念性 setup 把 Murakkab 论文中的模型、异构 GPU、parallelism、replica、
autoscaling、workflow/quality tier 和 SLO 选择空间带入了比较，而本项目的 PASTE
模型与部署已经固定。这会给 Murakkab 额外的 treatment freedom，不能回答同 setup
下谁的 runtime policy 更好。

之前真正实现并产出 `28.86%` 数字的实验还有更根本的问题：它是离线
adaptive-`top_k` PASTE trace replay，优化变量是 PASTE speculation width；不同合成
SLO tier 各自 replay 后再按权重组合。其 latency 是 trace counterfactual，资源指标是
admitted-request proxy，不是 live E2E、GPU、energy 或 cost。它可以保留为 PASTE
配置宽度 ablation，但不能称为 Murakkab system performance，也不能与现在的 M-only
live result 混用。

## 证据边界与文件

- 本次没有使用或复现官方 Murakkab runtime，而是按论文思想实现的受约束 emulation。
- singleton planner/type checking 在计时外，未测 control-plane overhead。
- 只校验执行成功、调用次数和提交契约，没有独立评测答案语义质量。
- 所有 cell 都固定四张 GPU；没有测量 energy、cloud cost，也不能声称节省 GPU 数。
- 主聚合和完整 provenance：[`m_fixed_v9_clean_aggregate.json`](m_fixed_v9_clean_aggregate.json)。
- 自动生成的工程报告：[`M_FIXED_V9_CLEAN_ENGINEERING_REPORT.md`](M_FIXED_V9_CLEAN_ENGINEERING_REPORT.md)。
- 被排除 r2 的时间戳证据：[`rep2_host_coload_observation.json`](rep2_host_coload_observation.json)。
- 固定执行配置：[`../../configs/murakkab_fixed_v9_m_only.json`](../../configs/murakkab_fixed_v9_m_only.json)。
- 历史 PASTE 报告：[`../live_joint/PREFIX_AND_LIVE_CLOSED_LOOP_FINAL_REPORT.md`](../live_joint/PREFIX_AND_LIVE_CLOSED_LOOP_FINAL_REPORT.md)。
- 已废弃的旧 system-comparison 报告会继续保留并标注 superseded：[`REPORT.md`](REPORT.md)。

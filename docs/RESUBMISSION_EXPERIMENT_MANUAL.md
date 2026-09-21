> **当前 policy 执行规范已升级为 v11：** [完整、可冻结的实验 setup 手册](POLICY_EXPERIMENT_SETUP_V11.md)。下文是历史 v9/v10 计划；新实验不得继续使用旧的负优先级、TTL60、combined pending128 或 FIFO80-only 消融配置。

> **2026-09-20 用户指令更新（v9）**：所有 policy 相关消融统一使用 trace duration 异步模拟工具执行，保留真实 LLM、原有并发/预算和其他 setup；不准备真实工具环境、不跑 smoke。未来时长仅供 executor 使用，不能传给 scheduler。正确性与实际 CPU/network 成本沿用单独标明的真实工具 audit/实验，模拟结果不代替这两类证据。下文要求 policy 实验必须跑真实工具及 smoke 的部分由本条覆盖。具体语义与状态见 [TIMED_POLICY.md](../reproduction/results/resubmission/TIMED_POLICY.md)。

**PASTE 重投稿补充实现与实验手册**

日期：2026-09-20。适用项目：DeepResearch `/home/aiscuser/PASTE-Qwen-DR`；Coding `/home/aiscuser/gemini-cli-PASTE`。

目标只有三个：确认加速不改变 authoritative behavior；量化 speculative extra work 换来的 E2E 收益；识别 tool information 对排序的独立价值。沿用已有最佳结果的 setup，优先补实验和记账，只有发现实际缺口才补机制。本文件是待执行计划，未实施项不代表已经完成。

**1. 从哪套已有结果继续**

以两个项目明确指定为 result of record 的 **unified-workset-v1** 为主要配置起点，不把不同实验中的最高百分比拼成一套配置。

| 项目 | 已有结果 | 沿用的 setup 与入口 |
|---|---|---|
| DeepResearch | Mean E2E `514.967 → 310.755 s`，降低 **39.66%** | Tongyi-DeepResearch-30B-A3B；80 sessions；FULL working set 40、decode target/max 32/33；`run_dr_trace_hybrid_pair.py` |
| Coding | Mean E2E `104.069 → 49.462 s`，降低 **52.47%** | Qwen3-Coder-30B-A3B-Instruct；80 sessions（10 个 SWE task × 8 replicas）；FULL working set 24、decode target/max 24/25；`run_swe_trace_live_pair.py` |

两边均继承各自 `reproduction/configs/unified_workset_v1.env.example` 的完整配置：模型 revision、TP=4、BF16、`max_num_seqs=48`、prefix caching、physical-KV target 0.93、gain weight 0.25，以及原有 workload、到达序列和 tool capacity 16。外部 admission coalescing 为 3.2 s。具体命令直接沿用下列文档，新实验另写输出目录与 manifest。

- [DeepResearch setup、结果与命令](../reproduction/UNIFIED_WORKSET_V1.md)。
- [Coding setup、结果与命令](/home/aiscuser/gemini-cli-PASTE/reproduction/UNIFIED_WORKSET_V1.md)；[现存结果](/home/aiscuser/gemini-cli-PASTE/reproduction/analysis/unified_workset_v1/gemini/REPORT.md)。
- Coding unified 实验也使用 **Qwen 项目的 scheduler hook 和 start/stop 脚本**，后续修改共享实现，避免改到 Gemini 的旧 hook。

这两个数字属于 **live LLM + trace-conditioned 工具收益**；包含完整 trace 提供的剩余工作量/收益标签。Coding 工具侧还是 retrospective offline projection，未计实际 readiness、wrong work 和 speculative contention。因此沿用的是配置和实验基础，新增加的真实成本、正确性与在线收益必须重新测，不能直接附在旧数字后面。

Qwen 另有真实 Bing/Jina 四格参照：[formal-v9](../reproduction/results/live_joint/PREFIX_AND_LIVE_CLOSED_LOOP_FINAL_REPORT.md)。完整系统 A→F 降低 **28.42%**，Joint 内 speculation E→F 降低 **4.04%**；后者未过原先 5% promotion 门。其 80 tasks、10k padding、`max_num_seqs=96`、4 tool workers、最多 2 speculative workers、visit interval 2.5 s 是独立 setup，不与 unified 混用。它采用 frozen/perfect URL，可验证执行闭环，不能单独回答错误预测的成本。

当前 checkout 缺少 Qwen unified 的原始 `plan.json` 和 `comparison.json`，39.66% 来自已有说明文档。正式开跑前恢复原始 artifacts 并核对 SHA；若无法恢复，按原协议重新生成并标为新基线，不能称为原结果的精确复现。Coding 的 plan/result 当前存在。

**2. Correctness：先做实验，按失败补实现**

已有基础：

- DeepResearch live 路径：`online_learned_agent.py → live_broker.py`；结果确认前不进入正式历史，按 session/tool/完整参数匹配，单次采用，过期和失败 fallback。本轮已运行 broker 的 37 个测试，全部通过；这不是完整 agent 等价实验。
- Coding CLI 路径：`speculativeToolRuntime.ts`、`coreToolScheduler`；有 session/mutation-epoch/参数匹配。可执行的本地 speculation 主要为 `read_file`，已有执行前后及 commit 时的 stat 检查，写操作使旧候选失效。不能据此声称任意 shell/edit 已具备 private-workspace publication。

额外实现：先加一个成对 audit 入口，复用实际 executor/broker；记录 authoritative 调用序列、交付结果、提交历史以及 Coding 的文件状态摘要。已有校验通过就不重写 isolation/validation/fallback。验证日志与最终性能实验使用同一条运行路径。

额外实验：固定 authoritative 调用和工具输入状态，比较 speculation off/on；覆盖有效 completed reuse、in-flight match、错参数、跨 session、过期、执行失败，以及 Coding 的文件/epoch 改变。检查错误候选没有进入历史或修改正式状态，失效候选确实执行普通路径，有效匹配的结果与正式状态一致。

Claim 限定在受支持工具及明确的输入有效性条件下。网页 audit 先用固定内容；Qwen live 目前的 TTL 不证明网页内容不变，若要复用动态网页需补明确有效性检查，否则 fallback。Coding 已有 HTTP freshness 机制可作为实现参考。stat/epoch 也不应写成对任意外部并发写入的完整 snapshot 保证。

正确性 audit 先独立完成；最终性能运行保留必需的隔离、校验和 fallback，并计入其开销。固定调用审计证明结果交付与状态保持，不替代 autonomous agent 的任务质量评价。

**3. Speculation：补账本与预算实验**

已有开关、top-k、worker/pending 限制，以及调用时间、命中/浪费、HTTP attempts 和部分字节记录。额外实现集中在：

- 按 `task_id + job_id` 串联 admission、start、complete、claim、discard、cleanup，汇总到 task E2E。
- 补实际 CPU core-seconds、HTTP 请求/传输字节、worker occupancy、结果保留 peak bytes 与 byte-seconds；失败、取消、重试和 task 结束后的清理也归入原 task。未测量项记为缺失，不能当成零；worker wall time 不能当成 CPU time。
- 同时报告 total resource cost 与相对 zero-speculation 的增量，避免把被采用的提前执行全部算成额外工作。GPU 成本先记同配置下的整轮占用，不伪造无法归属的 per-task GPU 时间。

实验矩阵：**zero + 低/原配置/高预算 × 低/中/高负载**。保留原 operating point；优先改变实际 speculative execution cap，固定 predictor、排序、总工具容量和 authoritative 预算；只有现有宽度限制使预算点无法区分时，再单独扫 top-k。预算设置与实际消耗都要记录。

每个点同时输出 mean/p95 E2E、完成率、total/extra cost、useful/unused work，画成本—E2E 曲线。主实验使用会产生真实错误候选的 predictor；perfect-URL 仅作参照。

**关键接线检查：** unified runner 当前的工具收益是投影；计数器加在投影上得不到真实成本。必须让新实验调用经 correctness audit 的实际工具路径。先小规模确认两项目的 runner 与 executor 能接通；若存在缺口，单独列出最小适配工作，不能预先承诺只是加日志。Coding 先覆盖现有可安全执行的工具，其余普通执行，不为复现旧投影增益扩展任意 shell/edit speculation。

**4. Co-scheduling：在最佳配置上只替换排序**

已有 unified 的 session working set、engine gate、physical-KV 保护与 tool-aware score。额外实现：将 admission/gate 与 ranker 独立配置，支持 **FCFS / length-aware / tool-aware**；复用共享 Qwen hook，同时检查两个 runner 的 pre-engine admission。

最小对照：

| 对照 | 固定条件 | 回答的问题 |
|---|---|---|
| Shared gate + 三种 ranker | 同 predictor、speculation budget、长度估计、KV/gate、coalescing、aging/rescue | Tool information 是否增加价值？ |
| Speculation + tuned fixed cap + FCFS vs full | 同工具路径、模型、负载、资源预算；cap 在独立 tune tasks 上选择 | 收益是否超出单纯限流？ |

先固定 pre-engine 的顺序，仅切换 engine ready-turn ranker；若还要归因 pre-engine ordering，再单独比较。不得同时改变两处排序后归因到某一处。检查 final/progress/remaining-call lanes：非 tool 项保持共同，length-aware 不接收 tool 特征，防止隐藏优先级混入比较。Shared gate 指同一规则和参数，不强求不同排序下实际 running 数逐时刻相同。

unified 的 exact remaining-work/retrospective-gain 标签可保留在明确标注的 trace-conditioned 对照；正式在线 claim 使用已有因果预测或当前已观测信号，不能读取未来 trace。不新增复杂 predictor，信号变更后的性能单独重测。

报告 mean/p95/p99 task E2E、pre-engine wait、in-engine wait、exposed tool wait 和 throughput，并在相近预测长度内比较。E2E 从外部 arrival 起算，必须包含 coalescing、admission、validation 和 publication；不能只从进入 vLLM 后计时。

**5. 执行顺序与完成标准**

1. 恢复并锁定两套最佳 setup 的 plan、配置、模型、代码和结果来源；旧结果作为历史参照。
2. 跑 correctness 成对 audit；仅修复实际失败，固定最终 speculation path。
3. 接通实际 executor 并补资源账本；同时拆分 gate/ranker，完成小规模 smoke。
4. 独立 tune 数据选择预算档与 fixed cap；冻结代码、配置和测试集后统一跑正式矩阵。不同矩阵只改变其目标变量。
5. 每个正式对照至少三个 fresh-server 配对 block，交替顺序；报告配对区间、尾延迟、失败/超时与资源账。Coding 的 8 个 replicas 用于制造负载，统计以原始 task 为单位，并保留 block 波动。

最终交付三份简短报告：`correctness_audit`（案例、采用/fallback、差异数）、`speculation_cost_e2e`（负载×预算曲线）、`coscheduling_ordering`（同 gate 排序与 tuned cap 对照），共同绑定冻结版本与 manifest。

不新增与这三个问题无关的系统功能。暂不实施的 lead-time scoring、通用 workspace publication 或复杂资源管理，在论文中同步收缩承诺；旧 replay/prototype 的性能不能替代最终实际路径的测量。

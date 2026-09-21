> **2026-09-20 修正：本文旧结果全部属于 oracle 条件诊断。** 旧入口向 scheduler 提供真实剩余调用、完成 token 和下一轮 prompt/finality；因此下文无论正收益还是负收益均不能推断在线设计的效果。“tool-aware 慢 2.2%”不表示在线 tool information 无价值。保留原始数据与历史分析，但暂停以此选择在线配置或作论文主结果；以修正后的 causal 实验为准。正确性和资源原始计数仍是该 oracle workload 的事实记录。

# 最小实验的 setup 选择（2026-09-20，探索阶段）

用户要求以已有证据寻找最有利的实际运行条件，并停止无差别全矩阵重复。原每项目 45 点计划已在当前点结束后撤下；完成的结果和负结果保留。所有新运行仍使用原入口，不新增 runner。

## 历史证据怎样用于选择

| 证据 | 最好/主要结果 | 本轮采用的部分 | 不可直接继承的部分 |
| --- | --- | --- | --- |
| DR unified-workset-v1 | 514.967 → 310.755 s，39.66% | 原模型、40/32 working set、固定 trace、共享工具容量16 | 原 plan 缺失；工具收益是离线投影，包含完整剩余工作标签 |
| Coding unified-workset-v1 | 104.069 → 49.462 s，52.47% | 原模型、24/24 working set、原 retained SWE traces、Top-3 起点 | 离线预测给予完整工具服务收益；包括本轮未提前执行的 test/env 工具 |
| Coding retrospective Top-3/5 | 工具服务降低97.68%/97.71% | Top-3，暂不扩宽 | 乐观全服务抵扣，不含真实 readiness、错误执行与竞争 |
| DR session-URL Top-10 | mean-flow 24.28%，earlier-decision hits159 | 同一 frozen predictor，检查跨轮候选保留 | 无限TTL、无内容过期、独立spec资源；不能用于真实HTTP的直接加速claim |
| DR formal-v9 real HTTP | 整体28.42%；Joint内spec4.04% | 足够LLM排队、可重叠工具工作、有限spec预算；execution-aware解释 | perfect URL、Bing/Jina API，不是当前无API的Search模拟条件 |

来源：`reproduction/UNIFIED_WORKSET_V1.md`、Coding同名文档及`analysis/unified_workset_v1/gemini/REPORT.md`、Coding `analysis/real_trace_hybrid_v3/source_policy/REPORT.md`、DR `results/pattern_v2_trace_all_visit_wall_cache_session_url_top10_c1_128_r32/REPORT.md`、`results/live_joint/PREFIX_AND_LIVE_CLOSED_LOOP_FINAL_REPORT.md`。

## 当前实际测量决定下一步

DR 原80条测试trace、高负载首block：

| budget | mean E2E(s) | p95(s) | HTTP请求 | 实际隐藏服务(s，总和) | useful / unused physical spec |
| --- | --- | --- | --- | --- | --- |
| 0 | 320.106 | 567.735 | 447 | 0 | 0 / 0 |
| 2 | 316.908 | 571.962 | 1073 | 2.743 | 1 / 569 |
| 4 | 313.952 | 563.851 | 1190 | 5.836 | 3 / 715 |

budget4有413次TTL过期；保留上限60s，而LLM请求p95约127s。部分origin明确freshness长于60s。允许候选保留600s只改变本地保留上限，正式采用仍受origin HTTP freshness和参数/session校验限制，绝不将600s当成网页有效性保证。大多数URL没有可用freshness，延长保留未必能带来收益，须实测。

零spec平均暴露tool等待20.500s，仅占320.106s E2E的6.40%。这是此测量中可直接隐藏的等待规模，不是考虑队列反馈后的全系统理论上界。仅凭工具侧提前执行，不能合理预期重现39.66%的历史整体数字。

同gate、budget4的首block：tool-aware313.952s，FCFS357.687s，length-aware307.258s；独立选择native cap20为390.034s。Tool-aware相对FCFS约12.23%、相对tuned cap约19.51%，但比length-aware慢约2.18%。这不支持tool信息的独立正收益claim。

Coding snapshot复位后的首个零spec点64/64完成，mean91.924s。344次repo_read的总暴露等待84.009s；所有工具总暴露等待1979.494s。当前可安全提前执行的read部分只占工具等待4.24%，平均每task1.313s。历史大幅工具收益主要包含test/env抵扣，本轮不能移用。按照零spec工具组成，requests与xarray的read占比分别34.79%和17.90%，可作为读取密集分组解释；不据此删除其他6个测试source，也不称作新的独立holdout。

## 已执行的最小调整

- DR原入口仅开放两个已有机制参数：执行候选Top-K与broker候选TTL。默认10/60保持旧行为。冻结为source_manifest_v5；broker/记账66 tests及23 subtests通过，未跑smoke。
- DR独立20条tune trace：zero；budget2/Top-10/TTL600；budget2/Top-3/TTL600。先看实际reuse、工具等待、成本与E2E，再决定保留哪个设置；不自动扩至budget8。
- Coding保留已批准2/8划分、全部8个测试source及每source8副本、现有Top-3实际read prefetch。先补budget4的三ranker及tuned cap对照、固定budget2敏感性；预算选择单独用2个tune source的zero/2/4。候选宽度暂不扫描。
- 两边各补一个历史定义的整体参照：native FCFS、zero speculation、client cap80。它只回答完整系统效果；tool信息价值仍由同gate三ranker识别，不能混用归因。
- Coding64个测试环境快照已全部完成；tune所需16个不同task路径只初次安装一次。各点同路径恢复workspace与venv，清空私有运行状态；setup不进入E2E、单独记录。

## 后续选择规则

最有利的候选条件是：工具有可合法复用的结果、提前量足以覆盖工具服务且不超有效期、speculation没有挤满共享工具池、LLM存在可通过排序改善的等待。对speculation成本曲线优先验证较低负载，避免高负载的LLM时间掩盖很小的read/Web节省；对co-scheduling保留真实GPU排队和任务长度/工具耗时差异。

预算和policy以tune数据选择；高负载结果只作为既定对照和失效原因诊断。若依据已看过的测试结果更改到达过程或挑子组，须标为自适应探索，不声称未见数据泛化。候选设置只有E2E改善且实际tool reuse/等待变化能解释收益，才值得追加反序配对；如果只有计时噪声，不继续扩大搜索。最终只对选中的主要对照补足3个fresh-server block；来源task是统计单位，Coding副本不是独立样本。

原始证据与运行状态：`/home/aiscuser/resubmission-runs/frozen_20260920/{protocol.json,active_runs.json}`。旧完整矩阵保存在`protocol_full_matrix_superseded.json`，不删除或覆盖已完成结果。

### 补充的单点排序候选

9月2日已有 `full_paper_sensitivity_quick/REPORT.md` 中，beta_G=1.8的mean319.041s低于center331.763s；这是单次开发结果，不是已证最优。本轮只增加这个现成系数的一个探索点，其他模型/gate/trace/predictor/budget4/Top-10/TTL60全固定。当前80任务按completion长度分成4个等大小组，tool-aware对length-aware的mean均回退（6.36%、4.47%、1.31%、0.66%），没有依据靠挑长度分组声称tool信息有效。先验证历史系数候选，再决定是否重复，不新增scheduler机制。

DR延长保留Top-10调参初点mean231.236→214.369s，但实际只采用1个候选、隐藏0.974s总服务。该幅度主要不在直接reuse上，不能选一个最大百分比就宣称收益；在Top-3之后追加一个反序zero锚点检查顺序/网络/服务器波动。

Coding已有`reproduction/README.md`的本地2×2实测表明：路径缓存仅约1.8–2.7ms，而61ms来自另一个name-only进程预热机制。这说明当前read路径的大幅E2E变化可能包含LLM队列反馈/噪声，不能把进程预热收益贴到路径预测上。本轮不扩建新机制。

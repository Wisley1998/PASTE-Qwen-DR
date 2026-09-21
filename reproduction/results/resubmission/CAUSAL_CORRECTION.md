> **v8 readiness 与公平性修正（当前运行版本）**：两组共用完全相同的在线LLM估计、gate、预算、trace、固定执行量、环境恢复和工具状态观测开销；只由tool-aware排序消费tool signals。不向length-aware提供oracle标签，也不人为降低它的预测质量。此前v7只完成length-aware单点，保留为causal/history-only诊断，不能和v8的tool-aware拼成一对。
>
> 已接入现有broker/cache的提交时readiness：DR识别unclaimed/TTL/HTTP freshness有效的完成候选；Coding复用authoritative cache validator，验证epoch、源文件和cache artifact，已完成future仍可ready。pending不等于ready；ready不等于未来命中；失效候选gain=0。DR gain是候选概率加权的服务估计，Coding只估计source I/O减lookup开销（不冒充整个tool/process时延）。两者都是估计，actual adoption仍独立验证。GPU队列等待中不持续刷新snapshot，这一版不声称动态readiness调度。
>
> 当前只跑一组真实配对，逐请求核验EMA/公开cap/固定workload和readiness曝光后再决定重复。DR107测试+23子检查、Coding34测试通过。原始路径：`/home/aiscuser/resubmission-runs/frozen_20260920/causal_readiness`；源码`source_manifest_v8.json`，SHA `713d762fea14316809c7592fccf113dddb1ee253a9afda631829f83d3e6cab8e`。以下v7细节是修正经过，实验次数与当前运行状态以v8 protocol为准。

# Causal co-scheduling correction — 2026-09-20

旧结果保留为 oracle 条件诊断。无论旧结果正收益还是“tool-aware 慢2.2%”，均不能推断在线设计的效果。新结果单独保存，不与旧点拼成配对实验。

原 live 入口直接发出既有 `paste.schedx.causal_prediction.v1` metadata；旧 oracle builder 仅保留于非 live replay。两个 ranker 共用每 session 的完成输出 EMA（初值128，alpha=0.5）、固定剩余一轮先验、下一轮 prompt=当前 prompt 的 persistence 预测。metadata 不包含真实剩余 token/轮数、下一轮 prompt/finality 或未来命中。

DR：原 replay `request.max_tokens` 是 `target_output_tokens +buffer`，并非在线可知原请求上限。新 metadata 使用预先配置的 benchmark 上限1024，仅按当前 context headroom裁剪。Coding：使用原 trace 的 `request_config.max_tokens`，不使用强制回放完成长度。scheduler hook 在 strict schema下也屏蔽executor的强制输出limit。固定LLM工作量保持不变，完成后才更新EMA。

Tool signals仅来自DR既有crossfit预测器与已完成tool wait EMA，或Coding实际prefetch pending状态与已完成tool wait均值。模型/策略保持现有版本；crossfit/posthoc配置的既有评估局限仍适用，不作未触碰测试集上的泛化主张。

返回KV预留使用预测next prompt/output，不回退旧npt/nmt；预测finality不视为authoritative任务结束。本轮关闭final lane，将旧累计foreground session gate上限设为80（达到workload总量，避免借oracle final标记释放它）；保留DR decode32/33、Coding decode24/25和physicalKV0.93 gate。ranker间上述设置完全相同。

实验：保持原80条DR测试trace及Coding已批准2/8划分中的8条测试source（每条8副本）；budget4，tool capacity16，arrival scale1，FIFOcap80，coalesce3.2s。只比较length-aware/tool-aware，3个fresh-server配对block，第二block反转顺序。Coding复用64个prepared snapshots，恢复状态在计时外，无重复安装。未依据旧oracle正负结果重选trace或beta。

验证：DR103测试+23子检查，Coding20测试通过；覆盖live/oracle分支边界、共同估计输入约束、未来字段poison不影响排序/准入/返回KV、executor输出limit屏蔽、DR上限来源。每个完成实验另外审计逐请求EMA、workload和共同LLM metadata一致性。

源码：`/home/aiscuser/resubmission-runs/frozen_20260920/source_manifest_v7.json`，SHA `b178aed6b060db0f3b7d5fe1de2cb38e97f4c338effa09e5776b73efca79787f`。

原始实验：`/home/aiscuser/resubmission-runs/frozen_20260920/causal_ranking`。Coding第一length-aware点使用v6；其Coding runner与scheduler hook逐字节等同v7。DR v6点因output-derived cap被中止，保存在`excluded/causal_v6_dr_output_derived_cap`，不参与比较。

目前GPU实验正在运行；以该目录的result.json与paired_analysis.json为准，尚无完成的性能结论。

> **2026-09-20 修正：本文旧结果全部属于 oracle 条件诊断。** 旧入口向 scheduler 提供真实剩余调用、完成 token 和下一轮 prompt/finality；因此下文无论正收益还是负收益均不能推断在线设计的效果。“tool-aware 慢 2.2%”不表示在线 tool information 无价值。保留原始数据与历史分析，但暂停以此选择在线配置或作论文主结果；以修正后的 causal 实验为准。正确性和资源原始计数仍是该 oracle workload 的事实记录。

# Speculation cost / E2E：新实验，测量中

沿用原 `run_trace_all_visit_live.py --prepare-only` 与 `run_dr_trace_hybrid_pair.py --tool-backend live`，前 80 条 source trace 测试、后 20 条调参。由于原 unified DR plan 缺失，本轮是新基线，不能称作旧 39.66% 的精确复现。原 source traces 是近期实验继续使用的留存 workload。

Search 按要求模拟并经过共享工具池；Web 使用实际 HTTP。工具调用按 task/job 记录物理开始、完成、采用与清理，分别统计模拟 Search 与物理 HTTP 成本。发送请求数包含 redirect；字节是实际交付的响应 body（解压后），不包含未读取的取消缓冲区及 headers。未测的 CPU / outbound bytes 记为 null。

冻结协议：`/home/aiscuser/resubmission-runs/frozen_20260920/protocol.json`。最终测试 DR 代码为 `source_manifest_v3.json`。v2 已修正排队候选转正式执行后不应重复 freshness fallback 的边界，cap 调参在 v2 上完成；cap12/20 mean 分别为 301.690/250.117 s，选择 cap20。较早的 `final-cap12` / `final-cap20` 有该重复执行问题，保留作诊断，不作为最终结果。

目前没有最终路径的完整 load × budget 正式对照。先前调参的提前采用很少，不能据此宣称加速。严格 HTTP freshness 校验保持开启；不能通过放宽校验或引用旧 replay 收益来得到正结论。正式报告需要三个 fresh-server 配对 block，并同时呈现完成率、HTTP 错误、E2E 和相对 zero-speculation 的实际成本增量。

2026-09-20 续跑：v2 首个测试点因 2/80 个 session 的 LLM ServerDisconnectedError 已排除，原始数据完整保留在 `frozen_20260920/excluded/dr-v2-block1-high-b0-tool-aware`。v3 只修正本地 LLM 连接复用；最终测试各组统一使用 v3，speculation 校验和独立选择的 cap20 不变。

v3 首个 zero-speculation 测试点已完整完成：80/80 sessions，696 次 LLM 请求；mean E2E 320.106 s、p95 567.735 s。实际 Web 调用 411 次、发送 HTTP 请求 447 次、交付 body 45,437,536 bytes；模拟 Search 323 次单列。106 个失败 URL 按原样记录，80/80 表示 trace 运行结束，不是全部工具成功。配对 budget4 尚在运行；逐任务数据见 `per_task_cost_e2e.csv`。


首个 v3 配对 block（相同 tool-aware gate/ranker，80 个 source task，非正式结论）：

| Spec cap | 完成 | Mean E2E | p95 E2E | 实际 HTTP 请求 | 交付 body bytes | Useful / unused spec |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 80/80 | 320.106 s | 567.735 s | 447 | 45,437,536 | 0 / 0 |
| 4 | 80/80 | 313.952 s | 563.851 s | 1190 | 143,805,138 | 3 / 715 |

单 block mean E2E 下降 1.92%，同时增加 743 次 HTTP 请求和 98,367,602 bytes 响应体。715 次未采用提前执行消耗 1788.737 worker-seconds。真实被隐藏服务时间从物理 job 账本汇总为 5.836 s；旧 summary/task 的 saved-tool 字段沿用 offline 占位值，不能作 live 收益指标，CSV 的 `actual_saved_service_s` 使用 job 账本。

总暴露 tool 等待反而从 1639.997 增至 1720.364 s；LLM 请求平均时间从 34.061 降至 33.237 s。此次 E2E 时间变化主要在 LLM 请求阶段，不能将全部 1.92% 解释为 tool latency hiding。真实工具错误 URL 数为 106/108；错误与环境波动保留，不剔除。此 operating point 尚不支持“额外 speculative work 值得”的结论，需其余负载、预算及配对 block 后判断。

未采用原因细分：413 次过期、282 次任务结束取消、15 次 freshness 校验拒绝、5 次执行失败。只有 23 个已执行候选等到了对应正式确认（其中 3 个采用）；因此不能把全部 715 次 unused work 都称为“错误预测”，也不能用 unused 比例直接替代 predictor accuracy。

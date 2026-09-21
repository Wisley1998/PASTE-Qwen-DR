> **2026-09-20 修正：本文旧结果全部属于 oracle 条件诊断。** 旧入口向 scheduler 提供真实剩余调用、完成 token 和下一轮 prompt/finality；因此下文无论正收益还是负收益均不能推断在线设计的效果。“tool-aware 慢 2.2%”不表示在线 tool information 无价值。保留原始数据与历史分析，但暂停以此选择在线配置或作论文主结果；以修正后的 causal 实验为准。正确性和资源原始计数仍是该 oracle workload 的事实记录。

# Co-scheduling：首个高负载block（探索结果）

全部对照使用80条source trace、同一模型与predictor、budget4、实际Web/模拟Search、FIFO pre-engine coalescing3.2s。前三项保持同一Joint/KV gate，仅切换engine ranker；native为独立20条tune trace选出的cap20对照。

| 策略 | mean E2E(s) | p95(s) | 完成 |
| --- | --- | --- | --- |
| Shared gate + tool-aware | 313.952 | 563.851 | 80/80 |
| Shared gate + FCFS | 357.687 | 579.857 | 80/80 |
| Shared gate + length-aware | 307.258 | 560.202 | 80/80 |
| Native FCFS + tuned cap20 | 390.034 | 721.534 | 80/80 |

Tool-aware相对FCFS降低12.23%、相对tuned cap降低19.51%，但比length-aware慢2.18%。因此当前不能声称tool信息提供了独立正收益；第一组完整结果包含反例，不以FCFS较弱结果替代length-aware。

验证：同plan、同source tasks、同LLM token work、同model/predictor，所有session完成，资源清理完整，见`first_block_validity.json`。实际Web错误104–108次，全部保留；trace完成不等于所有网页访问成功。不能宣称动态网页跨轮byte一致。只完成一块，未包含块间波动，不能作正式显著性结论。

根据用户要求，停止原45点自动全矩阵。先用独立tune数据选择可有效采用的speculation设置，再只对有依据的负载/排序设置补配对重复。详见`SETUP_SELECTION.md`；原始结果位于`/home/aiscuser/resubmission-runs/frozen_20260920/dr/block1`。

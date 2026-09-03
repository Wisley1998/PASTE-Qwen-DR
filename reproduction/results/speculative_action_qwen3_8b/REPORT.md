# Speculative Actions（Qwen3-8B）固定 Trace 实验

## 结论

在最新 PASTE Qwen-DR `llm_x0_42` trace 的 100 个完整 session、599 个
`LLM generation → tool call` 边界上，Qwen3-8B Top-3 speculator 只有 8 次
工具名和完整参数完全匹配（1.336%）。这 8 次全部晚于对应的权威大模型
generation 窗口，因此有效命中为 0，工具侧和端到端延迟节约均为 0%。

这不是忽略小模型耗时得到的结果。每条收益使用实际测得的小模型 HTTP、
vLLM 排队和生成总延迟计算：

```text
head_start = max(0, recorded_generation_window - measured_speculator_latency)
saved      = exact_match ? min(tool_duration, head_start) : 0
```

## 固定设置

- Trace：`traces/my_traces_tool_slo_search_uniform_1_3s_visit_serial_uniform_2_8s_llm_x0_42`
- Trace session / LLM call / tool call：`100 / 873 / 599`
- Speculator：`Qwen/Qwen3-8B`
- Revision：`b968826d9c46dd6066d109eabc6255188de91218`
- 部署：BF16、单张 A100、vLLM 0.10.1、32K context、FCFS、prefix cache
- 推理：Top-3、temperature 0、non-thinking、512 max output tokens、并发 1
- 权威模型：不重新生成；只使用 trace 中记录的 generation window 和输出
- 验证：工具名与完整 canonical JSON 参数完全相同才算命中

## 结果

| 指标 | 结果 |
|---|---:|
| 可预测边界 | 599 |
| 内容完全匹配 / 不匹配 | 8 / 591 |
| 内容完全匹配率 | 1.336% |
| 有效按时命中 / 未命中 | 0 / 599 |
| 有效命中率 | 0.000% |
| malformed、fail-closed | 36 |
| 合法空预测 | 24 |
| 有候选但完整参数不匹配 | 531 |
| 小模型在权威窗口内完成 | 137 |
| 按时且产生候选的 case | 121 |
| 实际可启动的 speculative tool call | 157 |
| 物理工具调用数 | 599 → 756（1.262×） |
| Speculator 平均延迟 | 3.058 s |
| Speculator p50 / p95 延迟 | 2.724 / 9.110 s |
| 工具 stall | 3274.878 → 3274.878 s |
| 工具侧节约 | 0.000 s（0.000%） |
| 汇总端到端时间 | 5908.854 → 5908.854 s |
| 端到端节约 | 0.000 s（0.000%） |
| 平均每 session 端到端时间 | 59.089 → 59.089 s |

按工具拆分的内容完全匹配为：`search 7/368`、`visit 1/229`、
`google_scholar 0/2`；三类工具的有效按时命中均为 0。

## 0.42x Speculator 反事实与 PASTE 总表

为与已缩放到 `0.42x` 的权威 LLM 时间保持同一口径，本表额外将
Qwen3-8B Speculator 的每次实测延迟乘以 `0.42`。PASTE 使用同一份
599-tool-call trace 上的已有 fixed Top-10 + session-cache 结果。没有重新运行
LLM 或工具实验。

| 方法 | Cache-inclusive hit rate（目标范围） | 实际被加速 tool / 全部 599 tool | 被加速 tool 原始耗时 / 全部 tool 时间 | 平均每个命中 tool-call 的加速比例 |
|---|---:|---:|---:|---:|
| PASTE | 359/499 URL = **71.944%**（聚合为 187/229 visit tool calls = **81.659%**） | 187/599 = **31.219%** | 2125.283/3274.878 s = **64.897%** | **67.441%** |
| Speculative Actions（Speculator latency ×0.42） | 8/599 = **1.336%** | 4/599 = **0.668%** | 12.584/3274.878 s = **0.384%** | **4.180%** |

第一列保留各方法的原生命中单位。PASTE 预测和复用的是可执行
`visit` URL：359 个命中全部包含 session cache 复用，其中 257 个在权威
调用到达时已就绪，102 个仍在执行但可复用已经完成的部分，159 个由更早
的 decision 启动。按原始 `visit` tool call 聚合，229 个调用中有 187 个
至少命中一个 URL，即 81.659%。第二列才把这 187 个被加速的
`visit` call 放到全部 599 个异构 tool call 中衡量覆盖面；31.219% 不是
PASTE 的原生 hit rate。最后一列是先对每个命中 tool call
计算 `saved / original tool time`，再取算术平均；迟到的内容命中按
`0%` 计。Speculative Actions 的 8 个内容命中中有 4 个缩放后仍然
迟到，因此 8 个命中的平均加速比例为 `4.180%`；若只对 4 个实际被
加速的 call 取平均，则为 `8.360%`。

## 完整性与边界

- 599 个 case ID 与 599 个 prediction ID 一一对应且均唯一。
- Case SHA-256：`373acb823c8309ccf604d70b1d9a5c6a16bc4c4b9414ad51f37408524152a711`
- Prediction SHA-256：`b45cdf5013c2a7951f8290b791cdf558cb88be61ecc5835932f28e602bcab91d`
- 独立重算严格匹配、按时命中和节约时间均与主报告一致。
- 预测出错、迟到或参数不完全相同都会执行原 trace 的 demand path，不会改变
  权威输出。

逐 case 证据、原始模型输出、token usage 和实测延迟保存在
`reproduction/artifacts/speculative_action_qwen3_8b/`。该目录中的
`prepare_manifest.json`、`collection_manifest.json`、`predictions.jsonl` 和
`report.json` 构成完整可审计证据链。

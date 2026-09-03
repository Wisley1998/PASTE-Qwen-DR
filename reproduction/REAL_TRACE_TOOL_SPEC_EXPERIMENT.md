# Corrected real-trace speculative tool experiment

更新日期：2026-09-02

> **Wall 结果已被后续实验取代。** 原始 corpus 的 search timing 含约 `10 s/query` 的串行
> 膨胀和 `7277 s` 异常点；visit timing 也有 65 个 `25--30 ms` batch，且 terminal tools
> 被当作零时长。最终修复版对每个 search call 采样 `Uniform(1,3 s)`、对每个 visit URL
> 采样 `Uniform(2,8 s)` 并串行求和，同时因果平移 timestamps、补齐 terminal completion。
> 本文件的旧 multi-URL credit 公式也会低估 Spec 收益；最终口径和结果详见
> `PER_TASK_MULTI_SPEC_WALL_EXPERIMENT.md`，不应再使用本文件的 modeled wall。

## 结论

旧的固定 `20 ms` 实验只能测控制面开销，不能估计 Web `visit` 的真实收益。第一版
real-trace runner 虽然改用了真实 stall，但仍有三处 setup 问题：没有去除 LLM 排队膨胀、用
全体 evaluation trace 的 batch-level 中位数参与策略、并且 service 估计与按 URL 计分的
atomic 口径不一致。本版已修正这些问题，并把矩阵扩展到 C1--C128。

修正后，所有 C/K cell 的净收益仍为正。固定 K4 在 C1、C64、C128 分别净省
`15.589 s`、`2.304 s`、`1.580 s`。高并发不应固定 K4：在 physical-call amplification
不超过 `1.5x` 的已扫配置中，C64/C128 采用 K16，分别净省 `6.061 s`/`6.081 s`，相对各自
K4 提升 `2.63x`/`3.85x`。

## Trace 与时间审计

- 输入仍为 `traces/my_traces` 的 100 个 JSONL session。
- 对路径排序的 100 条 `sha256sum` 记录再次取 SHA-256，manifest 为
  `a38141de38b8207efea20f93904c23c48c80c329fe3b6ad681b8bb05bedbbfae`。
- 仓库及 frozen worktree 中只有两份该 100-session corpus；文件名和内容逐字节相同，没有
  找到另一套已做 30% 校正的 trace。
- OOF search decisions 为 340；116 个下一步为 `visit`，共 235 个可执行 HTTP URL。
- 93 个 visit 有后续 LLM，可测 stall 合计 `209.984 s`；23 个 terminal visit 没有后续 LLM，
  仍按零可测收益处理。
- 可测 visit stall：均值 `2.264 s`、p50 `0.270 s`、p95 `8.654 s`、最大 `44.488 s`。

Trace 的 LLM `timestamp` 是完成时间。不能把 LLM 缩短后再用新的 duration 反推 tool 完成
时间，否则会把删掉的 queue time 错算成 tool stall。本版采用：

```text
observed visit stall = next raw LLM request start - visit timestamp  # 保持不变
scaled LLM lead      = 0.70 * raw inference_time
hit saving           = min(observed visit stall, scaled LLM lead) / URL count
```

即 LLM lead 和 modeled LLM wall 统一乘 `0.70`，但已经观测到的工具间隔不变。测试会逐
decision 验证 scaling 前后 tool stall 完全相等。

## Predictor 与 service 策略

- URL pattern 概率仍使用 Pattern-v2 nested OOF：5 个 outer folds、4 个 inner calibration
  folds、whole-session split。此前的 `20 ms` 没有进入 pattern 特征或概率训练；它影响的是
  pattern 之后的 admission/ranking。
- 每个 outer validation fold 的 service 模型只读取另外四个 folds 的 visit timing，消除了第一
  版用全体 trace 中位数造成的 evaluation-to-policy leakage。
- 多 URL visit 先转为 atomic 样本：`atomic service = visit stall / executable URL count`；训练
  fold 一次 n-URL visit 贡献 n 个 URL 样本。
- 对每个候选 URL，按 hostname 取得训练 folds 的 empirical service 分布，并以固定 prior
  strength `10` 向 outer-fold global 分布收缩；未知 domain 直接回退 global 分布。
- 排序不再使用 `min(median_service, lead)`，而使用长尾分布的截断期望：

```text
expected overlap = E_train[min(atomic service, 0.70 * trace LLM lead)]
value            = p_exact * expected overlap - coordination_cost
```

- 每个 decision 最多选择一个 `value > 0` 的 exact URL candidate；每个 lockstep batch 跨当前
  active sessions 做 global Top-K。
- coordination cost 主矩阵仍取 `1 ms/start`；另做了 `5.2--200 ms` 的 resource-price
  sensitivity。这个参数不能再解释成 tool service。
- exact hit 复用 speculative result，并省掉匹配 AUTH URL call；物理调用数为
  `selected + authoritative - exact hits`。

## 矩阵

- Task concurrency：`C={1,2,4,8,16,32,64,128}`。
- Global K：`K={1,2,4,8,16,32,64,128}`。
- 每个 cell 使用 32 个确定性 session-order seeds；它们是调度敏感性重复，不是独立 trace。
- corpus 只有 100 sessions，因此 C128 的实际 active-session 上限为 100。
- 指标包括累计 gross/net saved tool stall、tool-stall reduction、lockstep modeled wall speedup、
  visible AUTH hit rate、precision 和 production physical-call amplification。

## 固定 K4 的校正结果

| Task C | Net saved stall | Tool-stall reduction | Modeled wall speedup | Visible hit rate | Call amp. |
|---:|---:|---:|---:|---:|---:|
| 1 | `15.589 s` | `7.42%` | `0.93%` | `16.60%` | `2.128x` |
| 2 | `15.589 s` | `7.42%` | `0.81%` | `16.60%` | `2.128x` |
| 4 | `15.589 s` | `7.42%` | `0.69%` | `16.60%` | `2.128x` |
| 8 | `8.938 s` | `4.26%` | `0.52%` | `9.63%` | `1.621x` |
| 16 | `5.281 s` | `2.52%` | `0.39%` | `6.13%` | `1.324x` |
| 32 | `3.289 s` | `1.57%` | `0.29%` | `3.64%` | `1.187x` |
| 64 | `2.304 s` | `1.10%` | `0.21%` | `2.25%` | `1.120x` |
| 128 | `1.580 s` | `0.75%` | `0.09%` | `1.70%` | `1.098x` |

相对第一版 real-trace K4，0.7 lead 会降低可隐藏的累计 stall；因此第一版的 `19.025 s`
并不是保守值，而是偏高。与此同时，缩短 modeled LLM denominator 后，C1/C64 wall speedup 从
`0.83%/0.16%` 变为 `0.93%/0.21%`。旧 setup 对不同指标的偏差方向不同，不能笼统说成全部
低估。

## 高并发预算优化

如果只最大化累计 latency，K 至少等于 C 时会启动全部 314 个 eligible decisions：所有 C 都
净省 `15.589 s`、tool stall 降低 `7.42%`，代价是 `2.128x` call amplification。更实用的
选择是在已扫点中限制 amplification 不超过 `1.5x`：

| Task C | Selected K | Net saved stall | Tool-stall reduction | Wall speedup | Call amp. |
|---:|---:|---:|---:|---:|---:|
| 4 | 1 | `4.977 s` | `2.37%` | `0.35%` | `1.300x` |
| 8 | 2 | `5.443 s` | `2.59%` | `0.37%` | `1.305x` |
| 16 | 4 | `5.281 s` | `2.52%` | `0.39%` | `1.324x` |
| 32 | 8 | `5.628 s` | `2.68%` | `0.38%` | `1.359x` |
| 64 | 16 | `6.061 s` | `2.89%` | `0.25%` | `1.417x` |
| 128 | 16 | `6.081 s` | `2.90%` | `0.09%` | `1.357x` |

C1/C2 下 K1 仍会在每个 sequential batch 启动一次，总 amplification 分别为 `2.128x` 和
`1.604x`；单靠 K 无法满足 `1.5x`。resource-price sweep 没有找到既低于 `1.5x` 又保持正
modeled net saving 的已扫阈值，说明低并发需要更好的 confidence/value calibration，而不是
继续提高一个拍脑袋的“协调成本”。

## 解读边界

- `15.589 s` 是跨 100 sessions 的累计节省，不是 makespan 节省。并发越高，多数节省发生在
  非关键路径；因此 C128 即使放开到 K128，lockstep wall speedup 也只有约 `0.09%`。
- 当前 wall 仍沿用 `session_stream_batches` 的 lockstep model，而不是 live event-driven HTTP/
  vLLM benchmark。它适合与旧 artifact 做同口径比较，不应当作真实集群 makespan claim。
- Search 结果直接复用 trace；search latency 不进入 treatment-baseline 差值。
- Wrong speculation 的远端排队/限流外部性没有由此 deterministic replay 重新生成，主要通过
  call amplification 和 resource-price sensitivity 展示。

## 复现与 artifacts

Runner：`reproduction/scripts/run_pattern_v2_trace_timing_net_benefit.py`

```bash
python reproduction/scripts/run_pattern_v2_trace_timing_net_benefit.py \
  --concurrencies 1 2 4 8 16 32 64 128 \
  --repetitions 32 \
  --global-k-sweep 1 2 4 8 16 32 64 128 \
  --llm-duration-scale 0.70 \
  --domain-prior-strength 10 \
  --coordination-cost-ms 1.0 \
  --output-dir reproduction/results/pattern_v2_trace_timing_corrected_domain_oof_k1_128_c1_128_r32
```

主 artifacts：

- `reproduction/results/pattern_v2_trace_timing_corrected_domain_oof_k1_128_c1_128_r32/REPORT.md`
- `reproduction/results/pattern_v2_trace_timing_corrected_domain_oof_k1_128_c1_128_r32/metrics.json`
- `reproduction/results/pattern_v2_trace_timing_cost_{5.2,10,20,50,100,200}_k1_32_c1_128_r32/`

测试：`reproduction/tests/test_pattern_v2_trace_timing_net_benefit.py`。

本次 runner SHA-256：
`9b49773c4c4d7b365b3a5b87e8ca76a9ba5d73d52157679e61281ae430713e41`。

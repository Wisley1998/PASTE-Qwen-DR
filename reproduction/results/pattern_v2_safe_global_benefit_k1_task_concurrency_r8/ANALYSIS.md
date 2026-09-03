# Safe Global Benefit：Task 并发实验分析

## 结论

`safe_global_benefit` 达成了 worker/tool resource entitlement 的结构性隔离，但没有
达成字面意义上的端到端 no-regression：

- Task concurrency 为 1、2 时，authority exposed latency 有可重复收益；
- concurrency=4 只有边缘收益；
- concurrency>=8 时，没有稳定、且能归因于 speculative hits 的 latency 收益；
- 计入 predictor、selection 和 physical drain 后，所有并发点的 drained-wall / throughput
  都比 demand-only 差；
- deterministic all-wrong 下存在可重复的共享 control-plane/event-loop/cancellation 开销，
  即使 baseline worker/tool slots 没有被 speculation 征用。

因此，本轮数据支持“投机不会占用 baseline worker/tool entitlement”，但不支持“所有
端到端性能指标在任何情况下都不比 demand-only 差”。

## 实验配置

```text
trace sessions                 = 100
task concurrency               = 1, 2, 4, 8, 16, 32, 64, 98
baseline workers / visit cap   = 4 / 2
isolated speculative slots K   = 1
synthetic service / lead       = 20 / 10 ms
paired repetitions             = 8 (AB/BA counterbalanced)
scenarios                      = observed nested OOF + deterministic all-wrong
```

每个 scenario/concurrency/repetition 使用相同 seed 和 session batching 做 demand-only 与
treatment 配对。主 artifact 包含 16 个 aggregate cells、256 个 sample runs。

Payload SHA256：
`ee7af1ef13f166c66bac85ffe7bfd5c8a732bbfb2eb56b2bcd33ccc3d70bb67d`

## Observed-label 结果

正的 conservative net 表示在 authority wait 收益中扣除 pattern feature、probability
lookup 和 selection charge 后仍然优于 demand-only。负的 drained-wall benefit 表示
计入排空与计算 charge 后整体更慢。

| Task C | Mean authority wait baseline -> safe | Conservative net ms/target | Repeat median；正重复 | Overlap coverage | Call amp | Waste ms/target | Drained-wall benefit | 判断 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 30.75 -> 28.90 ms | +0.992 | +0.961；8/8 | 16.1% | 1.892x | 15.716 | -0.4% | 稳定收益 |
| 2 | 34.16 -> 32.87 ms | +0.500 | +0.374；8/8 | 10.6% | 1.517x | 8.939 | -2.2% | 稳定但较小 |
| 4 | 39.63 -> 38.75 ms | +0.124 | +0.313；6/8 | 6.2% | 1.299x | 5.280 | -4.6% | 边缘收益 |
| 8 | 53.57 -> 52.64 ms | +0.196 | +0.049；4/8 | 3.6% | 1.172x | 3.142 | -5.3% | 不稳定 |
| 16 | 81.39 -> 80.44 ms | +0.221 | +0.163；5/8 | 2.3% | 1.095x | 1.738 | -5.6% | 不稳定 |
| 32 | 130.31 -> 129.06 ms | +0.525 | +0.724；6/8 | 1.4% | 1.055x | 1.020 | -5.3% | 不能归因于 hits |
| 64 | 220.16 -> 219.43 ms | +0.024 | -1.080；3/8 | 0.9% | 1.035x | 0.694 | -5.4% | timing indistinguishable |
| 98 | 291.42 -> 291.25 ms | -0.534 | -0.185；4/8 | 0.4% | 1.031x | 0.663 | -5.7% | 无收益 |

### 如何解释并发趋势

随着 concurrency 从 1 增加到 98：

- overlap hits 从 303 降到 8，coverage 从 16.1% 降到 0.4%；
- wrong starts 从 1388 降到 56；
- physical-call amplification 从 1.892x 降到 1.031x；
- wasted service 从 15.716 降到 0.663 ms/target；
- 被选候选的平均预测概率从 0.148 上升到 0.248。

这说明全局 Top-B 的资源收缩行为符合预期：session 越多，仍只有一个全局 K slot，有限
机会会优先给更高预测收益的候选，投机量不会随 session 数线性增长。但同样因为每个
batch 只有一个安全 start，单个 task 获得 overlap 的机会会迅速下降。

C>=8 的 pooled raw benefit 明显超过 direct hits 能解释的量级。例如 C=32 只有 26 个
hits，10 ms lead 下直接隐藏预算约为 260 ms，但 conservative pooled benefit 约为
988 ms。Closed-loop ripple 和 separate replay 抖动无法在本实验中分离，所以不能把该
cell 的正值作为 speculative execution 的因果收益。

## Deterministic all-wrong

这里保留同样的 candidates、probabilities、batching 和 load，但把每个 authority URL
替换成确定不命中的 URL。

| Task C | Raw authority regression ms/target | Conservative net ms/target | Repeat median；正收益重复 | Call amp | Drained-wall benefit |
|---:|---:|---:|---:|---:|---:|
| 1 | +0.270 | -1.109 | -1.067；0/8 | 1.899x | -4.5% |
| 2 | +0.202 | -0.991 | -1.019；0/8 | 1.525x | -5.1% |
| 4 | +0.268 | -1.024 | -1.024；0/8 | 1.309x | -5.7% |
| 8 | +0.454 | -1.206 | -1.214；0/8 | 1.182x | -6.6% |
| 16 | +0.256 | -0.993 | -1.050；0/8 | 1.105x | -6.3% |
| 32 | +0.710 | -1.445 | -1.379；0/8 | 1.062x | -6.2% |
| 64 | +1.596 | -2.313 | -2.738；1/8 | 1.041x | -6.5% |
| 98 | +0.494 | -1.207 | -1.556；1/8 | 1.034x | -5.7% |

Raw authority wait 在 64 个 repeats 中有 55 个回归。C=1--64 的 raw drained wall 在
56/56 个 repeats 中更慢。由于 speculation 无法命中，这些结果不能解释为收益；它们
测到的是共享 asyncio event loop、task creation、broker lock/cancellation 和 loser drain
等未隔离成本。

## K=0 no-op sanity

另一个独立 R4 sanity 在 C=1/8/32/98 使用 `K=0`：

- selected/requested/admitted/started 全为 0；
- predictor windows、probability candidates、selection compute 全为 0；
- physical amplification 为 1.000x；
- broker 保持 baseline W=4/C=2；
- 全部 safety invariants 通过。

K=0 all-wrong 的 paired raw wait 差异符号混合，wall 差异约在 -0.19% 到 +0.11%，符合
零中心 replay 噪声。相同并发下 K=1 的大部分一致负偏因此不能只归因于 harness noise。

## Safety 与 provenance

- 256 个主实验 samples、每个 15 项，共 3840 项 safety invariant 全部为 true；
- baseline max running total 为 2；K=1 treatment max total=3、max speculation=1；
- authority worker/visit caps 始终保持原始 W=4/C=2；
- all-wrong 每个 cell 的 1880 个 authority targets 全部走 `executed`，hit/reuse/race=0；
- payload、runner、broker、policy、100 条 trace hashes 均复核一致；
- `REPORT.md` 可由当前 runner 逐字重建，CSV 的 16 行标量与 metrics 一致。

## 下一步

在继续增加 K 或做更多 repetitions 之前，应先处理当前暴露出的两个固定成本：

1. 把 predictor/selection 计算移出 authority 的共享 event loop，或为 control plane 也提供
   与 tool slot 相同的隔离证书；
2. 增加一个真正便宜的 pre-gate，只对与全局安全预算同量级的 shortlist 做完整概率计算，
   并把 control-plane overhead/risk margin 纳入正收益阈值。

完成后应重复同一 observed/all-wrong/K=0 concurrency matrix。只有 all-wrong 相对 K=0
不再出现系统性回归，才有依据把声明从 structural resource no-regression 提升到端到端
no-regression。

# Azure LLM 2024 Trace 驱动 Agent 实验说明书

## 1. 设计结论

本仓库把 Azure LLM Inference Trace 2024 用作 **Agent session 的外部到达过程**，而不是把它当作 Agent trace 本身：

```text
Azure CSV 第 i 行的 TIMESTAMP
          │
          └──> 启动一个完整 Agent session
                    ├── LLM call 0：消息、prompt/max tokens 来自 Agent trace
                    ├── tool wait：来自 Agent trace
                    ├── LLM call 1：来自 Agent trace
                    └── ...
```

也就是说，替换前后只改变顶层 session 的启动时间。以下内容不变：

- Agent 的 message history；
- 每轮 `prompt_tokens`、`target_output_tokens` 和 `max_tokens`；
- session 内部的 LLM/tool 调用顺序；
- 第 2 个及之后请求的原生 tool wait；
- PASTE 的 tool-overlap 变换和 scheduler metadata 逻辑。

Azure CSV 的 `ContextTokens` 和 `GeneratedTokens` 会写入输出 workload 的溯源元数据，但不会覆盖 Agent 请求。原因是 Azure 的一行代表一次独立 LLM invocation，而本实验的一个 Agent session 包含多轮、工具依赖和不断增长的上下文；直接替换 token 数会生成内部不一致的 Agent workload。

接口实现位于：

- `scripts/azure_llm_trace.py`：读取、筛选和映射 Azure CSV；
- `scripts/prepare_azure_agent_workload.py`：离线生成固定 workload；
- `scripts/run_vllm_trace_experiment.py`：运行时直接接入 Azure CSV。

## 2. 数据集

使用 Microsoft 官方的 [Azure LLM Inference Dataset 2024](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md)。数据集包含 conversation 和 code 两个匿名化的一周 trace，CSV schema 为：

```text
TIMESTAMP,ContextTokens,GeneratedTokens
```

下载：

```bash
mkdir -p datasets/azure_llm_2024

curl -L \
  https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv \
  -o datasets/azure_llm_2024/AzureLLMInferenceTrace_conv_1week.csv

curl -L \
  https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv \
  -o datasets/azure_llm_2024/AzureLLMInferenceTrace_code_1week.csv
```

对当前 Qwen DeepResearch 实验，建议把 conversation trace 作为主结果，把 code trace 作为 workload robustness 补充结果。数据集被 DynamoLLM 使用；DynamoLLM 发表在 HPCA 2025，而不是 ISCA。

## 3. 推荐流程：先生成一次，再让所有系统复用

这是论文实验最稳妥的用法。它确保 FCFS、PASTE、其他 baseline 使用逐字节相同的到达过程和 Agent-template 映射。

### 3.1 准备原生 Agent workload

如果已有以下文件，可直接进入下一步：

```text
reproduction/artifacts/workloads/eval_learned/prepared_workload.json
```

否则先按仓库原有流程生成 calibration/evaluation split：

```bash
bash reproduction/scripts/prepare_joint_workloads.sh
```

Azure trace 只能替换 evaluation workload 的到达过程。scheduler 的 calibration workload 仍必须使用独立的训练 session，不能使用待测 Azure-mapped workload 做校准。

### 3.2 生成 Azure-driven Agent workload

下面示例选择 conversation CSV 中最早的 300 个 invocation，并把到达时间压缩 10 倍：

```bash
python scripts/prepare_azure_agent_workload.py \
  --agent-workload reproduction/artifacts/workloads/eval_learned/prepared_workload.json \
  --azure-trace datasets/azure_llm_2024/AzureLLMInferenceTrace_conv_1week.csv \
  --azure-dataset-variant conversation \
  --azure-max-sessions 300 \
  --azure-arrival-speedup 10 \
  --azure-session-mapping shuffled_round_robin \
  --seed 20260417 \
  --output reproduction/artifacts/workloads/eval_azure_conv_300.json
```

如需选择特定时间窗口，可再加入：

```bash
  --azure-start-time '2024-05-12T08:00:00+00:00' \
  --azure-duration-s 600
```

`--azure-start-time` 是 inclusive；第一个实际选中的 CSV 行被归一化为实验时间 0。`--azure-duration-s` 从这个行开始计算，区间右端不包含。`--azure-max-sessions` 是安全上限；如果要覆盖完整时间窗口，应把它设置得足够大。

模板不足时的映射方式：

- `round_robin`：按固定顺序循环复用 Agent sessions；
- `shuffled_round_robin`：每轮用固定 seed 洗牌后复用，推荐用于正式实验。

每个输出 session 都有唯一 `trace_id`。复用不会改写模板 message，因此各系统之间仍可严格对齐。

### 3.3 在现有 Joint/FCFS 实验中替换 workload

现有 wrapper 不需要改动，只要把 evaluation workload 指向新文件：

```bash
export PASTE_EVAL_WORKLOAD="$PWD/reproduction/artifacts/workloads/eval_azure_conv_300.json"
export PASTE_CALIBRATION_WORKLOAD="$PWD/reproduction/artifacts/workloads/calibration_learned/prepared_workload.json"
export PASTE_TRACE_SPEEDUP=10
export PASTE_MAX_ACTIVE_TRACES=0

bash reproduction/scripts/run_joint_cell.sh fcfs azure_conv_fcfs
bash reproduction/scripts/run_joint_cell.sh joint azure_conv_paste
```

这里将 `PASTE_MAX_ACTIVE_TRACES=0` 设为不限制。`run_joint_cell.sh` 默认的 30-session semaphore 会在 30 个 session 已经 active 时推迟后续 session 的启动，从而把 open-loop Azure 到达过程变成 admission-gated workload。若必须限制 client 并发，应明确把它作为单独的实验变量，而不能称为原始 Azure arrival replay。

## 4. 备选流程：运行器直接读取 Azure CSV

调试时可以省略离线转换：

```bash
python scripts/run_vllm_trace_experiment.py \
  --prepared-workload reproduction/artifacts/workloads/eval_learned/prepared_workload.json \
  --azure-arrival-trace datasets/azure_llm_2024/AzureLLMInferenceTrace_conv_1week.csv \
  --azure-dataset-variant conversation \
  --azure-max-sessions 300 \
  --azure-arrival-speedup 10 \
  --azure-session-mapping shuffled_round_robin \
  --seed 20260417 \
  --output-dir reproduction/artifacts/runs/azure_conv_direct \
  --server-url http://127.0.0.1:8000 \
  --model Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \
  --speedup 10 \
  --scheduler-metadata-mode online \
  --scheduler-calibration-workload reproduction/artifacts/workloads/calibration_learned/prepared_workload.json \
  --tool-overlap-mode learned \
  --tool-prediction-model reproduction/results/tool_only/url_rank_mapper.json \
  --tool-wait-mode sleep \
  --max-active-traces 0
```

运行器会把最终固定 workload 保存到：

```text
reproduction/artifacts/runs/azure_conv_direct/prepared_workload.json
```

正式 A/B 对比时，应把该文件固定下来，并在其他系统上通过 `--prepared-workload` 复用；不要为每个系统重新截取一次 CSV。

## 5. 两个 speedup 参数不能混淆

| 参数 | 作用对象 | 对第一次 LLM call 的影响 |
|---|---|---|
| `--azure-arrival-speedup` | session 之间的 Azure 到达偏移 | 有影响 |
| `--speedup` / `PASTE_TRACE_SPEEDUP` | session 内部第 2 轮及之后的 tool wait | 无影响 |

例如 Azure 原始 session 到达偏移为 `[0, 2, 7]` 秒，`--azure-arrival-speedup 2` 后为 `[0, 1, 3.5]` 秒。Agent 内部 10 秒 tool wait 只有在 `--speedup 5` 时才变成 2 秒。

要做 wall-clock faithful replay，两者都设为 1。要做压力实验，可以分别调节它们，但论文中必须同时报告。

## 6. 输出与校验

转换后的 `meta.arrival_process` 记录：

- Azure CSV 路径和 SHA-256；
- conversation/code 类型；
- 第一、最后时间戳；
- 原始和 replay 后的时间跨度；
- arrival speedup、映射方式和 seed；
- Azure invocation 数和 Agent template 数；
- `azure_token_fields_used_for_agent_payload=false`。

每个 session 的 `azure_arrival` 记录原始 CSV row number、UTC timestamp、Azure token 数以及原始/压缩后的 arrival offset。

快速检查：

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path('reproduction/artifacts/workloads/eval_azure_conv_300.json')
w = json.loads(p.read_text())
print(json.dumps(w['meta']['arrival_process'], indent=2))
offsets = [t['initial_delay_s'] for t in w['traces']]
assert offsets == sorted(offsets)
assert all(t['requests'][0]['wait_after_prev_s'] == t['initial_delay_s'] for t in w['traces'])
print('sessions:', len(w['traces']), 'replay span (s):', offsets[-1])
PY
```

代码测试：

```bash
python -m pytest -q tests/test_azure_llm_trace.py
```

## 7. 论文中应如何描述

建议使用下面这类表述：

> We derive the open-loop arrival times of top-level agent sessions from the
> 2024 Azure LLM Inference Trace. Each selected Azure invocation launches one
> complete held-out agent session. The agent's messages, multi-turn call graph,
> token budgets, and tool delays remain unchanged; Azure token-count fields are
> retained only as provenance. We use a fixed seed for session-template mapping
> and replay the identical materialized workload across all systems.

同时报告：Azure 子集（conversation/code、起止 timestamp、CSV SHA-256）、session 数、最终 LLM call 数、两个 speedup、映射 seed，以及是否设置 client concurrency gate。

这个实验能够回答“真实云端到达突发性下系统表现如何”，但不能声称 Azure CSV 本身是原生 Agent workload。Agent 的多轮结构仍来自本仓库的 held-out Agent traces。这个边界应主动写进 methodology，正面回应 reviewer 对 trace suitability 的疑问。

## 8. 常见错误

- **CSV schema 报错**：必须是官方的 `TIMESTAMP,ContextTokens,GeneratedTokens`；接口也支持 `.csv.gz`。
- **时间窗口没有数据**：检查 timestamp 是否带 UTC offset，并确认落在对应一周内。
- **不能重复映射**：已经含有 `meta.arrival_process` 的 workload 不能再次应用 Azure CSV，避免二次压缩。
- **到达时间被推迟**：检查是否设置了正数 `--max-active-traces` 或 `PASTE_MAX_ACTIVE_TRACES`。
- **实验负载远高于 Azure 原始 QPS**：一行 Azure invocation 在这里启动的是一个多轮 Agent session，因此总 LLM request rate 会被 Agent 平均调用轮数放大；必须同时报告 session rate 和 LLM-call rate。
- **校准数据泄漏**：不要把 Azure-mapped evaluation workload 传给 `--scheduler-calibration-workload`。

## 9. 引用

- Azure Public Dataset, *Azure LLM Inference Dataset 2024*: <https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md>
- Stojkovic et al., *DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency*, HPCA 2025: <https://arxiv.org/abs/2408.00741>


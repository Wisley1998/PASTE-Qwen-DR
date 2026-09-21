# Correctness：本轮可支撑的 claim

2026-09-20。提前执行仅在受支持的只读操作及明确有效性条件下复用；失效时执行普通路径。此处验证工具输出与正式工具状态，不宣称任意动态网页、任意外部并发写入、任意 shell/edit speculation 或完整 autonomous agent 答案等价。

| 实际路径 | 成对场景 | 结果 |
| --- | --- | --- |
| DR `LiveToolBroker → WikipediaLiveExecutor`，真实本地 HTTP fixture | completed、in-flight、错参数、跨 session、TTL 过期、执行失败、缺少 freshness、无 freshness 的内容变化、固定编码 Vary、Cookie Vary，共 10 个 | off/on 输出与正式历史/fixture 状态 0 差异；采用/fallback 均与预期一致 |
| Coding CLI `SpeculativeToolRuntime → ReadFileTool` | completed、in-flight、错参数、跨 session、过期、失败、文件变化、epoch 变化、单次采用、zero budget，共 10 个 | 0 差异 |
| Coding trace 实验实际调用的 `swe_typed_tool` 私有读缓存 | 有效命中、错路径、文件变化、缓存内容损坏，共 4 个 | off/on 完整公开工具输出与正式 workflow state、源文件内容一致；prefetch 不推进正式状态 |

DR audit 原始记录：`correctness_audit.json`。Coding CLI 记录：[correctness_audit.json](/home/aiscuser/gemini-cli-PASTE/reproduction/analysis/resubmission/correctness_audit.json)。Typed 场景在既有 `test_swe_typed_workflow.py::test_speculation_off_on_preserves_authoritative_output_and_state` 中执行，不新建实验 runner。

HTTP 复用依赖 origin 的 freshness 契约：明确 max-age、有效 Date/Age、固定请求表示；拒绝未知 Vary、Set-Cookie、private/no-cache/no-store。关闭持久 cookie jar，避免 speculative 请求污染后续请求。该条件不等价于对任意动态内容的 snapshot 证明；未知 freshness 自动 fallback。

Coding 性能路径只提前填充已有的只读 path cache。正式 repo_read 仍验证 attempt/workspace、epoch、路径、文件元数据及缓存完整性，并按正式参数生成结果。其余 typed tool 普通执行。文件 stat/epoch 检查不宣称可抵御任意外部并发写入。

复查命令：

```bash
cd /home/aiscuser/PASTE-Qwen-DR
PYTHONPATH=reproduction python reproduction/scripts/run_resubmission_audit.py --output reproduction/results/resubmission/correctness_audit.json
PYTHONPATH=reproduction /opt/conda/envs/ptca/bin/python -m pytest -q reproduction/tests/test_live_broker.py reproduction/tests/test_qwen_workset_scheduler_patch.py reproduction/tests/test_resubmission_resources.py

cd /home/aiscuser/gemini-cli-PASTE
node reproduction/scripts/run_resubmission_runtime.mjs reproduction/analysis/resubmission/correctness_audit.json
PYTHONPATH=reproduction /opt/conda/envs/ptca/bin/python -m pytest -q reproduction/tests/test_swe_trace_live_pair.py reproduction/tests/test_swe_typed_workflow.py reproduction/tests/test_qwen3_swe_controlled.py
```

最近验证：DR 96 passed + 23 subtests；Coding Python 57 passed。额外检查涵盖取消后的真实 HTTP 请求/redirect/body 计数，以及超时后只清理工具自己的进程组，避免漏计残留子进程。未运行 smoke。

性能使用原入口 `run_dr_trace_hybrid_pair.py --tool-backend live` 和 `run_swe_trace_live_pair.py --tool-replay-mode live`，绑定同一 broker/typed executor。Search 按用户要求模拟并经过共享容量池；Web、本地读写和测试实际执行。LLM 消息/token 工作与 authoritative calls 固定于 trace，因此性能结果不等同于 autonomous solve-quality 评估。

排队候选在实际执行前转为 authoritative 时，按普通执行处理：不重复做 speculative freshness fallback，也不记作 useful/unused speculative work。对应定向回归验证只执行一次。此修正后的 DR 路径绑定 `/home/aiscuser/resubmission-runs/frozen_20260920/source_manifest_v2.json`；Coding 与共享 scheduler 的源码未变。

# PASTE 完整 policy 实验 setup：v11

2026-09-20。两个项目共用的当前执行规范。本文件取代旧手册中 policy 的配置与命令；旧 correctness / 物理资源报告仍按原版本保留。

**当前状态：实现已修正，reference setup 已定义；尚未用修正后的调参结果证明最优或取得新的性能数字。** “选择最大提升”是下文的调参目标，不是预先宣布 FULL 胜出。机器配置以 [policy_setup_v11.json](../reproduction/configs/policy_setup_v11.json) 为准，每次执行绑定不可覆盖的 `setup_lock.json`。不要手工拼接不同历史实验的最高收益和参数。

## 1. 实验究竟运行什么

- 保留两个原入口：DR `run_dr_trace_hybrid_pair.py`；Coding `run_swe_trace_live_pair.py`。使用共享 Qwen scheduler hook 和 Qwen `start_vllm.sh` / `stop_vllm.sh`。
- 每次 LLM 请求在真实 vLLM/GPU 上执行，保留冻结的 prompt / completion token 工作量。DR 为原消息；Coding 为原实验的 deterministic token envelope。输出不用于新的任务质量评价。
- 工具是 **recorded-duration 异步模拟**：时间真实流逝，等待、排队、重叠和 GPU requeue 都进入测量；不跳过工具时间、不再缩放时钟。
- 不恢复 64 套工具环境，不重新执行真实 HTTP/test/env，不启动 smoke。CPU 单元测试和输入审计不是性能运行。
- 调度器只接收当前请求、已完成历史、当前候选及提交请求时可见的 readiness；完整 trace、future durations、suffix lengths、命中标签仅供 executor / 事后分析，不能用于排序。
- 结果只支持这个 trace-conditioned latency workload；模拟 worker-seconds/calls 不能写作真实 CPU/network 成本。正确性与物理成本引用独立的实际工具证据。

## 2. 确认修正的机制及版本边界

| 项目 | v9/v10 问题 | v11 规则 |
|---|---|---|
| speculative priority | runner 传负 confidence，broker 大值优先，方向相反 | 传正 confidence；Top-K 作为原子 batch 入队，broker 大值优先 |
| pending / cache | 已完成结果也占 128 个 pending 名额 | queued/running 上限 1024；completed LRU 独立上限 4096 |
| 满队列 | 直接拒绝后来候选 | 高 confidence 可替换最低 confidence 的 queued 候选；不读真实命中 |
| TTL | 从入队起 60 秒，覆盖了排队和 LLM 等待 | timed 稳定结果按 session 保留；结束即释放；LRU 仍受限 |
| 重复使用 | 第一次 authoritative claim 后移除结果 | 成功执行的 speculative 结果在同一 session、完整 invocation 一致时可再次使用 |
| 工具并行 | 80 个任务共享最多 4 个 speculative workers | DR reference 为 shared pool 64，spec 可利用全部空闲 slot；总数仍不超过 64 |
| authority 保护 | 错误投机占满 worker 时需等待 | timed executor 可取消最低 confidence 的未确认 running speculation；authority 优先。取消的已执行时间仍计成本；不把取消当作成功 |
| URL 时长 | multi-URL 总时长平均分配 | 采用 plan 的逐 URL `visit_units.duration_s`，严格验证 URL 顺序和串行和；无 executable unit 的原非 HTTP 调用按 batch 均分 |
| 收益账本 | live/timed summary 仍累加 offline_saved_s，显示零 | 逐调用记录实际采用的提前服务，汇总至 task 与 summary；timer credit 受当前 authoritative 服务时长约束 |

无限保留和抢占只由 timed 模式显式启用；真实 HTTP 模式仍必须满足结果有效期，不继承“网页无限有效”的假设。共享 broker 的旧默认行为保留给其他入口。

**串行转并行的含义必须保持一致：** baseline 的一个 multi-URL visit 按原 URL 顺序逐一执行；FULL 提前并行执行 Top-K 候选。authoritative 仍逐一确认，有 completed hit 立即采用、in-flight hit 等剩余尾部、miss 普通执行。不能只给 FULL 额外并行化 authoritative batch，再把那部分增益全部归为预测。

## 3. 固定硬件、模型和服务配置

| 参数 | DR | Coding |
|---|---|---|
| 模型 | Alibaba-NLP/Tongyi-DeepResearch-30B-A3B | Qwen/Qwen3-Coder-30B-A3B-Instruct |
| revision | `4b0ac5767427a55d08a254f0367e2934976598e0` | `b2cff646eb4bb1d68355c01b18ae02e7cf42d120` |
| GPU quartet / port | 0,1,2,3 / 8200 | 4,5,6,7 / 8300 |
| TP / dtype | 4 / BF16 | 4 / BF16 |
| max model length | 16384 | 32768 |
| max_num_seqs / batched tokens | 48 / 2048 | 48 / 2048 |
| GPU memory utilization / CUDA graph size | 0.86 / 32 | 0.86 / 32 |
| prefix caching / chunked prefill | 开启 / 开启 | 开启 / 开启 |
| FULL & controlled arms session working set | 40 | 24 |
| decode target / max | 32 / 33 | 24 / 25 |
| physical KV target / respect Joint limits | 0.93 / 开启 | 0.93 / 开启 |
| gate min running / max wait | 16 / 180s | 12 / 120s |
| physical KV rescue | 180s | 120s |
| fixed FIFO admission / coalescing | session 持槽 / 3.2s | session 持槽 / 3.2s |
| shared tool capacity（所有 arms 相同） | 64 | 16 |
| speculative concurrent cap（启用 spec 的 arms） | 64，空闲借用，authority 可抢占 | 4 |
| predictor / Top-K | 冻结 Pattern-v2 crossfit / 10 | 冻结 result-path prefix emissions / 3 |

DR 64-slot 来自旧 Top-10 load-curve 的机制参考，是这次有利条件的 reference；不是已经测得的最优点。16/32/64 在 tune 上统一比较，每个 operating point 的所有 baseline 同样使用该总工具容量。不得拿 FULL 的 64-slot 与 baseline 的 16-slot 比。

每个项目的全量基础环境来自自己的 `unified_workset_v1.env.example`，在干净环境解析后写入 lock。表中关键字段及以下共同覆盖一并固定：final lane=0、remaining-call lane=0、running priority=0；禁用未来 finality。其余 service/pressure/aging 参数保留并完整记录：prefill=38112 tokens/s、decode=113.7 tokens/s、EMA α=0.5、初始输出128、remaining-LLM weight=1、gain weight=0.25、tool beta=0.9、tail beta=0.25、remaining-tool weight=0.35、context alpha=1.4、aging alpha=0.2。完整环境以每 cell 的 `cell_manifest.json` 为准。

共同 LLM 估计暂不伪称为完整剩余任务预测：remaining calls prior=1；remaining LLM tokens=2×已完成输出 EMA；下一 prompt 取当前 prompt。它相对旧 oracle 输入的能力差异必须保留。不得为扩大收益把完整 trace 标签重新传入 scheduler。

保护既有 ResNet/background workload；记录每次 GPU UUID、型号、驱动、可用显存和背景进程占用。不同 arms 在同一 quartet 串行运行，每个 cell 启动 fresh server。只用匹配 state 目录的 stop 脚本管理本次 server，不按 GPU 占用批量杀进程。

## 4. 工作负载、调参集和负载强度

| 项目 | 调参 | 测试/验证 |
|---|---|---|
| DR | 原独立20个 source sessions ×4独立 runtime replicas =80 | 原80个 source sessions，各1次 |
| Coding | sphinx-8474、pylint-7114，各32个 runtime replicas =64 | 原8个 test source，每个8 replicas =64 |

增加调参副本的原因：旧 DR20 / Coding16 并发量低于32/24 decode gate，可能没有可排序的 queue，不能用来选择高负载调度。副本各自持有独立 session/cache，不共享结果；统计单位仍是 source session/task，不能把80/64写成独立样本数。

arrival 来自既有冻结 Azure arrival 序列，DR80 与 Coding64 的调参副本复用相应测试负载的 **arrival timestamps only**，不读取测试任务内容/命中选择参数。reference arrival_scale=1，arrival span约2.956154s。调参只允许下文预列的1/2/4倍间隔；每个候选的所有 arms 完全相同。

准备脚本 [prepare_policy_tuning.py](../reproduction/scripts/prepare_policy_tuning.py) 仅复制 trace 与工具记录；已生成：

- `/home/aiscuser/resubmission-runs/setup_v11/dr_tune_80.json`：20个源任务，708次 LLM 请求。
- `/home/aiscuser/resubmission-runs/setup_v11/coding_tune_64.json`：2个源任务，640次 LLM 请求。
- `setup_v11/coding_tune_predictions_64/`：从 `tune-original/coding/cap8/tasks/*/trace` 复制192个候选的64个任务记录；忽略原 release offsets/hit/outcome 标签，按 origin turn 在真实完成前缀后发布。
- 测试 DR：原 `dr_test_original_entry.json`，696次 LLM 请求。
- 测试 Coding：原 `real_trace_hybrid_v3/plan.json` 按8个source过滤，1240次 LLM 请求、64个任务、280个候选；使用 v8 length-aware audit。

原测试集已经在探索中被检查过；即使本轮参数仅按 tune 选择，也应称 retrospective validation，不能重新声称全新 sealed holdout。

## 5. 必须同时保留的五种对照

| cell | LLM scheduler | working set / gate | speculation | 回答问题 |
|---|---|---|---|---|
| `native_fcfs` | 原生 vLLM FCFS，无 Joint hook | client FIFO80，原生max48 | 关闭 | 完整系统相对原生 baseline |
| `length_only` | Joint length-aware | DR40/32；Coding24/24 | 关闭 | 相对仅长度调度的方法 |
| `length_spec` | Joint length-aware | 与 FULL 完全相同 | 与 FULL 完全相同 | 加入工具信息排序的独立价值 |
| `full` | Joint tool-aware | DR40/32；Coding24/24 | 开启 | 完整因果方法 |
| `gated_fcfs_spec` | Joint FCFS ranker | 与 FULL 完全相同 | 与 FULL 完全相同 | 控制 gate/spec 后的排序价值 |

所有 arms 都使用同一模型、tokens、source tasks、arrival、总工具容量、3.2s coalescing和计时起点。完整系统对比允许表内声明的 working-set/gate差异；`full` vs `length_spec` / `gated_fcfs_spec` 除 ranker 外必须完全相同。不能将同gate FCFS称作原生 baseline。不能只展示 length_only 而省掉更强的 length_spec。

Coding 目前仍是 **partial-read 实现**：path+epoch 匹配，仅抵扣源文件 I/O 减去 lookup；不抵扣整个 subprocess，也不投机 env/test。历史52.47%包含这些昂贵调用的离线完整服务抵扣，不能贴到这里。DR修正不自动恢复Coding的那部分能力；两个项目的收益分别测量与报告。

## 6. 选择最有利 setup 的固定规则

1. 先用修正后的 reference 跑完整 tune 五臂，不启动测试集“挑最好点”。没有排队时先检查实际 waiting/ordering evidence，不能靠降低 baseline 限额制造优势。
2. DR阶段一：tool capacity/spec cap共同取16/32/64（idle-fill上限），arrival scale取1/2/4，共9个点；每点的所有对照共享同一总容量与arrival。阶段二只在阶段一最优点比较gain weight=0.25/0.9/1.8。Top10、cache4096、pending1024和session retention固定，避免再混入多种机制。
3. Coding独立比较spec cap=0/2/4与arrival scale=1/2/4；总工具容量16固定；不得拿DR调参结果直接替代Coding选择。
4. 每候选至少3个fresh-server blocks，旋转并交替cell顺序；保留失败、重试和所有负结果。候选比较以source task聚类，block作为上层重复；不能把副本当独立样本。
5. 主目标：最大化 `min(1 − mean_FULL/mean_native_FCFS, 1 − mean_FULL/mean_length_spec)`，并要求 FULL 优于 length_only；FULL p95不能比两个主对照各自恶化超过5%。所有 arms 完成率100%、workload一致、无fail-open才可入选。
6. 差异在1个百分点内优先较小 speculative worker-time，再优先较小总容量。若配对区间不支持正收益，记录“未证明优势”；不把噪声最大点写为方法优越性。
7. 选中后另存候选配置、全部调参结果/manifest SHA、目标值、区间及选择原因，生成 **新的** selected lock。测试前冻结，不再改参数。失败后的实现修复必须升版本并重跑整个配对，不能只补跑FULL。

## 7. 冻结与生成原入口命令

当前已物化的 reference lock 与命令位于 `/home/aiscuser/resubmission-runs/setup_v11/`。脚本使用Python标准库，不会自行启动GPU或工具。示例：

```bash
cd /home/aiscuser/PASTE-Qwen-DR
python reproduction/scripts/policy_setup.py verify \
  --lock /home/aiscuser/resubmission-runs/setup_v11/setup_lock.json
```

首次冻结/新版本冻结使用新输出文件（禁止覆盖）：

```bash
python reproduction/scripts/policy_setup.py freeze \
  --config reproduction/configs/policy_setup_v11.json \
  --output /absolute/new-version/setup_lock.json
```

生成完整3-block五臂命令，保持原runner：

```bash
python reproduction/scripts/policy_setup.py render \
  --lock /home/aiscuser/resubmission-runs/setup_v11/setup_lock.json \
  --project dr --split tune \
  --run-root /home/aiscuser/resubmission-runs/v11_dr_tune_reference \
  --output /absolute/new-command-file.sh
```

Coding换 `--project coding`；测试换 `--split test`。`--formal` 要求lock状态为 `selected_on_tune`，reference lock会拒绝该标签。生成命令本身不意味着执行；运行生成的shell脚本才会启动正式请求。

生成脚本会：

- 清除ambient `VLLM_*` 等变量，显式应用干净解析并锁定的完整profile；阻止旧shell里的高优先级alias覆盖。
- 每个cell之前验证源码、配置、原始输入、candidate records和serving Python/package版本；不一致即停止。
- 使用项目排他锁、全新run root、fresh server、独立state/log目录；启动前写入精确client argv与server环境；保存完整client日志。
- 正常完成和异常退出都调用有PID身份校验的stop脚本；已有result不得覆盖。

不要把v11命令塞回 `frozen_20260920/timed_policy/*_commands.sh`。那个目录是v9/v10证据，原始manifest和结果不修改、不混合配对。

## 8. 每次运行要核验与报告的指标

**公平性：**每cell原始task/request key、prompt/completion usage、authoritative tool序列一致；ranker比较的因果共同metadata一致；模型revision、fresh PID、GPU quartet、总工具容量、arrival、cache/budget一致。核验实际server日志/ready-turn记录，不能只根据env配置声称生效。

**效果：**从外部arrival开始的mean/p50/p95/p99 task E2E，包含coalescing、admission、模型/工具排队；另外报告makespan、throughput、LLM latency、exposed tool wait、pre-engine wait。cleanup与setup单列，端到端定义不能中途改变。

**预测到复用的漏斗：**候选Top-K覆盖率（事后审计）、admitted、capacity rejection、priority replacement、physical simulated start、completed/running/queued claim、重复cache reuse、cache eviction、TTL/session cleanup、取消/抢占、错误/失败。URL命中率分母是eligible visit URL需求次数，不包括search；queued promotion不能当作已隐藏完整服务的hit。

**并行证据：**spec/authority同时running的时间积分及high water；是否真的在LLM前完成或隐藏了尾部；总slot约束、authority抢占、串行URL顺序均应可还原。高预测覆盖但没有admission/start时不能归咎于predictor。

**成本：**模拟总调用、实际启动的spec调用、useful/unused worker-seconds、抢占前已执行时间、缓存peak bytes/byte-seconds；一份spec结果多次使用时物理执行只记一次，逐demand节约单独累计。CPU/network字段为null。

**验收不以预设速度为门槛：**工作量/正确性/资源限额/因果性不合格则拒绝性能归因；效果为零或负仍保留。未经新配对实验验证，不能宣称这些修正已恢复39.66%或52.47%。

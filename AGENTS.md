# PASTE 实验入口

DR / Coding 实验统一按 `/home/aiscuser/resubmission-runs/PASTE_METHOD.md` 执行。先读该手册中的组件、trace、setup 和接线状态，再选择代码与脚本。历史 README 与结果目录用于追溯历史实验。

新实验使用 `/home/aiscuser/resubmission-runs/policy_entry_20260921/` 的显式配置入口；不要调用历史生成器默认配置。policy / 消融只用 trace-duration sleep 工具；日常检查用该目录 `check_policy.sh`，真实工具集成另见 `TOOL_INTEGRATION.md`，不作为本类实验前置条件。

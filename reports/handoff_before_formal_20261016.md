# 当前批次交接

继续当前仓库 feature/physical-hierarchical-ppo；不重建、不恢复旧瞬移环境、不commit/push。legacy_snapshot_20261016和全部旧结果保留。

本轮用户已明确授权实际CARLA和6小时本地训练，旧“未安排不得长训练”条件已解除。批次为 runs/20261016T015226Z_research_batch_3377580f，开始于2026-10-16 09:52（北京时间），总截止15:52。所有队列共享batch.json的deadline_epoch，不自动越过预算。

## 已执行

- 指定D盘CARLA 0.9.15实际启动，隔离CPython3.7.9客户端严格匹配，主.venv保持3.11。
- 800步实际标定、静态绕障/弯曲行驶停车/动态横穿诊断、实际撞障传感器验证。
- 真实PPO冻结策略迁移；H/B1双方约21k CARLA物理步微调，各8次优化，模型与归一化已保存。它们是微调起步，不是收敛证明。
- 轻量H/B1各20k调试及配对100k，继续3个seed；原始失败保留。
- development_v2针对终点前长动作越界问题，版本化参考修复、共享beta=10与B1探索候选，重新训练后才能比较。
- 普通回归35通过；真实CARLA桥接测试单独通过。所有测试XML在reports。

## 读取实时状态

```powershell
Get-Content runs\20261016T015226Z_research_batch_3377580f\queue_status_*.json
Get-Content runs\20261016T015226Z_research_batch_3377580f\carla_pair_results.jsonl
.\.venv\Scripts\python.exe scripts\research_report.py --batch runs\20261016T015226Z_research_batch_3377580f
```

不要仅依据旧PID判断任务状态；Windows venv有启动器和实际Python子进程，watchdog JSON保存启动器PID，heartbeat保存实际执行PID。检查最新优化日志、checkpoint和heartbeat增长。

## 恢复约束

检查对应队列已结束后再启动同一run，避免并发写同一模型。在剩余本批预算内，start_research.ps1 +相同queue+source snapshot会从latest.json累计恢复优化器、归一化、物理状态和RNG；不重新从100k开始训练。OSQP工作区重建，不能声称恢复位级完全相同。

```powershell
.\scripts\start_research.ps1 -Batch runs\20261016T015226Z_research_batch_3377580f -Queue configs\development_v2_queue.json -SourceRoot runs\20261016T015226Z_research_batch_3377580f\source_development_v2
```

STOP文件存在时先读取原因；不要自动绕过6小时截止。预算到达保存真实进度，下一批需新的明确时间安排。CARLA只允许一个bridge tick进程；运行其他测试前确认没有CARLA训练/评估占用。

## 尚需判断/执行

以最新结果而非本交接静态数字为准：完成开发方案冻结后，累计推进100k/300k/1M及五种子；B2、hold10、shared_shield独立训练；最终固定测试清单与主要指标尚待对应训练完成。不得把开发集、小样本或一次胜出当最终结论。

服务有一次UE访问冲突，已保留CrashContext并重启本批服务；清理流程增加退役帧，结束后保持空世界同步/无渲染。崩溃根因尚无符号栈确认。当前CARLA曲线路径是直道路面上的障碍弯曲路线，原生弯道路形泛化尚未验证。详见CARLA_bridge.md及批次failure_analysis.md。

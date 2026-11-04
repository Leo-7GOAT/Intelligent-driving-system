# 当前批次交接

沿用 feature/physical-hierarchical-ppo；不重新初始化Git，不commit/push，不恢复旧瞬移环境。legacy_snapshot_20261016、所有历史结果和失败日志均保留。

本轮用户授权的本地六小时批次是 runs/20261016T015226Z_research_batch_3377580f，北京时间2026-10-16 09:52–15:52。所有队列共享batch.json硬截止；不要自动延长，不要无条件删除STOP。

## 真实当前状态

以脚本输出为准，不按静态PID判断。2026-10-16 11:58检查时：formal_v4五seed的H/B1均完成100k；300k已完成9/10，H seed55继续。最终1M和CARLA正式队列正在等待前置条件，不能称已完成。

```powershell
.\.venv\Scripts\python.exe scripts\batch_status.py --batch runs\20261016T015226Z_research_batch_3377580f
.\.venv\Scripts\python.exe scripts\publication_report.py --batch runs\20261016T015226Z_research_batch_3377580f
```

Windows venv有启动器和真正Python子进程。heartbeat记录真正执行PID；watchdog记录启动器。联合核验PID、heartbeat、optimization.jsonl、checkpoint和评估原始轨迹增长。旧queue_status文件可能代表已经结束的进程。

## 已实测

- 指定D盘CARLA0.9.15启动，独立Python3.7.9客户端匹配，主.venv保持Python3.11。一次UE访问冲突原始XML保留，当前重启实例空闲探测正常。
- 800个实际标定控制步；静态绕障、弯曲路线停车、动态横穿的真实底层诊断及真实撞障传感器验证。诊断不计PPO成果。
- H/B1各约20k轻量调试，配对100k扩至3seed；v2负结果、v3结果和旧640步均保留。
- 轻量100k模型真实CARLA冻结开发迁移H/B1均0/6；双方各约21k CARLA微调、各8次真实优化后H为2/6、B1为0/6，仅seed11。
- 普通回归40通过、3项条件跳过。真实CARLA桥接与真实PPO续训另有通过记录。续训计数曾失败，修复复测累计5304实际世界步=5120控制步+184重置步；原失败日志保留。
- ppo_closed_loop_evidence.json核对了9801个真实PPO评估控制帧，包含已保存模型hash、Actor变化、连续frame、真实命令和位移。

## 已实际启动的队列

1. formal_v4：seed11/22/33/44/55，H/B1按共同100k/300k/1M交错累计。当前执行源码source_formal_v4_iofix1；I/O修复前的状态文件争用记录保留，算法文件逐字节一致。不要同时启动第二个相同run。
2. final_evaluation_queue：等全部主模型1M后，用固定40项最终和12项泛化清单评估，冻结归一化。
3. carla_formal_pipeline：等主模型1M后，串行进行五seed冻结开发/最终迁移，双方100k真实CARLA微调，再开发/最终测试。代码hash和协议已登记，不能中途改策略/车辆算法。
4. supplementary_pipeline：主队列1M完成后，B2、B1_hold10、H_shared、B1_shared，seed11/22/33，独立累计100k/300k/1M；受相同截止约束。
5. B3固定最终清单评估等待主1M，规则策略、零训练预算。
6. 独立预算守卫：15:52到点停止任务，只关闭确认本批启动且世界空闲的CARLA服务。

以上“启动等待”不代表它们已经进行了对应训练。实时状态见batch_status和各pipeline/status.json。

## 本批剩余预算内恢复

先确认对应队列确实已经退出，读取STOP原因；不要与仍在运行的进程并发写同一模型。

```powershell
.\.venv\Scripts\python.exe scripts\parallel_research.py --batch runs\20261016T015226Z_research_batch_3377580f --queue configs\formal_queue.json --source-root runs\20261016T015226Z_research_batch_3377580f\source_formal_v4_iofix1 --workers 4
.\.venv\Scripts\python.exe scripts\final_evaluation_queue.py --batch runs\20261016T015226Z_research_batch_3377580f --source-root runs\20261016T015226Z_research_batch_3377580f\source_formal_v4_iofix1 --workers 3
.\.venv\Scripts\python.exe scripts\carla_formal_pipeline.py --batch runs\20261016T015226Z_research_batch_3377580f
```

这些命令会从现有检查点累计接续，并跳过完整结果。轻量恢复优化器、归一化、物理状态和RNG，OSQP工作区重建；CARLA恢复优化器等但新建真实回合，重置tick计费。开发v1缺后续配置字段，不能用新代码静默恢复成v4。应使用原版本快照或明确新版本训练。

六小时到达后，上述命令会按旧截止停止；后续必须有新的明确时间安排，保存原预算记录后再建立续跑窗口。当前批次不自动越界。

## 统计和局限

主判据：五个配对seed全部完成1M固定独立测试，且H−B1安全完成率的20,000次配对bootstrap 95%区间下界>0。开发结果不能替代主指标。次要指标不能事后升格。

当前方案是完整系统比较，H具有几何/车辆模型、规划、停车参考、安全及频率先验。停车区候选可规划率由42/90提升至90/90，但窄通道起点仍仅30/90；不能把底层反复制动当PPO已学会安全。原生弯道路形泛化尚未验证，CARLA曲线路径位于直道路面的障碍路线。实测停车加速度有尖峰；不裁剪真值。

详细输出：research_results.md、failure_analysis.md、analysis/formal_results.md、analysis/primary_decision.json、analysis/paired_final_metric_intervals.json、原始transitions*.jsonl.gz和每个checkpoint。

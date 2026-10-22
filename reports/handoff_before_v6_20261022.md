# 当前批次交接

沿用 feature/physical-hierarchical-ppo；不重新初始化Git，不commit/push，不恢复旧瞬移环境。legacy_snapshot_20261016、所有历史结果和失败日志均保留。

本轮用户授权的本地六小时批次是 runs/20261016T015226Z_research_batch_3377580f，北京时间2026-10-16 09:52–15:52。所有队列共享batch.json硬截止；不要自动延长，不要无条件删除STOP。

## 真实当前状态

训练与评估均已停止。五个B1各保存1000960步；H seed11/22/33/44/55分别保存1001899、842155、1001588、1001790、686986步。11/33/44三组完成1M、40项独立主测试和12项泛化；22/55尚未达1M。15:48:30统一发出STOP，为实测最长超过200秒的rollout留出收尾时间，15:49前两个未完成模型都完整保存；原15:52硬截止未变。v4因检查点RNG缺陷保留为历史开发，不能混入v5确认统计。无新窗口、无后台续训。

预算守卫于15:52:27确认世界为空并关闭本批拥有的CARLA服务器；最终汇总三个脚本均exit_code=0。后续仅进行文档与报告核对，没有预算外训练。最新实测回归另有report_infrastructure_pid_20261016.xml的2项通过；进程存活核验检查创建时间，防止Windows复用旧PID造成假活跃状态。

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
- 最新普通回归46通过、4项条件跳过；真实CARLA传感器启动、重放与回合隔离另外2项通过。真实PPO续训/RNG复测累计5304实际世界步=5120控制步+184重置步；原失败日志保留。
- 修复后的v5 100k源模型在CARLA配对微调：H 50346世界步=48580控制+1766重置，19次优化阶段、760次optimizer.step；B1 50592世界步=48640控制+1952重置，19阶段、594次optimizer.step。固定6条件开发评估H 1/6、B1 0/6，无未解决基础设施故障。不同optimizer步数来自原冻结KL提前停止，不是按成绩分配预算。
- ppo_closed_loop_evidence.json核对真实PPO评估控制帧，包含已保存模型hash、Actor变化、连续frame、真实命令和位移；随着完整评估新增而重新生成。它含历史开发证据，不能全部作为v5正式成绩。
- 1M源模型seed11的CARLA正式配对微调已完成：H50461世界步、B151814世界步；固定30条件测试分别9/30和0/30。H全部成功来自dynamic9/10，static和curve均0/10；冻结迁移双方0/30。seed33的冻结迁移及配对微调也已实际启动，后续完整成绩以批次正式表为准。
- 轻量独立测试H seed11/33/44各30/40，五个B1均0/40。三个完整配对的差值均0.75、样本标准差0，20,000次配对bootstrap区间[0.75,0.75]仅描述已完成子集；五种子主判据仍为incomplete，墙钟截尾并非随机缺失。B3同为30/40，与三个H种子逐条件成败完全一致；H泛化各4/12，B1均0/12。
- CARLA正式seed33微调H 50489世界步=48568控制+1921重置，B1 50685=48640控制+2045重置；最终H 10/30、B1 0/30。两个已完成微调种子的H平均完成率31.67%、样本标准差2.36个百分点，B1均0；独立种子只有2个，区间不计算。
- seed44的H冻结开发6项成功2项，随后独占锁残留导致B1开发6项及双方最终各30项基础设施未解决；H微调在首个物理步/优化前失败，训练交互0。原最多一次重试已经保留，不追加测试重试。确认旧PID37360不存在、无桥接进程、服务端版本匹配且世界为空后，用recover_stale_carla_lock.ps1归档锁。具体留下锁的退出路径尚未证实。

## 已实际启动的队列

以下队列均已在本窗口退出；“已启动”是执行历史，不能据此认为当前仍在运行。B2/hold10/shared_shield的新正式训练仍为0步，仅等待队列曾启动。CARLA的100k阶段和其余种子配对未完成。后续需要新有限时间安排；不要自行清除STOP或扩展已经耗尽的基础设施重试次数。

1. formal_v5：seed11/22/33/44/55，H/B1按共同100k/300k/1M交错累计。当前执行源码source_formal_v5；旧I/O争用和RNG缺陷的原始记录保留。不要同时启动第二个相同run。
2. final_evaluation_queue：每个主模型1M检查点就绪即使用固定40项最终和12项泛化清单评估，2个worker，冻结归一化；主结论仍要求全部5组配对。
3. carla_formal_pipeline：按既定种子顺序实际完成seed11/33的50k阶段；seed44出现上述基础设施故障后退出。100k阶段未执行。代码hash和协议已登记，不能中途改策略/车辆算法。
4. supplementary_pipeline：主队列1M完成后，B2、B1_hold10、H_shared、B1_shared，seed11/22/33，独立累计100k/300k/1M；受相同截止约束。
5. B3固定最终清单已完成40项评估，30项成功，规则策略、零PPO训练预算。
6. 独立预算守卫：15:52到点停止任务，只关闭确认本批启动且世界空闲的CARLA服务。

以上“启动等待”不代表它们已经进行了对应训练。实时状态见batch_status和各pipeline/status.json。

## 本批剩余预算内恢复

先确认对应队列确实已经退出，读取STOP原因；不要与仍在运行的进程并发写同一模型。

```powershell
.\.venv\Scripts\python.exe scripts\parallel_research.py --batch runs\20261016T015226Z_research_batch_3377580f --queue configs\formal_v5_queue.json --source-root runs\20261016T015226Z_research_batch_3377580f\source_formal_v5 --workers 6
.\.venv\Scripts\python.exe scripts\final_evaluation_queue.py --batch runs\20261016T015226Z_research_batch_3377580f --source-root runs\20261016T015226Z_research_batch_3377580f\source_formal_v5 --workers 2
.\.venv\Scripts\python.exe scripts\carla_formal_pipeline.py --batch runs\20261016T015226Z_research_batch_3377580f
```

这些命令会从现有检查点累计接续，并跳过完整结果。轻量恢复优化器、归一化、物理状态和RNG，OSQP工作区重建；CARLA恢复优化器等但新建真实回合，重置tick计费。开发v1缺后续配置字段，不能用新代码静默恢复成v4。应使用原版本快照或明确新版本训练。

六小时到达后，上述命令会按旧截止停止；后续必须有新的明确时间安排，保存原预算记录后再建立续跑窗口。当前批次不自动越界。

后续获得新的明确时间安排时，可使用 `scripts/start_authorized_next_window.ps1 -Batch runs/20261016T015226Z_research_batch_3377580f -Hours 6`。该入口不会被任何当前后台任务调用；本轮仅完成语法和 `-Inspect` 只读检查，未启动新窗口。它要求旧截止已过、旧进程全部退出、STOP仅为正常预算原因且冻结代码校验通过，先归档原batch.json、STOP、状态和统计，再设置显式有限的新窗口、确认/启动匹配CARLA并接续原检查点。源模型、确认方案和最终测试保持原版本，不把续跑视为新的独立训练seed。资源故障或人工STOP不会自动清除。

## 统计和局限

主判据：五个配对seed全部完成1M固定独立测试，且H−B1安全完成率的20,000次配对bootstrap 95%区间下界>0。开发结果不能替代主指标。次要指标不能事后升格。

当前方案是完整系统比较，H具有几何/车辆模型、规划、停车参考、安全及频率先验。停车区候选可规划率由42/90提升至90/90，但窄通道起点仍仅30/90；不能把底层反复制动当PPO已学会安全。原生弯道路形泛化尚未验证，CARLA曲线路径位于直道路面的障碍路线。实测停车加速度有尖峰；不裁剪真值。

详细输出：research_results.md、failure_analysis.md、analysis/formal_results.md、analysis/primary_decision.json、analysis/paired_final_metric_intervals.json、原始transitions*.jsonl.gz和每个checkpoint。

## 最终测试后的独立可行性诊断

15:14前已完成两项单独标注的探索性诊断。独立三次样条优化不调用PPO或Hybrid A*，对全部10个固定unseen布局使用相同搜索预算；在3项上找到曲率/整车几何合法路径，原MPC和轻量VehiclePlant实际执行成功。再把这3条路径组成共同候选库，逐一检查全部10项，成功执行5项（case_id 13、17、25、33、37）；其余5项仍未确定可行性。每次成功均477物理步、无MPC失败，两个诊断实际交互分别1431和2385步，PPO训练步均0；重复条件不增加独立场景数。

该结果说明至少5个布局中的PPO/规则失败不能归因于地图无解。全局特权参考不是可部署PPO成果，不进入完成率、训练预算或模型选择，也不据此修改/删除测试场景。旧解析路径搜索的0见证记录仍保留。见 `analysis/posthoc_spline_feasibility/`、`analysis/posthoc_shared_witness_library/`；全10项图和逐步状态已保存。后续算法改进必须在新开发流程中实施。


## 检查点随机状态修复（12:41，最终测试前）

实测发现旧累计训练在保存后用 PPO.load 验证模型，会重置 Python、NumPy、Torch 全局随机状态，且与保存的恢复状态不一致。现在验证过程用上下文保存并恢复 RNG；真实 B1 连续训练与两段恢复训练的最终参数完全一致，普通回归45通过、3项条件跳过。该测试不能推出重建OSQP工作区的H位级一致。

formal_v4及更早累计训练的观测结果全部保留，但仅作带此实现缺陷的历史开发记录。formal_v5在相同环境/PPO参数、相同5个种子和原测试清单下从头训练；没有续用v4权重。修改时未取得任何最终性能，主指标、统计规则及15:52硬截止不变。根因与修订见批次 checkpoint_rng_audit.json、checkpoint_rng_repair_amendment.json，旧冻结协议见 protocol_history。


## 防重复恢复入口与预算收尾

`scripts/resume_current_batch.ps1 -Batch runs/20261016T015226Z_research_batch_3377580f -Inspect` 只检查进程；省略 `-Inspect` 会补启动已经退出的队列。它拒绝删除STOP、拒绝延长预算，并跳过仍在运行的队列。只能在同一原预算内使用。

`finalize_research_batch.py` 已实际启动，在15:52硬截止后20秒重新从日志生成研究报告、失败分析和真实PPO闭环证据；此报告任务不延长训练。状态见 `finalizer_status.json`。实时训练与CARLA的停机仍由各自原看门狗及预算守卫负责。

12:45真实CARLA续训/RNG复测通过：累计5304世界步、5120控制步、184重置步、800次optimizer.step。额外gSDE检查也证明实际B1探索配置连续训练与恢复训练参数完全一致；工程测试记录不计性能样本。


13:00首次最终成绩前修订调度：主模型每个已完成1M检查点立即执行固定最终/泛化清单，2个评估worker；仍需完整5个配对seed才作主指标结论。B3在固定seed11双方1M后评估。模型、测试清单、统计标准和原截止不变。修订见ready_primary_evaluation_amendment.json。


## 13:27 CARLA传感器启动修复与失败开销

新建IMU在部分回合的出生帧未发送数据，旧实现立即等待该帧导致重置超时。现有30个制动保持重置tick的前5个用于传感器启动，后25个及每个策略控制tick仍严格等待对应帧；物理初始化tick数、动作、状态和奖励逻辑未变。实际30次连续三类场景重置通过，共929个计费重置/回合间清理tick；真实60控制步重放/回合隔离测试也通过。

失败记录未删除：第一次H尝试记录到920世界步、0优化，但额外末段控制/部分重置步不可追溯，实际总量标未知。第二次RPC审计完整恢复14075控制+495重置+2失败重置tick=14572，另有最终退出清理开销；实际日志5次优化，检查点仅保存到4次/10606步。两个失败尝试均不计成功，也不把其权重带入修复后的新配对微调。

新配对目录carla_v5_100k_source_finetune_warmup1，以同一v5轻量100k的seed11源模型重新给H/B1各50k CARLA开发微调。失败成本单列analysis/carla_infrastructure_training_costs.json。正式1M CARLA队列在此串行任务结束后自动恢复；轻量主训练持续运行。旧协议和修订保留于protocol_history及carla_sensor_startup_amendment.json。


13:38首次最终成绩前，CARLA调度改为在已完成配对1M的种子中按11/22/33/44/55固定优先级选择，完成该seed全部预定阶段后再选下一组。只看检查点是否就绪，不看成绩；五组仍全部计划，缺失不补齐、不声称完整结论。硬截止不变，修订和调度源码已归档。

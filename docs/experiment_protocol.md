# 实验协议：2026-10-16 实际批次

以 configs/research_preregistration.json、批次 preregistration.json、development_freeze.json 为准。批次 runs/20261016T015226Z_research_batch_3377580f，北京时间09:52–15:52，六小时硬截止。旧协议存于 reports/experiment_protocol_before_formal_20261016.md；旧“未执行/CARLA缺失”不是本轮状态。

## 固定主指标

共同名义1M物理训练步下，40项独立轻量测试的安全完成率差值 H−B1。训练seed固定11/22/33/44/55；每seed只贡献一项相同清单的完成率。不挑最好seed，不以早期最佳检查点替代1M检查点。

报告逐seed、均值、样本标准差、配对差值和95%区间。固定统计seed 20261016、20,000次配对bootstrap，至少3个独立训练seed才计算区间；回合不是独立训练重复。失败进入总体。基础设施故障单列、最多重试一次、保留首次记录；未解决不得成功。

次要指标：100k–1M开发曲线归一化梯形AUC、首次观测到80%完成率的检查点、碰撞/越界/超时、全回合进度、成功条件通行时间、舒适、接管、规划/MPC失败、计算耗时与泛化。缺失预算不外推；未达到阈值写null。不得事后交换主次指标。

## 冻结 formal_v5

共同任务45秒、物理周期.05秒、完整路线23米。H/B2高层.5秒；B1默认20Hz，hold10另列。Hybrid A*固定1800节点、无搜索墙钟截止；0.1秒部署模式不能混入离线成绩。

共同351维观测、相同静态几何与当前邻车信息、相同beta=1奖励。每物理步gamma=.99^.05；宏动作回报按物理时间折扣；GAE=.95^(N×.05)。真实参数保存于resolved_config.json。

共同轻量课程：<50k真实物理步为6米目标，50k–150k为16米，之后23米；目标在障碍采样前确定。验证/测试始终完整23米。CARLA使用完整道路场景，不套用轻量短路线课程。

终点参考前缀仅在PPO候选弧穿过停车区时提前结束规划参考；实测状态仍由物理执行产生。H/B2/B3拥有停车模型先验，B1直接控制。该差异必须披露。

PPO为64/64网络、lr=3e-4、clip=.2、最多10 epochs、共同target_kl=.03。H entropy=.01；B1采用gSDE、log_std_init=-2、纵向初始均值.2、entropy=.001、每10个20Hz动作重采探索噪声。正式评估始终确定性推理。v1、beta10的v2负结果、v3及冻结理由全部保留；最终测试后不调参。

每rollout名义2560物理步：H 256动作、B1 2560动作。提前终止导致真实覆盖量不同，预算依据实际物理步。记录rollout超额、策略动作、仿真秒、优化阶段、SB3 epoch计数、真实optimizer.step次数、Actor变化。KL提前停止时epoch计数包括已进入但未完成的epoch，不能将其视为完整数据遍历次数；实际参数更新以optimizer hook计数为准。

100k/300k/1M为同一次训练累计里程碑，保存优化器、归一化、物理状态和RNG；恢复重建OSQP工作区，不声称位级一致。首次并行状态文件争用发生在H seed11训练前；I/O修复的算法文件逐字节一致，失败保留。

## 基线与安全归因

H=PPO→局部目标位姿/速度→前向Hybrid A*→时间轨迹→MPC→物理执行。B1直接纵向/转向控制，不调用规划/MPC。B2独立训练PPO→二维A*→经车体/曲率检查的后处理→同一MPC。B3只规则评估，无训练预算。

原生比较称完整系统比较，包含模型、动作空间、频率、停车逻辑、安全机制和算力。shared_shield为双方共同执行端轻量风险制动；H额外时空轨迹制动关闭，但规划失败制动仍属结构差异。原始动作、盾、规划/MPC失败、实际命令分别保存。

supplementary_v5在首次最终测试前注册：seed11/22/33，B2、B1_hold10、H_shared、B1_shared各自累计100k/300k/1M，主H/B1五seed完成1M后启动。hold10每宏动作重采gSDE，保持同样.5秒探索更新间隔。受同一六小时截止约束；不以旧版消融冒充新版归因。

## 清单与模型选择

开发12项：lightweight_development_cases.json；独立最终40项：lightweight_final_cases.json；泛化12项：generalization_cases.json。前6项均为未见布局、附加障碍参数obstacle_num=6（加2个基础障碍，实际8个）；后6项为轴距×1.2、加速度×.8、输入延迟2步、转向滞后.25秒。

首次最终测试前的生成器审计发现，原泛化清单中4个narrow/dynamic项目不应用obstacle_num，因此不能算密度变化。已将密度子集统一为实际支持该参数的unseen布局，保留原清单和旧hash，登记generalization_density_audit_amendment.json与新hash。主40项清单、训练代码/模型和车辆失配子集未变；修改时尚未取得任何最终成绩。

最终清单hash写入development_freeze.json。评估器检查hash、正式版本来源与实际1M训练量，冻结归一化、不训练；完成结果拒绝重跑。中断只恢复未完成case，保留censored记录。无独立几何见证不等于无解，不能按H成败删地图。

新开发选择规则按完成率、碰撞/越界、全回合进度。旧select安全优先入口仅用于历史开发检查；正式主指标固定预算检查点，不调用该入口。

## CARLA迁移与微调

指定D盘CARLA0.9.15已实际启动；独立Python3.7.9客户端唯一同步tick，主Python3.11执行PPO/规划/MPC。dt=.05，物理子步.01；执行中不设置位姿/强制速度，状态和事件来自对应CARLA帧。

静态绕障、障碍形成的弯曲路线停车、动态横穿均有真实道路/actor。曲线路径在直道路面上；原生弯道路形泛化尚未验证。800步实际标定的车辆与执行器供双方共用，因此称低层重新标定后的冻结策略迁移，不称整个系统完全未适配。

开发6项、条件11/22；最终三类×10项、条件1000..1009；微调训练三类×0..9。策略无未来真实轨迹。碰撞按frame/episode归属，终止前排空传感器；不逐帧等待不存在的碰撞事件。

第一轮轻量配对100k后已执行双方开发迁移和双方约20k CARLA微调，仅seed11，不能据此声称收敛或多seed优势。

正式CARLA协议在首次最终评估前保存于carla_formal_pipeline_v5/protocol.json：五seed轻量1M源模型，先冻结开发/最终评估，再双方累计50k CARLA物理步微调及评估；全部五seed完成50k后再推进100k及评估。继承冻结PPO配置，不根据最终成绩修改。物理预算含重置tick，控制/重置单列；退出清理帧为基础设施开销。

正式CARLA按固定seed顺序逐对执行；该seed的H/B1双方均达到1M后即可接续，与后续轻量seed训练重叠。仍只有一个CARLA tick拥有者。补充模型的1M固定测试也在各检查点就绪后执行，避免把已经完成的模型评估推迟到硬截止之后。这些仅为首次最终性能观察前的调度修订，原协议和修订记录保留。

CARLA续训保存优化器、归一化、RNG和计数，重新初始化一个真实回合，不声称恢复服务端内部状态。实际续训测试曾发现重置步计数负值，原始失败保留；修复后真实两段测试通过，正式运行前登记修正。

## 复现与限制

全部run保留源码hash、配置、模型、归一化、状态/动作/奖励/事件、回合、优化与故障日志。无commit/push，不清理历史。普通与真实CARLA条件测试分列。图表由research_report.py/publication_report.py从日志生成。

无成功时条件通行时间为空；B1无参考，MPC误差不适用。每回合P95均值不伪称合并样本P95。并行负载下耗时和节点预算结果不能证明部署硬实时。

C盘空闲<8GiB或可用RAM<4GiB停止；心跳/轨迹300秒不增长触发自有子进程看门狗；截止前60秒请求结束rollout并保存，15:52硬截止。只关闭确认属于本批且空闲的CARLA服务。下一批需要新的时间安排，不自动越界。


## 检查点随机状态修复（12:41，最终测试前）

实测发现旧累计训练在保存后用 PPO.load 验证模型，会重置 Python、NumPy、Torch 全局随机状态，且与保存的恢复状态不一致。现在验证过程用上下文保存并恢复 RNG；真实 B1 连续训练与两段恢复训练的最终参数完全一致，普通回归45通过、3项条件跳过。该测试不能推出重建OSQP工作区的H位级一致。

formal_v4及更早累计训练的观测结果全部保留，但仅作带此实现缺陷的历史开发记录。formal_v5在相同环境/PPO参数、相同5个种子和原测试清单下从头训练；没有续用v4权重。修改时未取得任何最终性能，主指标、统计规则及15:52硬截止不变。根因与修订见批次 checkpoint_rng_audit.json、checkpoint_rng_repair_amendment.json，旧冻结协议见 protocol_history。


13:00首次最终成绩前修订调度：主模型每个已完成1M检查点立即执行固定最终/泛化清单，2个评估worker；仍需完整5个配对seed才作主指标结论。B3在固定seed11双方1M后评估。模型、测试清单、统计标准和原截止不变。修订见ready_primary_evaluation_amendment.json。


13:38首次最终成绩前，CARLA调度改为在已完成配对1M的种子中按11/22/33/44/55固定优先级选择，完成该seed全部预定阶段后再选下一组。只看检查点是否就绪，不看成绩；五组仍全部计划，缺失不补齐、不声称完整结论。硬截止不变，修订和调度源码已归档。


## 本批收尾与基础设施缺测说明（最终测试后）

原09:52–15:52窗口不变；因实测rollout最长超过200秒，15:48:30向全部队列统一发出STOP，允许完整优化/检查点收尾。此为资源保护，未按模型成绩选择停止对象，未修改模型、清单、主指标或统计seed。

seed44 CARLA出现旧PID退出后的独占锁残留，预定一次重试后仍无物理回合的记录全部保留。报告修正为：计划已关闭不等于实际评估完整；无观测的策略率为缺失，保守计划条件完成下界另列，不制造零碰撞结果或新增独立策略seed。缺测列表显式输出，实际任务失败仍全部计入。该修正不改变seed11/33的已观测结果。详见analysis/late_infrastructure_accounting_amendment.json；针对无观测和PID复用新增2项报告回归通过。没有追加测试重试。

## 2026-10-22 native-road development controls

The four native-road routes are development diagnostics selected by geometry before control execution; they are not independent final conditions. Actuator comparison holds route/reference/targetspeed/MPC/vehicle/goalgates fixed, retains everycase andframe, and reports comfort/speed/latency alongside completion. Optional modes do not become the training default without a new frozen version and matched H/B1 adaptation. The observed4/4cannot establish a population success rate or PPO contribution. See the native/pose report for failedcoast andmanualgear hypotheses. The v17terminal-posePBRS trial is single-seed development with preregistered0/100k/300k/1M/2M/3Mmilestones and nofinal-testselection.

# 可执行性感知分层自动驾驶研究

[English README](../README.md) · [验收记录](research_acceptance.md) · [方法](design.md) · [GitHub 预发布](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases)

研究主链路为 **PPO 高层决策 → 局部目标与速度 → Hybrid A* → 时间轨迹 → MPC → CARLA 实际状态与奖励反馈**，面向已知地图下的低速车辆任务。

**目前是开发预发布，整体研究验收尚未通过。** 已有真实 CARLA 训练、逐帧控制审计和公开模型复现；尚不能宣称分层优越、稳定 PPO 增益、实时性或实际道路安全。

最新seed22 H恢复后累计保留30,723个训练控制步/970次Adam更新，但完整known24验证为**23/24**，共同零PPO基线为**24/24**；多506个控制步、485次干预、49次规划失败，新增一个行人横穿条件超时。新段与端点raw/engine审计、全部起点及静态NN复算通过，祖先日志的不完整尾部仍单列披露。H0seed22已开始真实训练，目前仍仅一个独立H/H0配对完全闭合。[全部逐条件结果与记账更正](../reports/native_seed22_H_known24_complete_20261105.json)。这项结果不支持PPO收益主张。

[二维A*/Hybrid空间组件对照](spatial_planner_component.md)已修复二维搜索未执行节点上限的问题，并用公开输入复现全部16个静态/弯道查询。它与原生B2驾驶及贡献结论分开，不能替代正式消融。

首个独立热启动 **H/H0配对已全部闭合**：实际CARLA训练30773/30752控制步、各970次Adam更新，全部48个TRAIN条件及原始帧/引擎/保存状态核验通过。两组已知24条验证均24/24，与共同零PPO基线持平；H/H0评估控制8165/8096、干预408/364、规划失败7/3，这个种子未证明新增失败反馈的收益。实际初始参数/Adam/Torch RNG及保存RNG内容相同，H0只移除新增负失败分支。[完整配对及全部结果](../reports/native_independent_outcome_seed11_pair_complete_20261104.json)、[冻结流程](independent_outcome_cohort.md)。种子22已开始真实PPO更新；其余4对与最终确认仍待完成。

五种子B1配对队列已准备好，**正在等待当前H/H0队列完整结束，尚无新增B1原生训练**。五份保存优化配置一致，15组基础/TRAIN/验证输入预检通过。新增PPO窗口和共同教师交互量匹配，继承监督与历史开发消耗单列；实际学习后的基础驾驶合格检查仍待执行。[准备证据与边界](../reports/native_matched_flat_cohort_ready_20261104.json)、[运行流程](matched_flat_cohort.md)。

动作对齐反馈配对实验已全部关闭：**H/H0实际训练控制10482/10489、各330次Adam更新**，初始权重/完整Adam/RNG/RMS一致，低层固定。两组在相同两个已知条件均 **2/2成功**，未显示新增反馈的完成率优势。H/H0评估控制911/1014、干预1/34、规划失败0/3；这些成本差异只是单种子探索性证据。H两场决策时失败历史均为空，不能声称新失败项在复测中直接纠正动作。[完整配对、实际成本与每场耗时分位数](../reports/native_outcome_context_pair_complete_20261104.json)。两组完整30摆放及24已知验证清单也已全部闭合，零新增PPO、零最终条件。

[完整H/H0/B3开发预发布](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/tag/v0.4.0-dev.20261104-outcome-shared-control)现已公开5个证据包，保留**56688个评估控制步**。5个ZIP均已未登录下载，**3868个文件全部哈希一致**。独立干净工作树已重建H/H0全部四套清单：108个测量起点、3207个静态NN决策、四张逐字节一致图表及原始帧/引擎回执；B3两套原始执行证据也已重建。完整重建凭证已上传，新增JSON也已未登录下载核验。[公开链接、哈希、复现命令与边界](../reports/native_outcome_shared_public_release_20261104.json)。这是已知开发端点的复现，不能当作全部祖先训练来源重建或最终确认。

共享低层的强规则对照已全部完成，原始帧与引擎控制回执均通过核验。三个方法都完成 **30个静态摆放＋24个已知验证条目**，零碰撞/越界：

| 方法 | 成功数 | 实际评估控制帧 | 干预 | 规划失败 |
|---|---:|---:|---:|---:|
| H | 54/54 | 18972 | 561 | 18 |
| H0 | 54/54 | 18902 | 567 | 18 |
| 强规则B3 | 54/54 | 18814 | 384 | 0 |

[完整配对、全部条件和实际计算成本](../reports/native_shared_lowlevel_H_H0_B3_complete_20261104.json)。**没有PPO或新增反馈的完成率优势**，B3的干预和规划失败更少。学习策略选择本身更便宜，完整周期同时包含所选动作带来的规划/控制代价；P95/P99/max及超周期比例全部保留。当前只有一个继续训练种子和已知开发条件，不是独立最终确认。首次B3构造器失败发生在控制前，原记录保留，修复后另建不可覆盖的新批次完成两组评估。[实际执行与清理证据](../reports/native_shared_lowlevel_B3_execution_20261104.json)。同权重反馈探针仅在1086个静态决策中改变2次选择，未执行替代动作，不能当作驾驶收益。[机制记录](../reports/native_outcome_failure_term_action_influence_20261104.json)。

更早的开发版本（2026-11-04）：固定 option 接口的继续训练策略在已知静态摆放为 **21/30**，已知验证为 **22/24**，全部原始控制及策略推理已核验；低于此前另一套配置的23/30、24/24，不能宣称改进。[完整记录](../reports/native_option_full_development_complete_20261104.json)。后续速度与转向联合重定时修复了一个原地停滞条件（558个真实控制步成功），但预登记的五个已知条件合计仅 **2/5** 成功；不是新增PPO学习或独立种子优势。[成功与失败证据](../reports/native_joint_speed_four_condition_complete_20261104.json)。该重定时版本全量CPU测试1392项通过，CARLA审计另行提供。

随后在固定低层下实际进行 **5438 个 PPO 控制步、170 次 Adam 更新**，两场已知条件冻结复测由0/2变为1/2，但新增一次越界，安全目标未通过。[完整学习与安全退化](../reports/native_exact_point_PPO5120_complete_20261104.json)。参考中心MPC代价的冻结迁移仍为1/2，原越界改为超时，没有完成率增益。[控制器对照](../reports/native_mpc_reference_effort_transfer_20261104.json)。该MPC源码全量CPU测试1402项通过；不能把这些单策略、已知条件开发结果当作独立种子优势或研究验收完成。


最新完成的 B1 复习课程从既有真实 CARLA 策略开始，模型、Adam、RMS 和 RNG 初始字节全部保持一致，新增 **10,240 个实际 PPO 控制步、320 次 Adam 更新**，没有新增监督。基础道路 **4/4**，24 个已知开发条件 **15/24**：弯道/静态/行人分别 6/3/6（各8场），7次越界、2次碰撞。同预算但没有复习原基础道路的课程在基础测试为3/4。复习改变的是 TRAIN 暴露顺序和重复次数，没有增加独立场景数量。这只是单种子开发证据。[完整训练与验证审计](../reports/native_B1_matched_rehearsal_complete_20261102.json)。

H/H0真实适配已完成，24个已知开发条件均为 **24/24，零不安全回合**；各保留330次新增Adam更新，新增保留控制分别10467/10453。H0恢复时至少1643个已消耗控制和50次Adam未保留，未知尾段继续披露。相同成绩尚不能证明反馈收益。[完整核验](../reports/native_H_H0_diverse_recovered_complete_20261103.json)。

旧12场静态TRAIN对照现已补齐H/H0：H为 **3/12**（8超时、1越界），H0为 **4/12**（7超时、1越界）。H0只组合无重叠的已核验完整回合，模型、归一化及物理配置一致，损坏文件与未知消耗继续保留，不重跑已核验回合；零PPO、行为克隆初始化的教师迁移策略也已完成：**5/12**（6超时、1越界、7339控制）。该教师是神经网络模仿策略，与直接规则B3不同。[三策略完整核验证据](../reports/native_static_H_H0_teacher_complete_20261103.json)。单种子结果不证明反馈收益。[完整恢复证据](../reports/native_static_H_H0_recovered_complete_20261103.json)。共同越界案例的高层目标保持稳定、实际状态与命令逐位一致；离线地图和只读引擎射线进一步发现栅格、车道谓词与地面语义存在分歧。原失败、评分与当前判定标准均未更改。[几何证据及局限](../reports/native_static_shared_road_disagreement_20261103.json)。参考上下文 H/H0 已完成 **21024 个真实训练控制、各330次Adam更新**，RMS保持一致。完整评估及审计已结束：困难静态TRAIN12为 **H 3/12、H0 4/12**，另外的已知验证24条件两者均 **24/24**。全部32060个评估控制帧、各实测起点、静态NN动作及发动机回执均核验通过；静态成功数没有改进，单seed开发结果不证明反馈收益或最终验收。[完整学习及历史失败成本](../reports/native_reference_context_learning_complete_20261103.json)、[完整端点证据](../reports/native_reference_context_endpoints_complete_20261103.json)。

[参考上下文研发预发布](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/tag/v0.3.0-dev.20261103-reference-context)已公开两个关闭端点的数据包、图表和校验值。8项资产均无需登录下载成功，并从新checkout完整重建：1254文件、32060控制帧、72起点、2611静态NN决策及2张逐字节一致图表。[公开下载与复现凭证](../reports/native_reference_context_public_download_verification_20261103.json)。

仅H的scale100已全部结束：**10501真实训练控制/330次Adam，静态3/12、已知验证24/24**。16293个评估控制帧、36起点和1330个静态NN决策均通过原始帧/推断/控制回执核验，静态成功数没有改善。[完整端点](../reports/native_reference_scale100_endpoints_complete_20261103.json)。熵0.02课程也已关闭：**10485真实训练控制/330次Adam、静态4/12、已知验证24/24**。15745评估控制帧、36起点、1276静态NN决策全部核验通过；单seed已知开发集多成功1场，尚不构成普遍或因果优势，静态目标仍未达成。[完整结果](../reports/native_reference_entropy002_endpoints_complete_20261104.json)。

另一个独立注册的模型可行摆放开发清单中，同一冻结PPO为 **23/30，越界2/30**，12402控制和1252NN均核验通过；这是场景条件变化后的迁移，原4/12不改，不归因于学习。[完整清单](../reports/native_model_supported_placements_complete_20261104.json)。

动作结果反馈源码的完整本地CPU回归已结束：**1411通过、1跳过、4项CARLA排除**。[版本化初始化与CPU凭证](../reports/native_outcome_context_initialization_20261104.json)。[对应GitHub CPU工作流](https://github.com/Leo-7GOAT/Intelligent-driving-system/actions/runs/41791653607)也已成功。实际CARLA训练和驾驶审计分别提供。

[五种子 B1 扩展预发布](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/tag/v0.2.0-dev.20261102-broader-b1)的69项资产已公开下载重建：151,660个控制帧、120个初始动作及5张图表均核验一致。[公开复现凭证](../reports/native_B1_broader_public_download_verification_20261102.json)。教师、恢复采集和失败的离线观测修正均已结束，不要重复启动。

## 当前证据

以下实验具有不同训练历史与接口版本，应分别解释。

| 实验 | 实测结果 | 结论边界 |
|---|---|---|
| 同执行版本 H/H0，五个独立训练种子 | 在 24 个已知开发条件上 H 120/120、H0 119/120，零碰撞/越界 | 平均差 +0.83 个百分点，95% 区间 [−1.48, +3.15]，尚无稳定反馈收益；不是 120 个新条件。[完整统计](../reports/native_matched_broader_feedback_cohort_20261101.json) |
| 教师初始化与追加 PPO 对照 | 教师 H 117/120、PPO H 120/120；强规则 B3 24/24 | 差值 +2.50 个百分点，95% 区间 [−2.13, +7.13]，尚无稳定 PPO 新增收益。[归因报告](../reports/native_broader_PPO_teacher_rule_attribution_20261101.json) |
| 共享停车辅助的冻结迁移 | B1/H/B3 在同四条已知训练道路各 4/4 | 属于共同低层工程能力，不是新增学习；全部辅助控制均披露。[工程审计](../reports/native_shared_parking_engineering_20261101.json) |
| 全程固定停车辅助的单种子 B1 PPO | 新增 10,240 个真实训练控制步、318 次 Adam 更新，基础端点 4/4 | 初始模型已为 4/4，只证明能力保留。[学习审计](../reports/native_B1_shared_parking_PPO_20261101.json) |
| 同一冻结 B1 的原有完整条件复验 | 12/12，三类各 4/4；4,848 帧控制核验 | 已知开发条件，不是独立最终测试。[报告](../reports/native_B1_shared_parking_full_probe_20261101.json) |
| 同一 B1 的不同道路开发验证 | 14/24：弯道停车 5/8、静态 3/8、行人 6/8；8 次越界、2 次碰撞 | 10,210 帧控制已审计。第 17 条后发生行人生成异常，仅补齐 7 条未完成条件，混合初始化版本明确保留；泛化不足。[完整记录](../reports/native_B1_shared_parking_validation_complete_20261101.json) |

五个 B1 基础课程已全部完成：种子 11/22/33/44/55 分别为 **3/4、4/4、4/4、4/4、3/4**，两次失败均为越界，因此只有 **三个种子通过基础门槛**。保留的新增学习控制步共 51,200，Adam 更新共 1,599；这是重复使用四条已知训练道路，不能算作 20 个新条件或健康完整基线。[全部结果与成本](../reports/native_B1_shared_parking_complete_basic_cohort_20261102.json)。

第五个种子包含明确披露的中断恢复：3,840 个已保存控制步加 6,400 个续训步。报告的训练消耗至少 51,858 步，其中 51,837 步具有可读桥接回执，另 21 条仅有策略记录，未知尾段继续披露。seed22 扩展试验现已完成：新增 20,480 步真实训练、630 次 Adam 更新；验证为 7/24（弯道 5/8、静态 0/8、行人 2/8），11 次越界、5 次碰撞、1 次超时，未通过开发门槛。31,480 帧控制全部核验，归一化统计不变。这与此前 14/24 的初始模型不同，不作因果增益比较。[完成审计](../reports/native_B1_broader_seed22_20261102.json)。[恢复凭证](../reports/native_B1_seed55_recovery_registration_20261102.json)、[扩展登记](../reports/native_B1_broader_shared_parking_registration_20261101.json)。

真实测量输入诊断确认反馈进入 H：同时隐藏两类反馈会改变 446 个抽样动作中的 11 个；H0 的 448 个样本概率完全不变，全程保持原物理掩码。这不能证明反馈改善驾驶。[诊断与复现](native_feedback_use.md)。

在当前 seed22 扩展试验的验证结果产生前，其余四个种子已按同配方各登记追加 20,480 步，包含基础门槛失败的两个种子。冻结源码导入、配置、PPO 参数和启动器检查全部通过；seed11 也已完成并审计：11/24 成功（三类分别 4/8、4/8、3/8），5 次超时、5 次越界、3 次碰撞，仍未达标。seed33 已完成：11/24（三类 6/8、0/8、5/8），7 次越界、6 次碰撞，仍未达标；seed44 已恢复完成：15/24（三类 6/8、4/8、5/8），4 次碰撞、3 次越界、2 次超时；仍未达标。seed55 已完成：10/24（三类 5/8、2/8、3/8），7 次碰撞、7 次越界，仍未达标。[seed33 完整审计](../reports/native_B1_broader_seed33_20261102.json)。[登记、资源边界与报告规则](broader_B1_cohort.md)。本轮五种子全部完成：54/120 次成功（45%），58 次碰撞或越界，五个开发门槛均未通过；这是 24 个已知条件的重复评测，不是 120 个新最终条件。[完整汇总与成本](../reports/native_B1_broader_complete_cohort_20261102.json)。seed22 审计已在独立检出目录利用本地原始证据逐字段重建一致；本轮模型、日志与中断前缀现已通过页面上方的69项预发布资产公开；无认证下载重建已通过，完整凭证见页面上方链接。[核验凭证](../reports/native_B1_broader_seed22_independent_verification_20261102.json)。

seed44 因磁盘保护在保存 11,520 步后中断，已从原模型、Adam、归一化和随机数状态续完剩余 8,960 步，保留本轮 20,480 步和 640 次 Adam 更新。旧产物经逐文件哈希校验迁移至 D 盘，原路径通过目录链接保留；后续教师采集和拟合已接到恢复队列。此前四种子 44/96 的部分记录仍保留。[部分队列汇总](../reports/native_B1_broader_partial_cohort_20261102.json)。[中断与恢复凭证](../reports/native_B1_broader_seed44_interruption_20261102.json)。

停车辅助仅在车辆已进入原目标位置/朝向范围、速度仍超限时请求真实制动；不改转向、车辆状态或成功阈值，解除接管后才能成功。各方法可共享同一版本。[实现、成本与实验](shared_parking.md)。

## 安装、测试与复现

Python 3.11 使用项目隔离环境；已有环境无需重建。CARLA 0.9.15 沿用单独的兼容 Python 3.7 客户端。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pytest -q -m "not carla"
```

从仓库根目录运行。CPU 测试包括动力学、规划、MPC、真实 PPO 更新与恢复；跳过的 CARLA 测试不能算作真实驾驶通过。[桥接与标定](CARLA_bridge.md)。

[英文首页](../README.md#reproduce-a-published-trained-model-and-raw-results)给出公开下载、模型加载和原始结果重建的完整命令。真实 PPO 证据包已从独立目录公开下载并重建：572 个文件、11,639 帧控制、四个初态动作、图表及 Adam/RMS 学习审计全部一致。[执行凭证](../reports/native_B1_shared_parking_PPO_public_verification_20261101.json)。另有下载模型后的真实 CARLA 闭环复验，见[停车实验说明](shared_parking.md)。

## 尚未完成的原目标

- 五个独立训练策略、通过基础学习检查且实现与训练充分的 B1，以及公平的 H/H0/B1/B2/B3、频率、统一安全和固定参考对照。
- 至少 120 个新的去重配对最终条件，三类各至少 30 个并包含未见布局或车辆参数；现有开发集不充当最终确认。
- 轻量安全完成率总计 ≥85%、各类 ≥75%；CARLA 总计 ≥80%、各类 ≥70%，碰撞或越界回合比例 ≤5%。
- H 相对健康 B1 平均提高 ≥10 个百分点且配对 95% 区间下界 >0，或最终测试前锁定样本效率替代假设；另需证明 PPO 相对强规则的价值和可解释协作收益。
- 记录全部失败、训练/监督/算力成本及接管；可复现发布并保持结论与证据一致。目前公开的是开发成果。

已有原生队列不得重复启动。恢复前须将状态文件中的进程身份与操作系统核对。[当前接续状态与执行入口](handoff.md)、[验收台账](research_acceptance.md)、[失败分析](failure_analysis.md)、[中英文项目描述](resume_project.md)。

决策周期仍有超时，同步仿真能运行不代表实时控制。无高速、量产或实车能力结论。[历史开发快照](README.zh-CN.development-history.md)、[MIT 许可](../LICENSE)、[第三方说明](THIRD_PARTY_NOTICES.md)。

独立的36条TRAIN教师采集已结束，完整成功、失败和额外监督成本均保留；教师结果不等于PPO成绩，也不包含最终条件。[实际采集证据](../reports/native_B3_diverse_TRAIN_teacher_collection_20261102.json)。

2026-11-04开发进展：冻结PPO、保持原场景及越界口径，加入实测初始转角与有转向速率约束的Hybrid A*运动基元后，预注册case1/case4均成功（333/340控制步）。673控制、68次NN、2起点和全部参考重建/发动机回执通过。此前仅道路修正的case4越界失败保留。该两场工程迁移不证明PPO学习收益或整体验收达标，原完整静态4/12不重算。[证据](../reports/native_steering_transition_pilot_20261104.json)。

# 可执行性感知分层自动驾驶研究

[English README](../README.md) · [验收记录](research_acceptance.md) · [方法](design.md) · [GitHub 预发布](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases)

研究主链路为 **PPO 高层决策 → 局部目标与速度 → Hybrid A* → 时间轨迹 → MPC → CARLA 实际状态与奖励反馈**，面向已知地图下的低速车辆任务。

**目前是开发预发布，整体研究验收尚未通过。** 已有真实 CARLA 训练、逐帧控制审计和公开模型复现；尚不能宣称分层优越、稳定 PPO 增益、实时性或实际道路安全。


最新状态：[五种子 B1 扩展预发布](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/tag/v0.2.0-dev.20261102-broader-b1)已公开，69 项资产的远端哈希全部一致。独立环境无认证下载重建已通过：151,660 个控制帧、120 个初始动作、5 张逐字节一致的图表，以及 PPO/Adam/归一化和完整汇总均重建一致。[公开复现凭证](../reports/native_B1_broader_public_download_verification_20261102.json)。随后每种子追加 1,024 次 TRAIN 教师监督的 actor Adam 更新；固定 seed22 基础道路4/4，已知验证仅4/24，仍未达标。[监督与闭环验证完整记录](../reports/native_B1_broader_nominal_supervision_20261102.json)。正在原36个 TRAIN 条件采集纠偏数据；不使用验证标签训练。

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

- 五个独立训练策略、健康完整 B1，以及公平的 H/H0/B1/B2/B3、频率、统一安全和固定参考对照。
- 至少 120 个新的去重配对最终条件，三类各至少 30 个并包含未见布局或车辆参数；现有开发集不充当最终确认。
- 轻量安全完成率总计 ≥85%、各类 ≥75%；CARLA 总计 ≥80%、各类 ≥70%，碰撞或越界回合比例 ≤5%。
- H 相对健康 B1 平均提高 ≥10 个百分点且配对 95% 区间下界 >0，或最终测试前锁定样本效率替代假设；另需证明 PPO 相对强规则的价值和可解释协作收益。
- 记录全部失败、训练/监督/算力成本及接管；可复现发布并保持结论与证据一致。目前公开的是开发成果。

已有原生队列不得重复启动。恢复前须将状态文件中的进程身份与操作系统核对。[当前接续状态与执行入口](handoff.md)、[验收台账](research_acceptance.md)、[失败分析](failure_analysis.md)、[中英文项目描述](resume_project.md)。

决策周期仍有超时，同步仿真能运行不代表实时控制。无高速、量产或实车能力结论。[历史开发快照](README.zh-CN.development-history.md)、[MIT 许可](../LICENSE)、[第三方说明](THIRD_PARTY_NOTICES.md)。

另已登记独立的 [36 条 TRAIN 教师采集](broader_TRAIN_teacher.md)，待当前队列结束后使用同步执行接口和完整强 B3 配置采集，再重建直接控制样本。旧 H/H0 TRAIN 日志按完整成功回合及执行回执严格筛选仅覆盖 31/36，未生成标签或拟合模型。新采集不含开发验证或最终条件，额外监督与物理成本单列。

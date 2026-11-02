# 当前接续状态（2026-11-02 20:40）

继续当前仓库，分支 `feature/physical-hierarchical-ppo`。不重建项目、不初始化 Git、不恢复瞬移环境。完整研究验收仍未达成；不能把基础4/4、教师成绩、训练结束或已知开发集当成最终达标。

## 目标

主方法保持 PPO 高层 → 局部目标/速度 → Hybrid A* → 时间轨迹 → MPC → 真实 CARLA 状态与奖励反馈。需要健康五种子 B1、H/H0/B1/B2/B3 公平配对、独立 PPO 与反馈贡献、至少120个未查看的新最终条件及完整公开复现。现有已知24场仅用于开发。用户授权自主修改、测试、真实本地训练、提交和推送，不逐项询问。

## 当前实际作业：只观察，禁止重复启动

- 所有路线误差离线拟合、低率及高率PPO和全24验证均已完成/正常清理。高率最终4/24（低率5/24），原10k无新增FN策略作为下一次训练起点；失败新观测分支不继续追加预算。
- **唯一当前物理作业**：`runs/20261102T123912910706Z_native_canonical_parent_TRAIN48_PPO10240`，由 `scripts/run_strong_native_parent_course.py` 启动。训练协调器 **63804 / session71428**；execution `runs/B1_canonical_native_parent_PPO10240_execution_20261102`。固定10240新实际控制、3e-6学习率、原KL保护、TRAIN48，不新增监督。
- 训练后审计/端点协调器 **60904 / session10083**，execution `runs/B1_canonical_native_parent_PPO10240_postprobe_20261102`。它哈希绑定 `scripts/complete_route_feedback_ppo.py` 和 `scripts/audit_route_feedback_ppo.py`，期间不要修改。等待确切63804退出0，审计全部10240帧/35个原Adam状态/RMS，然后原4基础道路，达门槛才测24已知验证。
- 初始化为原生40960-PPO控制模型29efb9c…，来自 `runs/20261101T112758733226Z_native_B1_shared_parking_PPO/B1/checkpoint_000010240`。模型/RMS/优化器/RNG原样复制、观测4705维但关闭失败FN新增字段；与当前车辆标定和配置signature完全一致。历史14/24存在混合初始化，仅作为历史开发结果，不当作当前健康得分。原教师/位姿/反射增强监督历史保留并披露，当前零新增监督。
- 当前物理源采用已验证逐位等价的root观测目录复用与SAT窗口优化；其余物理控制、碰撞、奖励、桥接与当前模板一致。没有恢复旧模拟环境。实际状态以status/complete/failure为准，不重启。

## 本轮已经完成

- 新增路线误差输入及2758参数直接控制网络残差，只有已知地图和实测自车输入，无运行时教师、规则跟踪器或状态投影。4096次 TRAIN 辅助拟合后基础4/4，但已知验证 **1/24**：10超时、10越界、3碰撞，11528实际帧全审计。模型558f162f…，作为健康基线明确否决。见 `reports/native_B1_route_feedback_failed_validation_20261102.json`。
- 预先锁定的真实 PPO 短训练已结束：`runs/20261102T114745271816Z_native_route_feedback_PPO5120`，**5120新控制步、160 Adam、495重置/5615世界tick、418.624秒**。原35个及新增5个参数状态均实际增加160步，actor/critic均变化，原RMS统计一致，所有控制与发动机回执验证。TRAIN48交错16道路/三类，不保证短预算遍历全部条件。继承学习率3e-6、std约0.018315，没有中途调参。模型 **5085cfb491ebf1c07a14dcfc71f0c6cb5673476fd464ff28286fc20bcbb6e11d**，norm **d9c78f13a51ac235750264631f4138dc273b96fe78d864f169b0535dd1774d5f**。见 `reports/native_B1_route_feedback_PPO5120_training_20261102.json`。
- 上述模型训练后基础 **4/4**，347/357/308/308=**1320**实际控制，全部NN动作重建一致；batch `20261102T115516528316Z_native_route_feedback_PPO_basic`。完整24已知验证也已完成：**5/24**，弯道2/8、静态3/8、行人0/8；8越界/7超时/4碰撞，9641帧及24个起点/发动机回执全部核验。相较离线1/24有开发集改善，仍不健康。见 `reports/native_B1_route_feedback_PPO5120_complete_validation_20261102.json`。
- 同一模型已在原有独立 Git checkout、原有干净环境中载入，40个PPO Adam状态完整，4个实测起点按生产batch1推理逐位一致。证据 `runs/route_feedback_PPO5120_clean_inference_20261102/complete.json`。初次batch4对比有1.19e-7浮点差，诊断保留；没有放宽batch1标准。这不是公开下载复现。
- 本地完整CPU1230通过/1跳过；之后新增Adam/清单8项通过，SAT/规划13项通过。源码 **1f0975be75eed4b522d134142335470ad9fd8409** 已推送，GitHub CI **41440795733 SUCCESS**。
- SAT仅先取同样的局部闭合栅格再执行原判据，13项相关回归通过；8实测起点输入完全一致，观测平均94.5→82.3ms。只读微基准，不是实时性或驾驶性能证明。见 `reports/native_observation_spatial_window_benchmark_20261102.json`。

## 不能重跑或混淆的已完成结果

- 五种子更广B1：54/120，58不安全，8超时，只有24个已知条件；102400新增真实PPO控制/3148Adam。公开release `v0.2.0-dev.20261102-broader-b1`，69资产公开下载复现已完成151660帧、120起点、5图。不得再次下载或训练旧队列。
- 原TRAIN教师35/36；恢复混合采集12697控制/6179合格实际教师标签；新增12真正不同基准道路的教师31/36/14479控制均完成。所有失败保留。串/并行数据重建全数组一致；并行225秒、串行1777秒，非政策性能提升。
- 前一几何平衡拟合模型5b4311f…：基础4/4，已知验证3/24，12383帧，不能称健康。更早同数据长拟合16ecd9…为2/24；其他离线变体均失败，不再盲目重复同数据拟合。
- 五个H/H0配对：120/120 vs119/120，差值区间跨0；PPO vs教师117/120的净收益区间也跨0。不能宣称反馈或PPO优势已证明。强B3已知24/24。

## 单变量学习率对照（不重复旧训练）

`runs/B1_route_feedback_rate1e4_PPO5120_execution_20261102`，协调器 **67268 / session80592**，`python scripts/queue_route_feedback_ppo.py --rate-profile` 已在当前完整验证结果出来前注册并启动，等待确切13124退出0。

仅将学习率3e-6改为1e-4，预算同为5120新真实PPO控制；从同一个离线模型558f162f…重新开始，保留原35个Adam状态、新增5个空组、log_std、RMS、RNG和完全相同TRAIN48/低层/奖励。5120低率分支及后续评估成本全部计入消耗，不冒充权重保留。不是从低率训练后的5085继续；这是匹配起点的单变量诊断，不是多种子结论。初始化已逐张量验证权重/探索与Adam不变，仅调度率变化；前五个实际重置也一致。3项相关回归通过。

不要重启高率协调器。固定训练已经完成，后续仍由54584完成 `audit_route_feedback_ppo.py` 及原4基础/条件性24验证。现有 `complete_route_feedback_ppo.py` 仍被13124哈希绑定；13124已结束；接口已增加明确execution/dependency参数，高率作业的新审计/端点评测协调器54584已启动。

## 后续必须执行

当前完整验证结束后先据真实结果更新报告、README及本文件，保留失败和实际成本。若仍失败，不把本轮离线监督或低学习率短训练包装成健康充分基线；用证据决定下一项必要修改或合理PPO调参。完整研究目标未达成，不标记goal complete，也不将算法失败当作外部阻塞。

CARLA沿用 `D:/CARLA_0.9.15/WindowsNoEditor` 和原Python3.7客户端；不触碰其他项目进程。C盘旧runs/prerelease等有目录junction指向D盘，禁止递归删除/移动。只管理经PID/创建时间/可执行路径验证的本任务进程。旧只读等待器47232的单独取消与后续澄清已保留：本次源两组原率实际均3e-6，预防性审计兼容处理并非真实源错误；物理训练没有被打断。

完整历史原文已逐字节校验归档：[历史 handoff](../reports/handoff_before_route_feedback_PPO_20261102.md)。不要把历史时态覆盖上述当前状态。

当前源码f876eb0的GitHub CI41442770659已成功：1222通过/18按平台跳过/4排除。root候选目录复用新增18项检查通过，另24个已知验证实测起点的4705维输入和实际NN动作全部逐位一致；只读基准均值77.47→70.20ms，报告 native_observation_catalog_and_spatial_parity_20261102.json。不是实时性证明。

高率完整结果见 reports/native_B1_route_feedback_rate_contrast_complete_20261102.json。10240课程的审计器已显式支持无新增参数的35状态延续；10项相关检查通过。新源码待提交；a55ab48的GitHub CI41444605514已成功。

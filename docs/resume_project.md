# Lingzhou｜基于 PPO、Hybrid A* 与 MPC 的分层自动驾驶决策与控制研究

更新日期：2026-11-06

## 中文简历正文

- **分层方法：** 构建“PPO 局部目标／速度决策 → Hybrid A* → 时间参数化轨迹 → MPC”框架，在 CARLA 0.9.15 中形成基于实际车辆状态反馈的物理闭环，将高层驾驶意图与底层可执行运动衔接。
- **规划与执行：** 在规划中考虑车辆运动学、车体碰撞及转向约束，通过轨迹时间重整与 MPC 完成跟踪；将规划失败、跟踪偏差和执行干预信息反馈至高层策略，并开展执行反馈消融实验。
- **对照实验：** 完成五组学习种子的训练与评估；在相同24个已知开发条件的五组重复测试中，分层系统完成118/120回合（98.3%），固定预算直接控制 PPO 完成38/120回合（31.7%）；结合去反馈版本与规则高层对照分析收益来源。
- **实验复现：** 保存模型、Adam 优化器、观测归一化及随机状态，建立逐帧动作与仿真执行核验流程；公开模型与40个实验数据包，完成下载复现及逐帧执行核验，源码 CI 1497项测试通过。

## English CV summary

**Lingzhou — Hierarchical Autonomous Driving with PPO, Hybrid A* and MPC**

Implemented a physical CARLA 0.9.15 feedback loop combining PPO local-goal and speed decisions, Hybrid A* planning, trajectory retiming and MPC tracking. Across five learner seeds evaluated on the same 24 known development conditions, the hierarchy completed 118/120 episodes (98.3%), compared with 38/120 (31.7%) for fixed-budget direct-control PPO. Conducted feedback and rule-based high-level comparisons, published models and 40 evidence capsules, verified downloaded reconstruction and frame-level execution, and passed 1,497 source CI tests.

## 实验解读与使用范围（供面试准备）

使用边界：24个唯一已知条件；flat基础筛查仅2/5通过，seed22 H有恢复不确定性；热启动前120/120、H0 120/120，新增PPO/反馈收益未建立；120个新最终条件及完整研究验收未完成。不能写成“独立最终测试提升”或“已证明PPO自身优越”。[五组证据](../reports/native_five_pair_comparison_20261105.json)。下方为历史版本。

# Evidence-bounded project descriptions

This pilot's public reconstruction is complete:563files/12,219controls/4starts,byte-identical figure,exact initial intervention and320added Adam steps in all35PPOparameter states;observation RMS unchanged. Full local CPU:1,101pass;latest completed hostedCI:1,084pass. Overall research acceptance remains UNMET. [Public verification receipt](../reports/native_B1_exploration_PPO_public_verification_20261101.json).

Completed actual-PPOpilot:10,240new controls,320actual Adam updates,2/4pure endpoint with1bounds/1deadline. All12,219frames audited;model rejected,owned lifecycle terminal. The preceding activation is a historical snapshot. [Complete result](../reports/native_B1_exploration_PPO_pilot_20261101.json),[public reproduction](native_B1_exploration_reproduction.md). Overall acceptance remains UNMET.

Later original-TRAINrecovery and its projection ablation also fail;all9,010
recovery-profile controls are audited. The later real-PPOexploration pilot also fails2/4;it supports no superiority claim.

Latest B1recovery experiment remains negative:1,612TRAINmixture controls,
512auxiliary Adam proposals,then1,657pure-policy controls;2/4with timeout and
bounds. No healthy-baseline,added-PPOor final-confirmation claim is supported.
[Actual result](../reports/native_B1_executed_recovery_collection_and_probe_20261101.json).

Verified development wording — 2026-11-01,complete matched cohort:

- 中文：构建已知地图下的低速车辆PPO分层决策、Hybrid A*规划与MPC控制系统，接入CARLA真实PPO更新、车辆标定及逐帧执行审计。完成同执行版本五种子H/H0开发消融，在24个已知开发条件上分别120/120与119/120；公开模型、归一化、原始日志及可独立重建的统计和主图，核验287,811控制帧。CPU CI1036项通过。反馈差值95%区间跨零，健康非分层基线及独立最终确认尚未完成。
- English: Built a known-map,low-speed system combining PPO decisions,Hybrid A* planning and MPC,with real CARLA PPO updates,vehicle calibration and frame-level audits. Completed a common-harness five-seed H/H0development ablation:120/120versus119/120across24known conditions. Published models,normalizers,raw logs and independently reproducible statistics/figures,verifying287,811control frames;1,036CPU CItests pass. The feedback interval includes zero;a healthy flat-PPO baseline and independent final confirmation remain unestablished.

The [matched report](../reports/native_matched_broader_feedback_cohort_20261101.json)
has a paired mean+0.83pp,95%tinterval[-1.48,+3.15]pp. The
[public reconstruction](../reports/native_public_matched_broader_feedback_five_seed_verification_20261101.json)
retains all five trained pairs and24unique known conditions;120policy-case
results are not120new final conditions. The separate H/B2study retains its
bridge-initialization confound. No superiority,safety certification or real-vehicle
claim is made. Teacher-only attribution is117/120versus trainedH120/120,with a zero-crossing95%interval;the99label B1probe fails2/4with two bounds violations and the executed-label repair3/4with one.

Historical verified wording (2026-10-27):

- 中文：实现 PPO→局部目标/速度→前向 Hybrid A*→时间轨迹→MPC→CARLA 实际反馈闭环；单种子真实 PPO 训练20,541控制步后，在四条已知道路组成的12组开发条件中完成11组，25,454训练及评估控制帧逐帧审计。公开发布并独立复现三组模型/规则证据包的18,591控制帧、36个起点策略动作和两张结果图；GitHub CPU回归511项通过。规则对照12/12，健康完整B1和独立研究优势尚未建立。
- English: Built a PPO→local pose/speed→forward Hybrid A*→timed trajectory→MPC→measured CARLA feedback loop. A single-seed policy completes11/12known-road development conditions after20,541real PPO training controls, with25,454training/evaluation controls audited frame by frame. Published and independently reproduced18,591controls,36measured-start policy actions and two figures across three evidence capsules;511hosted CPU tests pass. The strong rule achieves12/12; a healthy full-task flat baseline and independent superiority remain unestablished.

The earlier descriptions below refer to their respective archived batches.

## 中文

**基于 PPO、Hybrid A* 与 MPC 的低速车辆分层决策与控制研究**

- 将原有离散演示重构为真实物理执行闭环：PPO选择局部目标与速度，前向Hybrid A*检查车辆运动学和车体碰撞约束，MPC逐周期求解并反馈实际车辆状态。
- 实现直接控制PPO、二维A*规划消融、规则高层、执行反馈消融及按物理交互量统计的训练评估流程；保存模型、归一化状态、配置、原始动作和失败记录。
- 接入本地CARLA0.9.15，完成客户端桥接、车辆标定、同步帧与传感器核对，并运行真实PPO训练和冻结策略迁移；通过故障测试改进独占锁、进程清理和断点恢复。
- 建立可追溯开发实验与复现工具：单轮三方法67,289个真实CARLA控制步完成原始帧审计；发布570文件证据包并在独立检出中下载复现。CPU回归本机295项通过、GitHub293项通过，平台跳过项单列。分层优势与五种子独立确认尚未建立。

## English

**Low-speed hierarchical vehicle decision and control with PPO, Hybrid A* and MPC**

- Rebuilt a discrete demonstration into a physical closed loop: PPO selects local poses and speeds, forward Hybrid A* enforces vehicle geometry and collision constraints, and MPC solves each control cycle against actual state feedback.
- Implemented flat-PPO and planning/high-level/feedback comparisons, physical-interaction accounting, and traceable checkpoints with normalization, configurations, actions and retained failures.
- Integrated localCARLA0.9.15 with an isolated client bridge, vehicle calibration, synchronized state/sensor checks, realPPO updates and frozen-policy transfer; improved ownership locks, cleanup and recovery through fault tests.
- Audited 67,289 actual CARLA control steps across three methods in one development batch; published a 570-file model/evidence asset and reproduced its public download in an independent checkout. CPU regression passed295tests locally and293on GitHub, with platform skips reported separately. Five-seed superiority and generalization remain unproven.

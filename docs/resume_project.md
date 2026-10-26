# Evidence-bounded project descriptions

These descriptions intentionally omit unproven superiority, final success-rate targets, production safety and real-vehicle claims.

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

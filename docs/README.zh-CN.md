# 低速车辆分层决策与控制研究

主链路为 PPO → 局部目标/速度 → 前向 Hybrid A* → 时间轨迹 → MPC → 真实车辆执行反馈。CARLA0.9.15已实际接入并发生PPO更新；完整研究验收尚未通过，不能声称分层显著更优或具备实车安全能力。

最新[原生路界与PPO实测](../reports/indexed_native_geometry_20261022.md)：v3索引车道并集修复了已记录的接缝/路口误判。固定规划Hybrid A*与B2各完成4/4道路，但真实CARLA H/B1各约1万控制步后仍为0/4；全部26,470训练及评估帧已对上原始记录。轻量B1中间检查点能完成23米停车，后期又退化，最终500万步完整任务仍0/10。117项回归通过、5项条件跳过；新的H0/H/B1原生道路10万步对照已启动，完成状态以日志和操作系统核验为准。

v18已完成5,002,111物理步、47,170次优化器更新，最终完整开发集0/10，23米停车能力退化；归一化交换实验没有恢复停车。v13–v17的频率、结构、奖励和训练跨度诊断也未建立可靠的完整任务B1。失败和冻结版本全部保留，见[失败诊断](../reports/flat_diagnosis_20261022.md)和[能力保持实验](../reports/rehearsal_feasibility_20261022.md)。

H/H0的v8百万步开发结果为8/10、9/10，强规则B3为10/10；v12为8/10、6/10，实际CARLA约10万世界步适配后4/6、3/6。H在更早检查点曾5/6，退化未删除。现有数据不支持PPO高层必要性或稳定的反馈收益。

共享道路栅格已接入规划、全部策略观测、风险和终止；原生道路H/B1真实短训练、重载及独立检出复验已执行，但没有任务成功。新可选B2后处理对四条已保存道路均通过曲率及连续车体包络检查，不能仅凭空间规划成功声称驾驶成功。实际Hybrid A*搜索路径经MPC/CARLA执行为2成功、2道路边界判定失败，制动尖峰仍存在。

真实摄像头记录了PPO成功与超时案例，模型、场景和帧号可追溯，见[摄像头说明](CARLA_bridge.md#real-camera-development-evidence-2026-10-22)。当前验收状态见[台账](research_acceptance.md)，完整开发记录见[交接](handoff.md)，历史说明保留在[历史中文文档](README.zh-CN.historical.md)。

## 安装与CPU验证

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pytest -q
```

最新本地全量回归为 **111通过、5条件跳过**。跳过项不计为实测通过。原生地图、执行器和完整物理帧实测见[原生道路报告](../reports/native_road_learning_20261022.md)，不再将早期“仅完成几何调查”的历史状态作为当前状态。

## 开发模型复现

将开发预发布资产解压到`runs/`，得到`runs/development_v8_seed11/`后执行：

```powershell
.\.venv\Scripts\python.exe scripts/reproduce_development_asset.py --asset runs/development_v8_seed11 --output runs/reproduced --full
```

该命令核对哈希、使用随包冻结源码、加载PPO及归一化状态、重跑全部20个开发回合，并重建端点图。数据包括失败，不是未见最终确认。模型内容和限制见[模型卡](model_card.md)。

CARLA需要独立安装服务端和匹配客户端，相关要求见[桥接说明](CARLA_bridge.md)。不把条件跳过的测试称作CARLA验证。运行中的批次、恢复入口和停止条件见[交接记录](handoff.md)。

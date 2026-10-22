# 低速车辆分层决策与控制研究

主链路为PPO→局部目标/速度→前向Hybrid A*→时间轨迹→MPC→真实车辆执行反馈。项目接入了CARLA0.9.15，并已发生真实PPO更新；目前尚未满足完整研究验收，不能声称分层显著更优或具备实车安全能力。

当前结果见[研究结果](research_results.md)、[失败分析](failure_analysis.md)和[验收台账](research_acceptance.md)。历史说明保留在[历史中文文档](README.zh-CN.historical.md)。

最新单种子开发结果：轻量环境百万物理步后H为8/10、无新增反馈的H0为9/10；强规则B3在相同清单上为10/10。两种20万步来源模型分别进行约10万CARLA世界步真实微调后均为3/6。因此尚未证明反馈收益或PPO高层优势。新候选共享评分版本已独立从头训练，保留无反馈消融，不屏蔽动作或使用规则替代。旧版与新版输入维度不同，不能混用模型或直接作为公平主比较。

## 安装与CPU验证

后续 v12 候选共享评分版已完成各约百万物理步：H 为 8/10，H0 为 6/10。另行训练的 450 维混合任务 B1 为 0/10，不能当作健康的最终对照。新版 CARLA 约 10 万世界步适配后为 H 4/6、H0 3/6；H 在约 2.5 万步曾为 5/6，后续退步已保留。共同课程 v13 的 H/H0/B1 正在训练，预算包含无障碍预训练阶段，未获得正式确认结论。

真实摄像头已记录 PPO 静态绕障成功和弯曲路线超时两种结果；最新全量回归为 74 通过、5 条件跳过。详见[最新实验记录](../reports/camera_staged_20261022.md)和[摄像头说明](CARLA_bridge.md#real-camera-development-evidence-2026-10-22)。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pytest -q
```

## 开发模型复现

将开发预发布资产解压到`runs/`，得到`runs/development_v8_seed11/`后执行：

```powershell
.\.venv\Scripts\python.exe scripts/reproduce_development_asset.py --asset runs/development_v8_seed11 --output runs/reproduced --full
```

该命令核对哈希、使用随包冻结源码、加载PPO及归一化状态、重跑全部20个开发回合，并重建端点图。数据包括失败，不是未见最终确认。模型内容和限制见[模型卡](model_card.md)。

CARLA需要独立安装服务端和匹配客户端，相关要求见[桥接说明](CARLA_bridge.md)。不把条件跳过的测试称作CARLA验证。运行中的批次、恢复入口和停止条件见[交接记录](handoff.md)。

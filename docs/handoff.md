# 当前接续状态（2026-11-02 22:24）

沿用当前仓库和 feature/physical-hierarchical-ppo 分支；不重建、不恢复瞬移。用户要求自主完成研发、真实训练、验证和 GitHub 交付。完整研究目标仍未通过，禁止将训练完成、基础4/4或已知开发集当成最终达标。

## 唯一当前物理作业：不要重复启动

- 协调器35188，工具session75294：`scripts/run_native_static_policy_contrast.py`。
- execution：`runs/H_H0_teacher_static12_transfer_execution_20261102`；batch：`runs/20261102T135514838123Z_native_static_H_H0_teacher_transfer`；worker14044、guard64888。
- 顺序运行 H_PPO、H0_PPO、H_teacher，全部12个先前注册的新增静态 TRAIN 条件，canonical seed22，零新增PPO/监督。22:23在H_PPO第8场，前7场只是未审计前缀，不能作完整成绩。工具无输出不代表停止；看 evaluation_heartbeat、guard 和确切进程身份。
- 三者原权重/优化器/RMS/RNG字节不变，共同迁移到当前 batch_sync_v1 与停车辅助；其余配置/标定完全一致。原H/H0训练是legacy actor_async且没有辅助，因此本批只能称冻结部署迁移。
- 全部结束后协调器自动做每个方法的原始帧、实测起点NN推理和严格发动机回执审计。已注册5400秒批次预算/6000秒执行超时，不自动重试；若耗尽，保留完整失败与前缀，不冒充全部完成。
- 运行期间不要修改 scripts/native_road_learning.py、scripts/audit_native_method_endpoint.py、scripts/audit_native_policy_inference.py、scripts/run_native_static_policy_contrast.py。物理worker使用冻结source，后续审计仍调用根目录入口。

## 本轮已完成并核验

1. 原29efb9c…实际原生PPO父模型，固定10240新增控制，TRAIN48课程：291Adam、27闭合训练回合，未接触原4基础条件；基础3/4，一次越界。batch `20261102T123912910706Z_native_canonical_parent_TRAIN48_PPO10240`。未启动完整VAL。
2. 同一父模型/Adam/RMS/RNG、同10240控制预算，唯一课程变化是每4回合复习一次原基础道路：64暴露/48独立条件，全部4基础在任意连续16暴露内出现。实际26重置、320Adam、10240控制均审计通过；不是全部48条件已遍历。batch `20261102T130625263733Z_native_canonical_parent_TRAIN48_PPO10240`，模型05896b45…。
3. 复习模型基础4/4（1416控制）；完整已知VAL15/24，曲线6/8、静态3/8、行人6/8；7越界、2碰撞、0超时，共10233控制，全24起点NN/原始帧/发动机回执已验证。全部协调器37776/32060正常退出0。证据 `reports/native_B1_matched_rehearsal_complete_20261102.json`。
4. 保守几何检查全部12新增静态条件：原teacher case14（合并TRAIN48 case19）在声明闭合道路/静态矩形模型下不连通，其余连通不代表车可执行。全部原失败保留，当前对照不剔除。复习课程没有访问case19，不能把与原课程差异完全归因遗忘。实际原planner为7200节点；私有离线诊断将1800误标original、7200误标higher，公开报告已纠正并保留原尝试。不是CARLA真实网格不可能性的证明。
5. 新命名冻结作业接口严格限定评估和零学习；完整决策计时包含NN+env，周期使用control_dt*substeps；审计逐动作重建外部计时后才允许排除独立时钟汇总。历史无时钟日志不变。
6. 完整CPU1260通过、1跳过、4CARLA排除，259.94秒；凭证 `runs/H_H0_teacher_static12_transfer_execution_20261102/cpu-regression.xml`。

## 不再重复已完成失败分支

路线误差残差离线1/24；5120实际PPO低率5/24、匹配高率4/24；新的几何平衡拟合3/24。全流程/原始失败保留，不继续盲目追加相同拟合。五种子更广B1为54/120（仅24重复已知条件）、58不安全、8超时；公开预发布及151660帧/120起点/5图无认证下载重建全部完成。H/H0旧执行版本120/120 vs119/120，PPO对零PPO教师120vs117，差值区间均跨0；强B3已知24/24且通行时间没有PPO优势。旧异步H与当前同步B1不能包装为公平完整对照。

## 后续执行边界与目标

先完成当前静态对照及自动审计，据实际失败定位高层选择/规划接口，避免继续把全部时间花在B1变体。B1要求基础学习、无明显实现缺陷和合理训练；允许完整任务较弱，不要求达到H的80%目标。单seed4/4尚不能代表健康五seed。正式H/H0/B1需相同当前物理与安全版本、透明且公平的完整交互/监督预算。现有5种子H和B1历史预算不同，必须披露。

完整目标：PPO→子目标/速度→HybridA*→时间轨迹→MPC→真实CARLA反馈；轻量85%/核心75%，CARLA80%/核心70%、碰撞或越界≤5%；五独立实际适配策略；至少120个NEW去重配对最终条件（三类各≥30，另有未见布局/车辆）；分层差≥10pp且95%下界>0或事前锁定合理效率假设；PPO对强B3贡献及反馈协作消融；代码/模型/原始失败/图表/公开复现/GitHub真实交付。最终新条件尚未生成和查看。

CARLA沿用D:/CARLA_0.9.15/WindowsNoEditor和现有Python3.7桥接；root .venv为3.11。不干预Desktop/papper其他项目。C盘旧runs/prerelease等目录有junction到D盘，禁止递归删除/移动。只管理身份核验的本任务进程。C空余约18GiB，下一作业前复核资源。不要新增子代理。

完整较早历史已逐字节归档 `reports/handoff_before_route_feedback_PPO_20261102.md`。最新源码推送前为ffbb465，GitHubCI41446722237成功；6ea0b19为后续文档提交。当前这批代码和报告需在回归通过后commit/push，不force push、不上传个人路径或巨型runs。

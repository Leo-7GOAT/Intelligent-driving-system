# 当前接续状态（2026-11-03 12:49）

目标不变，整体研究验收UNMET。继续当前分支，不重建项目、不恢复瞬移、不新增子代理。

## 唯一当前物理作业：训练后全部12静态对照

execution `runs/H_H0_recovered_static12_disk_ready_execution_20261103`，协调器31384/工具session94616，batch `runs/20261103T044757476793Z_native_static_H_H0_teacher_transfer`，phase frozen_native，正在真实CARLA执行。不要重复启动。顺序H_PPO、H0_PPO、零PPO教师H_teacher；全12新static TRAIN条件均保留，包括地图模型不连通case14。固定7200秒、零重试、零学习、零新最终条件；结束后自动逐方法原始帧/12起点NN/严格传送审计。

入口 `python scripts/run_native_static_policy_contrast.py --execution runs/H_H0_recovered_static12_disk_ready_execution_20261103 --pair runs/20261103T043242Z_native_recovery --h-parent runs/20261102T143204721793Z_native_diverse_H_H0_PPO10240 --wall-seconds 7200`。这是已经启动的入口，勿重跑。两个训练后神经策略保持其当前同步/停车训练低层，零PPO教师明确是冻结迁移。不是五seed正式结论。期间不修改static评测入口及三个审计入口；worker/source与模型都已冻结。

## H/H0真实适配与24场验证：已经完成

H原batch `20261102T143204721793Z_native_diverse_H_H0_PPO10240` 的方法H完整保留：10467新增训练控制、330Adam、24/24成功且零不安全，10467训练帧+7994评测帧核验。

H0原进程外部丢失，原因未知。持久化7913控制/250Adam，末次可读9556控制/300Adam；9556-7913=1643未保留控制，50次Adam未保留，未知尾部保留。原H没有重跑。

H0恢复batch `20261103T043242Z_native_recovery` 已退出0并完成owned_empty_server_stopped：新增2540控制、80Adam、完整24/24且零不安全，7966评测控制，原始帧/24起点/严格发动机传送全通过。合计保留课程10453控制/330Adam，已知消耗至少12096控制。两个方法的30Adam参数状态均从原初始各推进330，actor16/critic14张量变化，RMS均完全相同。完整执行分析脚本与报告：`scripts/analyze_native_diverse_recovered_pair.py`、`reports/native_H_H0_diverse_recovered_complete_20261103.json`。

恢复器的--additional-budget将原20539+10240映射到累计30779，从28452只补请求2327；实际一轮超出213步，仍在注册319最大超出内。16项回归通过，修复已推送d0c9fb1。原prepared失败043157目录标为未注册未执行，原pair唯一恢复登记指向043242，不要再恢复原pair。旧observer68268/session80599已经在static准备时受13GiB门槛拦下，未开始那一批物理世界；失败执行目录 `H_H0_recovered_static12_execution_20261103`保留。当前另名disk_ready才是实际新启动作业。

## 存储与已结束观察器

C盘约13GiB空余，static准备13GiB/运行保护12GiB。四份已完成旧H/H0cohort共1,069,061,697字节、3989文件逐文件Move-Item到D:/ThesisNativeArchive_20261103，移动前后SHA256一致；只移除验证为空的原目录，不递归删除。原路径现在是junction，禁止递归删除/移动这些路径或旧prerelease junction。公开receipt `reports/native_completed_artifact_relocation_20261103.json`。此前递归删除提案被自动审核拒绝，实际改用了上述逐文件移动安全方式。

不要触碰Desktop/papper其他项目。原协调器50820、1732、旧worker42136/guard64188等均已确认不存在；恢复59172/42916已正常结束。观察器55320仅等待时被替换，记录保留，未中断物理训练。工具session25797、80599终态可核验但勿重新启动。

## 后续仍须推进

先完成当前12静态对照和审计，判定新TRAIN上的泛化、PPO及反馈是否有实际改善；H/H0在旧24同为24/24本身没有新增反馈效益。不要重复失败B1观测残差/同数据拟合。五种子基础学习/训练充分与完整预算公平对照、PPO对强B3贡献、可解释协作消融以及120NEW去重最终条件仍缺，目标和门槛不降低。旧H异步训练与当前同步训练历史必须区分。B1可完整任务较弱，要求基本运动、实现正确、充分合理训练，不能要求达到H的80%门槛。

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

先完成当前实际H/H0适配及自动审计，据实际失败定位高层选择/规划接口，避免继续把全部时间花在B1变体。B1要求基础学习、无明显实现缺陷和合理训练；允许完整任务较弱，不要求达到H的80%目标。单seed4/4尚不能代表健康五seed。正式H/H0/B1需相同当前物理与安全版本、透明且公平的完整交互/监督预算。现有5种子H和B1历史预算不同，必须披露。

完整目标：PPO→子目标/速度→HybridA*→时间轨迹→MPC→真实CARLA反馈；轻量85%/核心75%，CARLA80%/核心70%、碰撞或越界≤5%；五独立实际适配策略；至少120个NEW去重配对最终条件（三类各≥30，另有未见布局/车辆）；分层差≥10pp且95%下界>0或事前锁定合理效率假设；PPO对强B3贡献及反馈协作消融；代码/模型/原始失败/图表/公开复现/GitHub真实交付。最终新条件尚未生成和查看。

CARLA沿用D:/CARLA_0.9.15/WindowsNoEditor和现有Python3.7桥接；root .venv为3.11。不干预Desktop/papper其他项目。C盘旧runs/prerelease等目录有junction到D盘，禁止递归删除/移动。只管理身份核验的本任务进程。C空余约18GiB，下一作业前复核资源。不要新增子代理。

完整较早历史已逐字节归档 `reports/handoff_before_route_feedback_PPO_20261102.md`。最新源码推送前为ffbb465，GitHubCI41446722237成功；6ea0b19为后续文档提交。课程/审计代码9ab91f3已推送，GitHubCI41461752584成功；后续训练队列代码958c922已推送，新增4项协议检查和原7项检查均通过。不要force push，不上传个人路径或巨型runs。

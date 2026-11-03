# 当前接续状态（2026-11-03 16:18）

目标不变，研究验收UNMET。继续当前分支，不重建、不恢复瞬移、不新增子代理。以下是实际已启动状态；继续前须复核OS进程身份，不能仅凭状态文件重启任务。不要触碰Desktop/papper其他项目进程。

## 唯一当前物理作业：补齐旧静态对照

协调器38744，工具session33086，execution `runs/static_remainder_after_reference_v4_execution_20261103`；当前batch `D:/ThesisNativeArchive_20261103/20261103_static_H0_remaining_remainder`，worker66632（创建15:55:41）、guard18204。16:17心跳case5/actions40，正在第6/8场，不是已完成6场。入口 `scripts/run_native_static_remainder.py` 已启动，勿重跑。完成H0缺证据的原case4..11后，自动执行未开始的H_teacher全部12场；零学习、固定模型及原物理版本，逐方法raw/起点NN/严格发动机回执核验，有限墙钟、零自动重试。

旧H完整12场为3成功、8超时、1越界、8306控制，raw及12起点NN审计均通过，不重跑。旧H0虽心跳曾到case5，损坏gzip仅保留case0..3完整证据：2成功、2超时、2462控制。case4终止丢失，故必须补case4..11共8场；所有损坏文件保留，没有合成evaluation.json。公开 `reports/native_static_interrupted_prefix_20261103.json`，审计 `scripts/audit_native_torn_static_prefix.py`。独立D盘准备夹具通过launcher -ValidateOnly，已标abandoned防止误启动，见 `reports/native_static_remainder_preparation_validation_20261103.json`。

## 修复后的reference评估已排队，尚未执行

协调器40108，工具session38578，execution `runs/reference_context_schema_fixed_endpoints_wall_execution_20261103`；入口 `scripts/run_native_reference_endpoints.py`，持有38744确切进程身份，只读等待上述整个旧对照队列正常结束。最长8小时等待，随后在D盘依次新建并执行 `...v4_static_schema_fixed`（完整12场）与 `...v4_known_validation_schema_fixed`（完整24场）。阶段墙钟预算14400/3600秒、零学习、零自动重试、零新最终条件；raw/全部起点NN/所有静态序列NN/严格发动机回执均自动核验。禁止另开CARLA或重复队列。若前序失败，队列保留失败并停止，需要根据实际日志继续处理，不能盲重试。

原等待器42296/session86576已在确切父38744与物理worker66632均确认存活、目标评估batch尚不存在时，仅通过保留句柄停止；不停止任何车辆进程。原execution的superseded_before_dispatch/exit receipt保留，当前40108才是唯一schema修复评估等待器，不能恢复旧等待器。

## 新reference-context真实训练已经全部完成

训练batch `D:/ThesisNativeArchive_20261103/20261103_reference_context_TRAIN48_PPO10240_v4` 已正常退出0、owned_empty_server_stopped、端口清空。原worker63072/guard40168已结束，不能重新训练这个父批次。H10511控制/1056决策/330Adam；H010513/1056/330，共21024实际训练控制。两方法32个Adam状态各从初始化推进330，RMS mean/var/count均完全一致。H模型f973176ff7731f6a3c0b1ed0ca20981dbdb6d42fad2fef74cf09ba355a054aff；H0模型4a7a222cb5ca0bef3903ab631e562166f63056ac8665b571931f3161d31b4cd3。

实际使用scale1、4736输入、原31动作。H新增参考系数0.0011252184631302953，H0为0。H0同时隐藏旧反馈与新context，不是单独新增项的消融。每方法请求10240，实际超出271/273，均低于注册319上限。零新监督、固定同步/停车低层、冻结RMS。新增参数零初始化，原30个Adam状态保留。每320控制保存检查点，最终选择固定预算端点，没有挑最优epoch。

独立只读训练帧观察器61720/session45413已正常完成：21024训练控制全部逐帧核对CARLA state/command/传感器证据，reward累计、全部实际控制、末次optimizer/最终checkpoint计数、1176重置步核对一致，22200世界步；严格发动机传送通过。主报告 `reports/native_reference_context_learning_complete_20261103.json`。单H先期状态证明 `reports/native_reference_context_H_learning_complete_20261103.json`。所有这些都不是新性能成绩。

## 首次reference静态评估失败与窄修复

原训练协调器47524/session43917因随后评估失败而退出1，不能把它的失败误当训练失败。失败batch `...v4_static` 的worker73148在首个控制前报 `Reference context observation schema differs`：旧eval-only构造器用了4705维RecordedRoadEnv，遗漏训练时31位context，模型需要4736。仅执行30个reset世界步，0策略控制、0新学习，数据/源码/失败均保留，server60112已owned_empty正常清理。

`native_road_learning.native_environment_class`统一学习与评估的版本化观测；为已冻结旧worker，`prepare_native_reference_evaluation.py`仅把eval-only表达式中的RecordedRoadEnv替换为其已定义的environment_class，逐字节保留其余源及换行。planner/轨迹/MPC/actuation/策略/奖励均未换。协议记录before/after SHA和修复范围；新root源的训练后普通评估也使用相同schema。分区安装器兼容新旧构造器，保留原teacher索引，不注入新的物理控制。

已在独立D盘夹具 `20261103_reference_context_static_schema_validation_only` 验证448源文件/校准/模型等launcher输入；夹具已abandoned，不能当真实评估。`probe_native_reference_evaluation_schema.py`在冻结源下执行实际评估构造表达式，客户端只回放失败尝试保留的真实CARLA begin，禁止step/finalize/tick；实际H/H0模型均得到4736观测并正常predict，全部权重及optimizer张量与训练最终模型逐位一致。0新世界步、0新控制。公开 `reports/native_reference_context_evaluation_schema_validation_20261103.json`。实际修复后物理评估仍PENDING。

## 回归与资源

前一完整本地CPU1272通过/1跳过/4CARLA排除；后续root源074927a的GitHub CI41592067959通过。本次schema修复：专项34项（分区/评估）及11项（实际PPO保存续训）通过，真实模型离线预检通过。完整CPU第一轮XML `D:/ThesisNativeArchive_20261103/reference_context_evaluation_schema_cpu_v1.xml`：1302通过、7失败、1跳过；6个失败明确由C盘不足8GiB触发训练保护提前停止，1个为分区源码锚点不兼容。原失败保留，不降低磁盘底线；分区已修复。完整CPU第二轮session79806已退出0，临时目录显式D盘，1310通过/1跳过/4CARLA排除，360.28s，XML `.../reference_context_evaluation_schema_cpu_v2.xml`。其后仅延长static双方法墙钟到14400秒（场景、模型、每场deadline不变），专项8项再通过。

C盘约8GB十进制空余，小于8GiB；D盘约465GB空余。后续pytest必须用新且专用的D盘 --basetemp，训练/评估大文件均D盘。多个runs旧路径是junction，禁止递归删除/移动。新增恢复器支持 --output D:/...，检查实际工件盘，默认仍workspace/runs；既有目录拒绝覆盖。早期scale1模型无reference_gain_scale属性，新worker缺省1保持原语义；带显式scale10仍用10。实际合成PPO保存/加载兼容已测。

`audit_interrupted_native.py`能按完整可读计数、预注册预算、reload_exact、最终checkpoint和optimizer辨别没有evaluation.json的已完成纯训练方法，防止另一方法失败后重复训练前者。只有最后一轮首次跨预算才算完成；不同rollout长度按实际记录判断，不硬套319。它不制造端点评估成绩。所有恢复前都要单独确认worker/guard/server终态及owned_empty清理。

## 前次失败、历史与剩余目标

v2 scale1实际已在13:35启动；终止旧observer72736时物理worker仍运行，“v2未启动”曾判断错误，原记录与 `reports/native_reference_context_execution_correction_20261103.json`均保留。v2在2412控制/70Adam之后、首个2560检查点之前崩溃，没有可恢复新权重。Windows事件1000的Python3.7 PID23612与tick锁一致，ucrtbase.dll/0xc0000409，C++根因仍未知；request2422未确认，没有重tick。证据 `reports/native_reference_context_bridge_crash_20261103.json`。v4从原已核验初始化显式开始，并保留至少2412失败消耗/70丢失Adam及未知尾部成本；父唯一恢复登记指向v4，禁止重放v2。scale10 v3只曾准备，因端口占用在worker启动前拒绝，0原生控制，不混用。

旧适配H/H0在已知24场各24/24是不同模型、不同阶段，不可写成当前context版本成绩。五seed旧H120/120 vsH0119/120、PPO对零PPO教师120vs117差值区间都跨0；强B3已知24/24，尚无稳定新增反馈或PPO收益。新的旧静态完整对照与context静态/known24均还需收齐。保持目标：至少5独立真实适配策略、健康充分训练B1、公平H/H0/B1/B2/B3及消融；至少120NEW去重最终条件、各核心30、未见布局/车辆；轻量85%/核心75%，CARLA80%/核心70%/不安全≤5%，H对B1预定差值与CI、PPO对强B3及协作贡献、原始数据/模型/失败成本/视频/公开复现。目标不降低，最终集未触碰，不能把当前单seed开发研究当最终完成。

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

# 当前接续状态（2026-11-03 19:24）

目标不变，研究验收UNMET。继续当前分支，不重建、不恢复瞬移、不新增子代理。以下是实际已启动状态；继续前须复核OS进程身份，不能仅凭状态文件重启任务。不要触碰Desktop/papper其他项目进程。

## 旧静态对照已全部结束，禁止重跑

协调器38744及worker52508/guard64696/server33076均已结束；execution `runs/static_remainder_after_reference_v4_execution_20261103/complete.json`存在，H0缺项8场和H_teacher完整12场的raw、全部起点NN、严格发动机回执全部通过。旧H0补齐后为4/12；教师5/12（6deadline、1bounds）、7339控制。新报告 `reports/native_static_H_H0_teacher_complete_20261103.json`，脚本可选 --teacher；原双策略报告原样保留。

教师是896步行为克隆监督初始化、零PPO的神经网络策略，不是直接规则B3。与旧H/H0共享冻结源码、校准、环境、全部12物理条件；教师模型/归一化与原注册哈希完全一致，零新增学习。没有证据支持PPO收益，不得把该单seed开发对照包装为正式主结果。只读真实分析执行通过；篡改raw绑定、出现PPO更新、缺严格发动机回执三种负检查均被拒绝，无新驾驶。

原损坏H0仅case0..3有完整证据，剩余case4..11已补齐。损坏文件及未知成本继续保留，没有合成evaluation.json，没有重跑已验证成功回合。公开 `reports/native_static_interrupted_prefix_20261103.json`。

## 旧H/H0静态全12结果与独立失败解剖（17:25）

`reports/native_static_H_H0_recovered_complete_20261103.json` 及 `scripts/analyze_native_static_recovered_pair.py` 已实际完成。H3/12（8deadline/1bounds），H04/12（7deadline/1bounds），完整case控制8306/7761。H0仅组合原已核验case0..3与新case4..11，模型/归一化字节、校准、所有冻结源、route/hazard一致；原4起点独立NN重建也通过。H0已知消耗至少7850控制（含未完整保留的尾段），未知尾部仍披露。当前是旧4705输入模型，不是新context模型，不证明反馈收益。

共同case4中，两方法101帧state及command逐位相同，11次action均26，首次plan成功后10次reference_reused，MPC全部solved，无planner/MPC失败或干预，但v4Driving足迹探测一角点miss。不能将全部失败归为NN切换参考。进一步离线复核：模型道路余量约1.194m，该实际角点与模型车身角点仅3.14微米差，却被栅格标可行驶；同哈希本地OpenDRIVE和同一冻结lane_projection算法复现该点非Driving、五个栅格采样均Driving，单格17x17采样中15点非Driving。扩大同lane局部s搜索未找回该点，不能归因于小范围投影搜索。

只读引擎射线在同点和四周±2cm共5点均返回Roads；已核验当前owned server33076创建时间/进程、端口及Map哈希，未调用tick/control/spawn。它说明lane近似与road mesh存在分歧，不证明车道合法性或连续扫掠安全。原v4 out_of_bounds统计和所有失败均不改。不得通过增大容差或重算旧分数达标。若后续修复geometry，须新环境版本、全方法共享、另行实际验证，并区分工程收益与PPO收益。当前新context评估仍按v4原几何冻结运行。

可复核证据 `reports/native_static_shared_road_disagreement_20261103.json`，便携输入 `protocols/native_static_case4_road_disagreement_input_20261103.json`，脚本 `scripts/diagnose_native_road_boundary37.py`（只加载本地Map无Client）、`scripts/probe_native_ground_surface37.py`（保留只读owned句柄，仅语义射线）。完整原始探针与密集采样在D:/ThesisNativeArchive_20261103/road_boundary_probe_20261103，未改CARLA安装资源、环境或当前物理actor。

## 唯一当前物理作业：scale100的单独H真实PPO

执行协调器48660（18:25:46.148，venv父61344），execution `runs/reference_scale100_after_scale1_endpoints_execution_20261103`，现SELECTED_H_COURSE / course/native_training。batch `D:/ThesisNativeArchive_20261103/20261103_reference_context_TRAIN48_PPO10240_scale100_H`，worker18496（19:22:21.706，creation_epoch1793704941.706）、guard29560、owned shipping73216（19:22:04.806，launcher33060）。已实际保存首个320控制/10Adam；19:24 latest_checkpoint阈值1280的实际累计1578控制/50Adam，guard healthy。必须复核OS身份再观察，不能重复启动。

`reports/native_reference_scale100_first_saved_learning_20261103.json`验证固定checkpoint_000000320：32个Adam状态相对注册初始化各前进10，RMS mean/var/count完全一致，实际learned gain0.0002250900、scale100、有效logit0.0225090。它是已保存学习证明，不是完整原始训练帧审计、固定预算端点或性能成绩；不得挑该检查点做最终结果。请求10240控制/最大319overshoot/每320阈值保存/零新监督/零新最终条件，H0不重训。完整训练后自动raw/engine、再静态12和known24及NN审计，入口已经运行，不重跑。

## scale1双方法全部学习与评估已结束

原评估协调器40108已结束，OS查询无此原进程，execution `runs/reference_context_schema_fixed_endpoints_wall_execution_20261103/complete.json`完整，后续等待器确切句柄已确认退出0才启动新course。静态旧worker11232/guard27340/server61932、known24旧worker59512/guard52272/server33644均正常结束并owned_empty_server_stopped、TCP/UDP清空。19:21观察时只读监控器遇到已结束进程的QueryFullProcessImageName失败，随后已用OS消失/complete及后续读取的退出0状态确认；没有因此重启任务。

新版4736输入模型静态：H3/12（8deadline/1bounds，8304控制、833NN决策）；H04/12（7deadline/1bounds，7761控制、778NN决策），各12起点及完整静态动作推理匹配。known24：H24/24、8002控制；H024/24、7993控制；各24实测起点、静态子集全部动作以及全原始帧/严格发动机回执全部核验。合计32060实际评估控制，0新学习，原始失败均保留。困难静态成功数与旧4705策略相同，没有改进，不能用known24全成功替代静态能力或最终独立测试。

公开 `reports/native_reference_context_static_complete_20261103.json`、`reports/native_reference_context_endpoints_complete_20261103.json`。单H早期报告/early_endpoint_raw_audit及当时pending快照继续保留，最新以完整端点报告为准。19:13真实known24启动后仅约数分钟完成；不是在等待原静态作业。原v4几何和评分未改变。

原等待器42296/session86576已在确切父38744与物理worker66632均确认存活、目标评估batch尚不存在时，仅通过保留句柄停止；不停止任何车辆进程。原execution的superseded_before_dispatch/exit receipt保留，当前40108才是唯一schema修复评估等待器，不能恢复旧等待器。

## 真实首场失败的参考切换诊断（17:43）

`diagnose_native_reference_switch.py --inspect-policy`已在实际冻结source根下执行：H case0前24个决策全部逐个NN重建相同，重新规划的status及节点数与记录相同，5次放弃仍可复用参考。decision23实测11.5秒，选择action2（概率0.68458）后no_path，原参考仍可复用，对应22/24/26/28/30的context位为1。额外context logit达到约2.36494才会与当时最大概率动作打平；实际训练出的有效系数仅0.00112522。它证明当前小系数没有消除这个可观察切换，不证明保留原参考必定成功，也不说明增大系数训练必定有益。后续若调整该新增参数的学习尺度，必须新版本、真实PPO及同预算对照；不得硬选动作或改当前评估权重。

公开 `reports/native_reference_context_case0_switch_diagnostic_20261103.json`。完整90决策及CARLA begin不可变输入快照位于D:/ThesisNativeArchive_20261103/reference_case0_switch_bound_20261103.inputs.json，报告保存其SHA。当前物理批次仍未完成，gzip尾部EOF是正在写入的前缀状态，不冒充完整raw端点审计。首次在root源码下的诊断因reference policy的scale兼容代码与冻结源不同而正确拒绝；通过NAVSTACK_AUDIT_SOURCE_ROOT选择真实冻结实现后通过，未绕过源码检查、未改当前执行源。新增代码只读、零控制/零训练。

## 单独H的scale100初始化与先期预检（已接续实际训练）

已补齐initializer/training/evaluation/course入口的H或H0子集支持，默认仍双方法；明确方法清单、预算及recovery必须相符，H单独评估仍保留完整12/24条件。course现在自动逐帧核验training-only raw并检查完整场景数。新training准备默认每320控制保存，避免再次在2560首次保存前损失学习。当前执行的旧源码与批次完全不动。

`D:/ThesisNativeArchive_20261103/reference_scale100_H_initializer_20261103`已完成，仅H、scale100、gain仍初始化0。与当前v4 H的初始模型核对：32个参数张量、30个原Adam状态及超参数完全一致，RMS mean/var/count和RNG字节相同；24真实起点策略概率及价值一致。物理控制/新PPO/新监督/最终条件均0。已有H0学习不用重跑；可将当前已完成的scale1 H与这个候选做相同起点/请求预算的开发比较，历史损失和调参成本另计，不宣称匹配总开发消耗。

真实D盘H-only准备与launcher -ValidateOnly通过459源。两个training_validation_only夹具均已写abandoned_before_launch，不能误启动。只保留可复用initializer；基于已完成H静态3/12及切换margin，已注册下述唯一串行等待器；须整个前序静态/known24正常完成及审计通过后，才创建新时限真实训练batch。目前已按下述门槛自动接续真实训练；这些预检夹具仍不能启动。它是对decision19仅需额外0.02563logit但实际系数0.001125的最小学习尺度假设，不是硬选轨迹，不保证修复decision23或任何成功率。

目标专项31通过（53.95秒），包括新增H-only全清单、拒绝空/重复/非法方法、scale100实际合成PPO训练保存加载；XML `D:/ThesisNativeArchive_20261103/selected_reference_cpu_20261103.xml`。公开证据 `reports/native_reference_scale100_selected_preparation_20261103.json`。该预检报告记录当时0控制；后续实际学习见首个保存点报告，不把预检当作训练。

## 串行门槛已通过并进入训练：只运行H的scale100真实PPO

协调器48660（18:25:46.148，实际Python311；venv父61344），execution `runs/reference_scale100_after_scale1_endpoints_execution_20261103`，入口 `scripts/queue_native_reference_scale.py`。只读OwnedProcess句柄绑定前序40108创建时间1793694326.681/实际exe，前序完整结束及全部审计后已自动进入SELECTED_H_COURSE，目标D:/ThesisNativeArchive_20261103/20261103_reference_context_TRAIN48_PPO10240_scale100_H已启动；当前只存在这个物理作业。身份文件与日志均保留，勿重复启动。

最长8小时等前序确切退出0，要求static12/known24双方法完整、固定模型/raw绑定、全部NN重建与engine通过；注册的460个源码与全部初始化文件字节须不变，变化或前序失败即保留失败并停止，零自动重试。满足后才新建新截止时限的单H10240请求控制course，每320控制保存，继而完整12/24开发评估及自动raw/NN/engine审计；不新增监督、不改最终条件、不重跑H0训练。前序完整结果另存predecessor_complete.json。不要修改已注册navstack/scripts源，避免源冻结门槛拒绝后续；文档、报告可照常完善。

条件队列专项20通过4.79秒，XML D:/ThesisNativeArchive_20261103/reference_scale_queue_cpu_v2_20261103.xml；含前序仍活不得准备/发车、缺清单/错模型/raw、NN或engine失败、等待期间源码变化拒绝。公开 `reports/native_reference_scale100_followup_queued_20261103.json`。0118623的GitHub CPU完整回归41614412744成功，实际1299通过/18跳过/4CARLA排除，161.54秒；不等同CARLA结果。

## 新reference-context真实训练已经全部完成

训练batch `D:/ThesisNativeArchive_20261103/20261103_reference_context_TRAIN48_PPO10240_v4` 已正常退出0、owned_empty_server_stopped、端口清空。原worker63072/guard40168已结束，不能重新训练这个父批次。H10511控制/1056决策/330Adam；H010513/1056/330，共21024实际训练控制。两方法32个Adam状态各从初始化推进330，RMS mean/var/count均完全一致。H模型f973176ff7731f6a3c0b1ed0ca20981dbdb6d42fad2fef74cf09ba355a054aff；H0模型4a7a222cb5ca0bef3903ab631e562166f63056ac8665b571931f3161d31b4cd3。

实际使用scale1、4736输入、原31动作。H新增参考系数0.0011252184631302953，H0为0。H0同时隐藏旧反馈与新context，不是单独新增项的消融。每方法请求10240，实际超出271/273，均低于注册319上限。零新监督、固定同步/停车低层、冻结RMS。新增参数零初始化，原30个Adam状态保留。每320控制保存检查点，最终选择固定预算端点，没有挑最优epoch。

独立只读训练帧观察器61720/session45413已正常完成：21024训练控制全部逐帧核对CARLA state/command/传感器证据，reward累计、全部实际控制、末次optimizer/最终checkpoint计数、1176重置步核对一致，22200世界步；严格发动机传送通过。主报告 `reports/native_reference_context_learning_complete_20261103.json`。单H先期状态证明 `reports/native_reference_context_H_learning_complete_20261103.json`。所有这些都不是新性能成绩。

## 首次reference静态评估失败与窄修复

原训练协调器47524/session43917因随后评估失败而退出1，不能把它的失败误当训练失败。失败batch `...v4_static` 的worker73148在首个控制前报 `Reference context observation schema differs`：旧eval-only构造器用了4705维RecordedRoadEnv，遗漏训练时31位context，模型需要4736。仅执行30个reset世界步，0策略控制、0新学习，数据/源码/失败均保留，server60112已owned_empty正常清理。

`native_road_learning.native_environment_class`统一学习与评估的版本化观测；为已冻结旧worker，`prepare_native_reference_evaluation.py`仅把eval-only表达式中的RecordedRoadEnv替换为其已定义的environment_class，逐字节保留其余源及换行。planner/轨迹/MPC/actuation/策略/奖励均未换。协议记录before/after SHA和修复范围；新root源的训练后普通评估也使用相同schema。分区安装器兼容新旧构造器，保留原teacher索引，不注入新的物理控制。

已在独立D盘夹具 `20261103_reference_context_static_schema_validation_only` 验证448源文件/校准/模型等launcher输入；夹具已abandoned，不能当真实评估。`probe_native_reference_evaluation_schema.py`在冻结源下执行实际评估构造表达式，客户端只回放失败尝试保留的真实CARLA begin，禁止step/finalize/tick；实际H/H0模型均得到4736观测并正常predict，全部权重及optimizer张量与训练最终模型逐位一致。0新世界步、0新控制。公开 `reports/native_reference_context_evaluation_schema_validation_20261103.json`。实际修复后静态及known24双方法全场均已核验。

## 回归与资源

前一完整本地CPU1272通过/1跳过/4CARLA排除；后续root源074927a的GitHub CI41592067959通过。本次schema修复：专项34项（分区/评估）及11项（实际PPO保存续训）通过，真实模型离线预检通过。完整CPU第一轮XML `D:/ThesisNativeArchive_20261103/reference_context_evaluation_schema_cpu_v1.xml`：1302通过、7失败、1跳过；6个失败明确由C盘不足8GiB触发训练保护提前停止，1个为分区源码锚点不兼容。原失败保留，不降低磁盘底线；分区已修复。完整CPU第二轮session79806已退出0，临时目录显式D盘，1310通过/1跳过/4CARLA排除，360.28s，XML `.../reference_context_evaluation_schema_cpu_v2.xml`。其后仅延长static双方法墙钟到14400秒（场景、模型、每场deadline不变），专项8项再通过。

C盘约8GB十进制空余，小于8GiB；D盘约465GB空余。后续pytest必须用新且专用的D盘 --basetemp，训练/评估大文件均D盘。多个runs旧路径是junction，禁止递归删除/移动。新增恢复器支持 --output D:/...，检查实际工件盘，默认仍workspace/runs；既有目录拒绝覆盖。早期scale1模型无reference_gain_scale属性，新worker缺省1保持原语义；带显式scale10仍用10。实际合成PPO保存/加载兼容已测。

`audit_interrupted_native.py`能按完整可读计数、预注册预算、reload_exact、最终checkpoint和optimizer辨别没有evaluation.json的已完成纯训练方法，防止另一方法失败后重复训练前者。只有最后一轮首次跨预算才算完成；不同rollout长度按实际记录判断，不硬套319。它不制造端点评估成绩。所有恢复前都要单独确认worker/guard/server终态及owned_empty清理。

## 前次失败、历史与剩余目标

v2 scale1实际已在13:35启动；终止旧observer72736时物理worker仍运行，“v2未启动”曾判断错误，原记录与 `reports/native_reference_context_execution_correction_20261103.json`均保留。v2在2412控制/70Adam之后、首个2560检查点之前崩溃，没有可恢复新权重。Windows事件1000的Python3.7 PID23612与tick锁一致，ucrtbase.dll/0xc0000409，C++根因仍未知；request2422未确认，没有重tick。证据 `reports/native_reference_context_bridge_crash_20261103.json`。v4从原已核验初始化显式开始，并保留至少2412失败消耗/70丢失Adam及未知尾部成本；父唯一恢复登记指向v4，禁止重放v2。scale10 v3只曾准备，因端口占用在worker启动前拒绝，0原生控制，不混用。

旧适配H/H0在已知24场各24/24是不同模型、不同阶段，不可写成当前context版本成绩。五seed旧H120/120 vsH0119/120、PPO对零PPO教师120vs117差值区间都跨0；强B3已知24/24，尚无稳定新增反馈或PPO收益。旧静态三策略、context双方法静态及known24全部收齐；scale100新H课程仍在运行。保持目标：至少5独立真实适配策略、健康充分训练B1、公平H/H0/B1/B2/B3及消融；至少120NEW去重最终条件、各核心30、未见布局/车辆；轻量85%/核心75%，CARLA80%/核心70%/不安全≤5%，H对B1预定差值与CI、PPO对强B3及协作贡献、原始数据/模型/失败成本/视频/公开复现。目标不降低，最终集未触碰，不能把当前单seed开发研究当最终完成。

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

最新CPU CI：792ab7d / run41619453934完成success，实际1307通过/18跳过/4CARLA排除，297.47秒；完整凭证 runs/reference_scale100_after_scale1_endpoints_execution_20261103/cpu_ci.json。它验证CPU回归，不代表CARLA验收达标。

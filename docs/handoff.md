# 当前接续状态（2026-11-03 14:02）

目标不变，整体研究验收 UNMET。继续当前分支，不重建项目，不恢复瞬移，不新增子代理。所有新增大体积物理数据写 D 盘；不要触碰 Desktop/papper 的进程。runs 中多个旧目录是 junction，禁止递归删除或移动。

## 唯一当前物理作业：参考上下文 scale1 的显式重启

实际 batch `D:/ThesisNativeArchive_20261103/20261103_reference_context_TRAIN48_PPO10240_v4`。协调器 `runs/reference_context_native_v4_execution_20261103`，PID47524，工具session43917；真实worker63072。启动时间14:00，已看到真实80控制/8决策。必须查最新 heartbeat、guard 和状态，不能仅凭本文判断仍在运行，也不要重复启动。

入口 `scripts/run_native_reference_course.py`。每方法 H、H0 请求10240真实控制，最多319步rollout超出；TRAIN48、固定物理低层、零新监督、冻结RMS。使用原v2冻结源和scale1初始化，4736输入、31种原动作，新增actor系数及critic项初始为0；原30个Adam状态保留。H0隐藏旧反馈和新增context，因此不是仅新增context的独立消融。

新增检查点每320控制一次，取代前次2560间隔，降低基础设施中断损失。协调器先核验严格发动机控制传送、32个Adam状态推进和RMS完全不变，再自动准备D盘的static12及known_validation24，逐批真实评估并完成raw/全部起点NN/静态完整序列NN/发动机审计。尚无新模型性能结果；尚未执行的检查不能当成通过。每个阶段独立有限墙钟、零自动重试。训练后任何失败均保留，不能重复父批次。

## v2中断与先前错误的纠正

v2 `20261103_reference_context_TRAIN48_PPO10240_v2` 实际于13:35启动，之后旧observer72736被终止，但物理worker/guard/server继续；“v2未启动”的先前说法不正确。原错误记录及纠正均保留，公开 `reports/native_reference_context_execution_correction_20261103.json`。

13:52:32 v2因 `InfrastructureFailure('bridge exited')` 终止，已通过Windows Application事件1000定位为本任务Python3.7桥接PID23612原生崩溃，模块ucrtbase.dll、异常0xc0000409；PID与tick锁完全一致，C++根因尚未知。末次确认request2421/frame5065，request2422未确认且未重tick。证据 `reports/native_reference_context_bridge_crash_20261103.json`。完整可读2412控制、末次70Adam，尚未达到首个2560检查点，因此没有可恢复的新权重。不是成功的短训练或可保留更新。owned_empty_server_stopped、端口空、记录进程终态已核实。接管observer72220也已因worker失败终止。

v4从原已核验scale1初始化显式重新开始，并将v2至少2412消耗、70次丢失Adam及未知尾部记入失败成本。唯一恢复登记已经写入v2，禁止重复父任务。该课程不是相同消耗预算的正式优势比较。公开 `reports/native_reference_context_restart_20261103.json`，准备器 `scripts/prepare_native_reference_restart.py` 拒绝已有检查点、已启动H0、活进程、未清空端口或重复登记。

scale10 v3只是候选准备，启动时因v2占用端口而拒绝，没有server_start.json或worker，零原生控制。不要重启它或混用其统计。目前实际物理学习唯一为scale1 v4。

## 已验证代码与推理迁移

默认scale1向后兼容；可选scale10只改变新增零初始化系数的单位，并在policy_kwargs保存，没有强制保持参考或规则替换动作。不得宣称scale10训练收益。

`audit_native_reference_inference.py` 独立新增，不改三个历史绑定审计入口。旧模型到新维度模型在实际记录上的H500、H0 496个静态/曲线决策及各24起点完全一致，0新物理控制、0学习。完整序列恢复成功参考并检查status/expanded_nodes；失败搜索只消费原失败记录，不将其当成不可行证明。范围为迁移等价，不能解释成新模型性能。

完整CPU1272通过、1跳过、4CARLA排除，293.36秒，XML `D:/ThesisNativeArchive_20261103/reference_context_cpu_v3.xml`。此后新增的恢复安全边界5项测试通过；新协调器及审计入口编译通过。无需无变化反复跑完整CPU。

## 已结束的旧静态对照与已完成适配

`20261103T044757476793Z_native_static_H_H0_teacher_transfer` 在13:35因C盘低于12GiB被保护终止。原H_PPO完整12场：3成功、8超时、1越界，8306控制，raw及12起点NN审计均已通过；不要重跑已完整H。H0心跳曾进入case5，但损坏gzip只保留case0..3完整终止证据：2成功、2超时、2462控制，原始帧/起始条件/逐帧发动机回执已独立核验。case4终止未保留，不能计入完成；须补原case4..11共8场，没有完整evaluation.json。H_teacher未开始，后续仍缺。原server3840正常清理；该旧失败与v2桥接退出是两个独立事件。

此前真实适配H10467控制/330Adam、H0恢复后保留10453/330，均已知VAL24/24且0不安全，原始帧/24起点NN/严格传送已验证。H0已知真实消耗至少12096，含1643未保存控制/50未保存Adam及未知尾部。公开 `reports/native_H_H0_diverse_recovered_complete_20261103.json`。同一seed已知24条件不能证明新增反馈或PPO对强规则的优势。

## 纯训练原始帧审计（14:39）

新增独立只读 `scripts/audit_native_training_only.py`，不改历史绑定审计入口。逐训练tick调用既有严格state/command/传感器证据校验，另核对所有applied控制完整覆盖、首帧与begin关系、注册TRAIN条件、累计reward_components、控制/动作计数、末次optimizer/最终checkpoint计数及reset+control世界步总账。已有两批B1实际训练各10240控制全通过（重置867/805步），源与各流哈希绑定，公开 `reports/native_training_only_raw_verification_20261103.json`。这些是已发生训练的核验，不是新增训练或评估。

当前v4额外只读观察器PID61720/工具session45413，持有worker63072确切身份，入口带 `--wait-identity .../worker_process.json`，等待终态和owned_empty清理后自动写v4 `training_only_raw_audit.json`。注册文件 `training_only_raw_audit.waiting.json` 已保存，有限等待、零重试、没有车辆控制。不要重复启动，当前审计PENDING不能当作通过。6个合成物理证据测试通过，包括state篡改、reward不符、未入账控制、累计计数错误和丢失首帧的拒绝。

## 续跑入口的存储与旧模型兼容（14:29）

`prepare_native_recovery.py` 新增可选 `--output D:/ThesisNativeArchive_20261103/<唯一新子批次>`，空间与内存预检针对实际工件盘，默认路径仍为workspace/runs。既有输出拒绝覆盖，12GiB运行存储底线保留，原权重、Adam、RNG、课程游标及绝对lineage目标口径不变。当前物理训练没有调用此恢复入口，也没有改变其冻结源。

若未来实际中断，必须先核实worker/guard/server已终态与owned_empty清理，再生成原始interruption审计并使用唯一恢复登记；有checkpoint就续原学习状态，不能再次用checkpoint-free restart。早期scale1冻结policy没有reference_gain_scale属性；新worker按其原语义缺省1.0，防止恢复后checkpoint保存/注册校验AttributeError。带显式scale10的模型继续精确使用其10，不能降成1。

恢复/存储/检查点/PPO回归20项通过；真实合成PPO模型的H/H0、scale1/10迁移与保存/加载4项通过，其中两项显式去掉早期模型不存在的属性和kwarg，再保存检查点并重新加载核对所有权重。这是恢复兼容证据，不是新增CARLA性能。新增完整开发评估准备5项也已通过；包含固定预算不足、选错保存端点、训练提前停止的拒绝，以及全部12静态/24已知条件保留。

## 已登记的旧静态缺项队列（14:12）

`runs/static_remainder_after_reference_v4_execution_20261103`，协调器38744/工具session33086，入口 `scripts/run_native_static_remainder.py`，持有当前reference协调器47524的确切进程句柄，只读等待，当前没有第二个物理worker。最长8小时等待，之后依次原H0缺证据case4..11共8场、未开始的零PPO教师12场，均零学习、D盘、7200秒有限墙钟、零自动重试。每阶段raw/全部起点NN/发动机传送审计，原始case映射保留，已完成H和H0case0..3不重跑。

两种准备已在独立D盘夹具上通过真实launcher -ValidateOnly：原路段及静态hazard逐项一致，模型/校准/442个源哈希/监督来源均通过。夹具显式abandoned_before_launch，零物理控制，不冒充真实补评。公开 `reports/native_static_remainder_preparation_validation_20261103.json`。实际批次将在队列唤醒后才准备，防止等待导致过期deadline。完整旧结果仍PENDING，公开 `reports/native_static_interrupted_prefix_20261103.json` 只报告可核验部分。

## 尚未完成的最终目标

新context训练及上述开发验证、旧静态完整H0/teacher0、五独立实际训练策略、健康且充分训练B1的公平对照、B2/频率/共享安全/固定参考消融仍需推进。最终至少120NEW去重条件、各核心场景至少30、未见布局/车辆、五seed配对统计和最终公开复现尚未完成。保持既定轻量85%/核心75%、CARLA80%/核心70%/不安全≤5%和H对B1差值目标，不降低验收线，不把已知条件或迁移等价当成未见最终成绩。

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

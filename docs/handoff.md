# 当前接续：2026-11-04 14:29 PPO与MPC对照均闭合，仍1/2；不要重复训练/测试/两场

当前无活动原生或CPU任务。MPC实验execution20261104_mpc_reference_effort_same2_execution/complete.json存在，32888/32960已结束：case21 success561/43interventions/4planfails，case27 deadline900/560interventions/56planfails；1461actualcontrols/1522worldticks/147NN及raw/engine全部PASS，858个solvedMPC tick均确认reference_centered_effort=True。相对同模型旧控制器success487+out_of_bounds460，成功数仍1/2，越界改为超时，不是任务完成率改进。控制代价解析平衡偏差已修正但只保留开发prototype，不宣称最终候选达标。

新源cb482f7全CPU exec76997 DONE0：1402PASS/1SKIP/4CARLA DESELECT/301.92s，XML D:/ThesisNativeArchive_20261104/mpc_reference_effort_full_CPU.xml。GitHub CI41784721153(cb482f7)和41782396600(d16cf45)均SUCCESS。公开reports/native_mpc_reference_effort_transfer_20261104.json已闭合，README中英已含5438学习的安全退化和MPC仅改超时结果。此前5438/170Adam课程完整report native_exact_point_PPO5120_complete_20261104.json及初始化/RMS/raw/95NN/engine证据均已闭合，禁止再次运行课程。

额外只读结构检查D:/ThesisNativeArchive_20261104/stalled_failed_action_context_sensitivity.json：在新MPC末端真实失败状态重建NN action2精确，重复action2概率.95331；last_behavior已经编码上一目标，不得声称策略完全没有动作历史。当前reuse context全0，现有学习系数有效logit仅.0118389，抑制失败action2至次高动作需要5.06198；直接把该context改成-1的数学反事实仍选2，没有运行新schema。不能人工扣5.1logit、强制保留参考或换规则后冒充PPO学习。

下一优先研发：固定当前已测试点约束/控制版本，检查真实candidate head及已有动作结果特征如何参与选择，针对95%置信度反复提出失败目标的学习/反馈表达做一个可验证改动。一个可研究的方向是版本化、动作对齐的失败反馈(例如signed execution context)作为学习输入/学习logit项，明确它是已有信息的结构化表示，不能伪称新物理信息；但尚未设计/实现/迁移/训练，不能据反事实宣称有效。需正式新版本和实际更新、H0公平消融。不要继续盲目追加5120、扩大5seed、重调低层权重或重复已关闭开发批次。当前source运行时保持cb482f7，最终120未生成/查看，验收UNMET。

# 历史接续：2026-11-04 14:19 点约束PPO与MPC（均已闭合）

5438实际训练控制/544决策/17rollout/170Adam已正常关闭，5654worldticks=5438+216reset；32Adam均推进170、全部初始化参数/完整Adam/RNG一致、RMS mean/var/count精确冻结。6完整随机训练回合1success/5deadline，最后未完整回合不计成功。source d16cf45，model13e5fa1e→6e885fd6f725bd7a9db295f7de9273f0690d2994e8dbbcb3f50e35ec2864307e，gain-.0005996→+.0001183888。原5120execution/complete.json存在，28696/21740/endpoint27392已退出，不得重跑课程。

冻结same2端点及raw/95NN/engine均完成：原21 success487/0interventions/0planfail，原27 out_of_bounds460/10interventions/1planfail。训练前0/2且0unsafe→训练后1/2且1unsafe，安全退化，不能宣称达标或稳定学习优势。端点947controls，训练+before1800+after947=8185本比较实际控制。完整报告reports/native_exact_point_PPO5120_complete_20261104.json；scripts/summarize_fixed_lowlevel_adaptation.py验证闭合hash/模型/任务及同低层AST，仅注册门限允许不同，可由原始记录重建。新只读诊断exact_point_case21_policy_switch_diagnosis.json重建10个NN精确，decision9放弃仍reusable参考，lat.18497/yaw-.0315/context26，换action30→no_path；并非只能归因旧参考失效。

新MPC试验有明确单改动：原QP R项以0acceleration/0steering为中心，对非零转弯产生偏差。navstack/mpc.py新增默认False的reference_centered_effort bool，True仅把同R项线性中心移到同reference.a与atan(L*kappa)前馈，weights/limits/QP容差不变。2个恒定转弯/直线平衡解析测试PASS，15个专项PASS。mpc_reference_effort_v1门限仅允许point/joint冻结评估，不能用旧训练登记训练新控制器。CarlaNavigationEnv.reset每个新episode应用client显式开关，默认False行为不变。新增公开reports/native_mpc_reference_effort_transfer_20261104.json尚未有结果。

当前唯一原生作业 D:/ThesisNativeArchive_20261104/20261104_mpc_reference_effort_H_same2_transfer，execution20261104_mpc_reference_effort_same2_execution；协调器32888 creation14:16:48.749075(basePy311，venv29164)，worker32960，身份D:/ThesisNativeArchive_20261104/mpc_reference_effort_coordinator_identity.json。模型6e885fd6冻结，相同[21,27]，零PPO/零新条件，native后自动raw/startNN/allstaticNN/engine。跟进真实终态，勿重复启动、勿在结果前扩展seed或低层变体。

新MPC/bridge reset源码全CPU exec session76997正在运行，basetemp D:/ThesisNativeArchive_20261104/pytest_mpc_reference_effort_full_CPU，XML mpc_reference_effort_full_CPU.xml；不能把旧1399当新MPC通过，不重跑全套。等待这两项关闭再做完整报告/决定保留。所有版本负结果不改分，最终120未生成/查看，验收UNMET。

# 历史接续：2026-11-04 14:00 固定点约束PPO（现已闭合）

唯一原生课程 D:/ThesisNativeArchive_20261104/20261104_exact_point_PPO5120_H 已从source commit d16cf45准备并启动，execution20261104_exact_point_PPO5120_execution。实际协调器28696 creation13:58:09.008901(basePy311)，venv14140；worker21740 creation_epoch1793771900.915，身份文件batch/worker_process.json和D:/ThesisNativeArchive_20261104/exact_point_PPO5120_coordinator_identity.json。必须先核实OS/status/heartbeat，禁止重复启动或重跑已完成5396旧课程。

已实际保存checkpoint_000000320：320真实控制/32决策/350worldticks(30reset)/10Adam，actor L2改变.05237，policy model13e5fa1e→4ad2d3aa；初始全部权重/完整Adam/RNG逐字节相同，H/initial_restore_verification.json已留证。最新heartbeat已440控制/44决策；这些是运行中计数，最终仍需raw/engine/learning proof，不能称整个课程完成或已改善任务。RMS冻结只在结束后对mean/var/count做最终证明，不以pickle文件SHA不同判断均值被更新。

注册单seed22继续链，5120请求control+≤319overshoot，固定全部点约束/转向/轨迹/MPC及option语义，零辅助监督。两条件[21,27]为之前完整0/2已知开发基线，没有新增独立seed或最终条件。协调器顺序训练→原始控制/engine/实际32Adam/RMS/初始证明→自动same2_endpoint→raw/startNN/allstaticNN/engine。无自动重试，4h guard/D12GiB/RAM4GiB/GPU85℃；跟到真实终态，不把训练随机成功代替冻结复测，不据小样本宣称正式研究验收。

当前源点低层完整1399CPU通过/1skip/4CARLA deselect/277.28s；之后注册扩展40项通过3.38s。只修改注册门限允许新5120课程，preparer AST已实际核对与frozen点pilot除settings/registered/显式新增注册常量外相同。source d16cf45已push。新训练未结束，尚无新版完整评估、五独立seed或最终120确认；验收仍UNMET。

# 已完成：2026-11-04 13:57 精确点约束与两场基线

新opt-in native_counterexample_point_tube_v1把原生同帧拒绝点逆变换到局部精确XY，在private planning_view里对同9offset完整车身施加点障碍，真实World/评分/接受阈值不变；规划空间memo key增加版本/1e-5 float32 roundtrip容差/全部float64点bytes，dynamic/墙钟仍旁路。最多8192点、3细化次、每次normal/.25各32000节点，所有joint search成本保留。默认无点输入键不变。源码navstack/reference_point_constraints.py，World.free_many仅private点属性启用。case21/27离线分别7989/2031nodes，source-bound复查两条轨迹SDK全9offset均0badpoints；source-bound D盘native_case*_exact_point_runtime_source_bound.json和native_case21_27_exact_point_road_probe.json。

两场真实baseline D:/ThesisNativeArchive_20261104/20261104_exact_point_joint_speed_H_two_condition_pilot 已闭合，execution20261104_exact_point_two_condition_execution/complete.json存在；16788/30940已结束，1800controls/180NN及raw/engine全PASS。仍0/2：原case21前进3.35m后PPO26→30，81次open_set_exhausted；原case27前进18.46m后转action2，57次open_set_exhausted。两者已能生成并实际执行通过native查询的参考，不是原初始query无法通过；不能说点约束修复完整任务或PPO学习已获益。

全CPU exec9009 DONE0：1399PASS/1SKIP/4CARLA deselect/277.28s，XML D:/ThesisNativeArchive_20261104/exact_point_full_CPU.xml。之后仅训练注册gate改动，另40项PASS3.38s，XML point_training_gate_tests.xml；full覆盖点低层，40覆盖新gate，不冒称全CPU重新覆盖注册扩展。公开reports/native_exact_point_constraint_development_20261104.json。

下一已具体实现：REGISTRATION_POINTS=fixed_exact_point_joint_speed_native_adaptation_v1，单5120新真实控制、零辅助监督、冻结RMS/全部低层，same2[21,27]端点。prepare_fixed_lowlevel_adaptation.py支持完整已审计point pilot，AST校验当前navstack与pilot除settings/registered及新注册常量外一致；原[0,1,4]课程不受影响。run_fixed_lowlevel_adaptation.py按注册ids派生same2_endpoint，实际权重/Adam/RNG初始化、32Adam步推进、RMS及训练/端点raw/NN/engine全自动核验。当前尚未准备或启动新课程，勿称正在学习；先提交当前源码，再准备 D:/ThesisNativeArchive_20261104/20261104_exact_point_PPO5120_H 并启动唯一execution。不要重复已关闭两场/低层变体/全CPU。仍一个continued seed22，所有旧负结果保留，最终120未生成/查看。

# 历史接续：2026-11-04 13:42 四场与全CPU（已完成下一精确点验证）

所有原生作业/审计/CPU/离线probe已结束，无活动任务需等待，不要重启旧队列。当前源2f3fb5d已推且GitHub CI41779692919成功；全CPU exec4192 DONE0，1392PASS/1SKIP/4CARLA DESELECT/321.32s，XML D:/ThesisNativeArchive_20261104/joint_speed_full_CPU.xml。四条件execution/complete.json已存在，31428/33496已退出；[0,4,21,27]为deadline900/success410/deadline900/deadline900，3110controls，raw/全部311NN/engine均PASS。与单case1success558合计同core源5个known条件2/5，v2实际3668controls；加v1失败900，共4568新控制，零PPO，不是独立seed。公开reports/native_joint_speed_four_condition_complete_20261104.json，英文/中文README已同步真实弱结果。

失败定位更具体：case0先执行约250控制，再选择新目标后出现retiming cap不相容/一次native拒绝/62次open_set_exhausted。case21/27的.25空间路径分别7989/486节点成功，但native九offset各拒绝1点，粗栅格cell本来occupied所以反复无效；不是未找到空间路径。按实际query request逐pose+offset匹配XY，case21坏点是最后pose227/228(lat=-.05,turn=-.05)，case27是pose221/228同offset；匹配XY精确，Z有原生车辆高度差.0016368，不能称完整3D逐位重建。D盘native_goal_region_terminal_crop_precheck.json：既有inner goal带.52m最早进入226/224，截到首次内目标带只可能去掉case21最后坏点，对case27无效；尚未改runtime，不可声称修复。

脚本scripts/probe_native_reference_tube_planning.py在case21初态对每搜索primitive使用同9offset栅格车身检验，32000节点timeout、57.6m footprint checks、218.7275秒，exec56002 DONE0；D盘native_case21_reference_tube_planner_probe.json，公开reports/native_reference_tube_model_probe_20261104.json。过度保守且昂贵，未采用运行时，不增加预算继续盲搜，不重复该负试验。

下一最小假设：精确native拒绝点(转为局部XY)作为同九offset车身的额外点约束，代替目前无效的栅格量化增量；只验证已有两条失败轨迹对应实测状态，保留所有原物理评分和全轨迹native查询。需把点约束完整加入空间缓存key并版本化，不能只给world monkeypatch导致旧memo假命中；动态/墙钟旁路仍须保留。先离线有限node验证，通过完整native道路查询后再pilot，不追加训练、不扩大整54场。终点内目标带截断也是有效待验证的语义假设，但不能据此跳过case27中途的badpoint。

# 历史接续：2026-11-04 13:30 joint-speed单场与四条件（现均闭合）

新增opt-in navstack/steering_retiming.py profile measured_steering_rate_retiming_v2。只在normal slew搜索失败后按.25m/s规划空间路径，再按曲率时间插值的最大转向导数/实际初速/加减速重新计时；保持PPO目标及请求速度、几何路径、MPC/原生控制/评分，动态检查和native道路查询使用新timing。原normal与fallback各<=32000节点，逐attempt成本另记录，不宣称总预算只有32000。默认行为不变，当前仅允许冻结评估，不能用旧training注册直接训练新低层。

离线case1低速路径全程.25需106.56秒，联合retime24.714秒，最高1.5m/s，采样模型转向<=.6rad/s、a<=1.5/decel<=2.5，原SDK九offset道路查询0badpoints(另.5空间路径3bad已拒绝)。第一v1真实pilot900deadline，推进14.59m但错误拒绝实测轮角轻微超命令上限/微负速度；raw/90NN/engine全通过，保留。v2修复保留实际signed初速和实测轮角，不钳制实际状态，原case1真实success558controls/27.9simseconds/42interventions/4planfails/0MPCfails，raw/56NN/reference/engine全通过。两个transfer共1458真实控制，零PPO、一个known条件、不是学习收益。公开reports/native_joint_speed_retiming_case1_20261104.json。

当前唯一原生队列 D:/ThesisNativeArchive_20261104/20261104_joint_speed_four_condition_execution，实际协调器31428(启动13:29:31.538，basePy311)，venv33560，worker33496；batch 20261104_joint_speed_retiming_H_four_condition_transfer。预登记原all30 IDs[0,4,21,27](3旧失败+1旧成功)，不重跑已成功的case1；count4，最多3600control/单批4h guard，零retry/零PPO，四场后自动raw/startNN/allstaticNN/engine。检查OS/status/complete后跟进，禁止重复启动。

当前新源全CPU命令正在tool exec session4192运行：pytest -q -m 'not carla'，basetemp D:/ThesisNativeArchive_20261104/pytest_joint_speed_full_CPU，XML joint_speed_full_CPU.xml；39/40/49针对性组均PASS(最新49项6.18秒)，不得把未结束全CPU称通过。不要重复启动全套。四条件结果出来后先审计对照再决定扩大验证/固定低层训练，不凭单case558直接推广30或最终集。

# 历史定位：2026-11-04 13:20 三个失败起点速度/转向过渡耦合

完整30+24已闭合，全部raw/NN/engine通过；无活动原生作业。新只读脚本scripts/probe_native_planner_reachability.py比较同一实测初态/实际目标：case1旧analytic416节点成功，关analytic恒转角998成功，现有1.5m/s slew模型72节点open_set_exhausted；case21为1824/2952成功vs60失败；case27为480/769成功vs74失败。所以这些起点不能归因于只关闭analytic，更不能泛称PPO选错目标。counterfactual case1降低slew模型速度上限到.5/.25后，在16707/13736节点找到几何路径。公开reports/native_slew_reachability_root_cause_20261104.json及D盘对应native_case*_planner*.json；无新控制/PPO/方法评分，源码未改控制逻辑。

下一最小验证已具体化：离线验证较低速度slew可行空间路径的转向速率感知时间参数化，保留实际初始速度、加减速、原生道路/车辆约束，允许在转向变化缓慢处恢复PPO请求速度。不能直接把规划速度假设.5而仍按1.5执行，也不能全程低速到超时后称成功。若该验证通过，再版本化单独pilot及固定低层训练，对B3等共享同样能力。此方案尚未实现，不得声称已解决。不要重复当前30/24或5396训练，也不要无依据扩展新参数组合。

# 已闭合：2026-11-04 13:15 完整30+24，当前方案有退化

所有原生训练/评估均已结束，当前无活动CARLA批次。恢复execution D:/ThesisNativeArchive_20261104/20261104_exact_compute_development_recovery_execution_closed_owner_fix/complete.json存在，协调器9060/31764与worker31316/31000均已退出，owned server停止。原28场1394NN、尾2场126NN、known24起点与全部staticNN、每个raw/engine均通过。完整摆放30场21success/9deadline(70%)，已完成15106控制，另保留中止旧case28的200控制，known24为22success/2deadline、9090控制；实际评估总控制24396。合并30采用两种公开compute实现，零PPO，不是新独立测试。

旧模型/低层的同批结果是23/30、24/24，本版21/30、22/24，没有优势，不保留为最终候选。9静态超时中8个主要open_set_exhausted/1个native_reference_road_refinement_exhausted；已知验证case15(曲线)先3次success/6reuse，随后81次search_budget；case17先1success/1reuse/4constant_velocity_spacetime_conflict，随后84次open_set_exhausted。持续制动不是安全任务完成。所有负结果必须保留。

公开完整报告reports/native_option_full_development_complete_20261104.json；脚本scripts/summarize_exact_compute_recovery.py只接受closed/raw-bound证据，可由D盘原始数据重建。compute优化源82bd22d已推且GitHub CI41776901661成功，50专项测试通过；下一提交仅闭合报告/汇总脚本/handoff。无需再跑5396训练、30/24评估、旧release或全量CPU。新compute profile默认关闭，90决策/900实测控制等价证据保留，不视为学习收益或实时性。

下一必要研发动作：针对已完整日志做一个有边界的离线根因验证，区分32k有限转向速率搜索禁用analytic后导致的可达性损失与策略选目标失败；先用同一失败实测状态/目标比较低层，不能直接盲目加训练或新规则接管。若提出新低层，需版本化及冻结训练，并给予B3等公平对照。当前仅一个continued seed22，未获得最终研究优势；验收UNMET，最终120清单仍未生成/查看。

# 历史接续：2026-11-04 13:03 精确计算优化补跑2+24（已闭合）

旧all30队列已主动STOP安全关闭，owned server停止且端口空，协调器26172/worker32608均已退出。前28完整场20success/8deadline，全部14055实际控制(含未完成尾部)及engine均通过；raw receipt在原batch/H/exact_compute_recovery_prefix_raw_audit.json。原case28未完整结束，须补28/29，不重复前28场，不重复5396已完成训练。原known24未启动并已abandoned，禁止误启动。

唯一活动恢复执行：D:/ThesisNativeArchive_20261104/20261104_exact_compute_development_recovery_execution_closed_owner_fix，实际协调器9060，venv14168，启动13:02:25；需按OS创建时间/可执行文件核验。恢复脚本scripts/run_exact_compute_recovery.py已启动，先补all30_placements_H_exact_compute_tail(2场)，再known24_H_exact_compute(24场)，保持模型13e5fa1e/normalizer/任务/评分，零PPO；source快照使用新exact_static_reference_compute_v3。同时后台只读核验原28场全部NN与参考。先看status/failure/complete和实际进程，禁止重复启动。

优化仅缓存精确静态空间路径并中止同帧不改变约束的重复细化，每次仍以实际状态重新时间参数化/原生道路查询；默认旧行为不变，动态/墙钟搜索旁路。case23已离线逐一绑定90NN/900实测控制，命令/约束/查询位姿一致，搜索402570→1491节点、270→90查询。证据D:/ThesisNativeArchive_20261104/exact_reference_compute_case23_all90_source_bound.json。不是新实车吞吐实测，更不是成功率或学习收益；原慢case23仍deadline900。50专项测试通过5.25秒，当前新源未跑全量CPU，旧1378不能冒称覆盖本次。公开reports/native_exact_reference_compute_development_20261104.json。

首次恢复协调器因旧OwnedProcess已结束抛RuntimeError(而非OSError)在任何新原生控制前退出，原execution/failure保留；修复只接收exact missing handle错误并验证终态/生命周期已结束，身份不符仍拒绝。后缀closed_owner_fix为当前真实运行。后续先完成2+24及原28场NN审计，按全部结果决定是否保留方案；不盲目追加训练/控制器变体。验收UNMET，最终120清单未生成/查看。

# 历史接续：2026-11-04 12:27 54场旧队列状态（已被上文覆盖）

当前只跟进已启动的完整开发评估；暂勿再做新控制器/训练/参数变体，也勿重包旧发布资产。主目标不变且研究验收UNMET。请按status/OS身份继续，而非根据旧文档重跑。

唯一活动队列 D:/ThesisNativeArchive_20261104/20261104_option_adapted_full_development_execution，实际协调器26172（身份D:/ThesisNativeArchive_20261104/option_full_development_identity.json），venv外壳32308。当前all30_placements_H worker32608 creation1793764744.307、watch4240、owned server25952/port2100，已到case23(原case7唯一摆放)的80动作。该场每20次高层决策耗时约260-280秒，暴露严重计算开销；不要宣称实时性，也不改既定900控制/45秒终止口径来抹去失败。之后自动评估known24并逐批raw/start/all-staticNN/engine核验。计数30/24，全部条件保留、无重试；source471文件和模型13e5fa1e固定。

新增纯只读帧监测scripts/watch_native_frames37.py（Py3.7编译通过），observer19964 creation1793765648.570，batch/frame_watch_process.json留身份。启动前核对owned shipping可执行文件及RPC监听PID，再以只读OwnedProcess句柄约束worker/server；只在实际观测到frame递增时写progress.json。已有54次真实帧前进观测(最近12849>12848)，无tick/control/state设置。用于防止每20决策才写心跳导致正常长规划被误判300秒无进度。worker退出后自动结束，不要把此监测当模型在后台思考。后续若known24也需监测，须重新验证该批worker/server身份，不沿用本批PID。

现有闭合证据：5396真实训练控制/170Adam、RMS冻结、32初始Adam/RNG一致；同3端点仍2/3，case0deadline900/1成功334/4成功349，1583控制及159NN全部核验。学习无成功数提升，原failed端点标志错误和0新增控制保留。公开reports/native_reference_option_adaptation_endpoint_complete_20261104.json。不要把随机训练11/12当确定性或独立测试提升。

sourcefc38cd3全量CPU1378pass/1skip/4deselect；endpoint编排补丁3项pass。已推15ce1e8且GitHub CI41770461387成功（fc38cd3/accc697 CI也成功）。本次只读监测和本handoff将另提交，控制代码仍fc38cd3。D盘额外只读拟合native_recorded_steering_response_fit.json：a=.1799,b=.8240,RMSE .00437rad，瞬时命令 .00562、配置tau.12模型 .00669；仅相关性，不是新标定，未据此改MPC。

下一必要动作：等待这两批的真实终态和审计；如失败先保存全部已读前缀/资源身份，不重训5396已完成课程，不覆盖原失败。完成后依据完整30/24结果决定是否保留32k低层，避免继续凭两场成功或未实现想法扩展。最终120确认清单仍未生成/查看。

# 最新接续：2026-11-04 12:00 学习端点无提升；完整54开发评估启动

5396控制/170Adam课程已关闭，修复后的same3端点也关闭并全部审计：1583控制/1675ticks/159NN/3起点、engine无mismatch，case0deadline900、case1success334、case4success349。与冻结接口前2/3相同，无学习成功数提升。原端点因normalizationflag失败保留，execution/normalization_endpoint_recovery_complete.json汇总新子批闭合，原failure.json不覆盖。公开 reports/native_reference_option_adaptation_endpoint_complete_20261104.json。训练worker正常结束，原协调器12704因原端点编排错误退出（不是成功complete），手工修复端点结果已关闭。

只读witness反事实检查未改变失败MAPaction，未采用新runtime补丁。现仅新增diagnose可选检查和评估队列/准备脚本，无控制或学习代码变动。

当前唯一原生队列：D:/ThesisNativeArchive_20261104/20261104_option_adapted_full_development_execution；新venv launcher32308 creation1793764732.427，实际协调器PID待CIM确认。先20261104_option_adapted_all30_placements_H，再20261104_option_adapted_known24_H，全部source_hashes与13e5fa1e模型/normalizer一致，471源，与训练固定源码一致。30完整旧模型可行摆放全部重跑，不复用任何旧模型pilot；24为原已知验证，54均非最终确认。准备工具核对calibration和world/vehicle/road_grid/batch/native_hazards/native_route/lane_projection字节一致；旧7200节点witness不保证新32k搜索可达，不删除新失败。登记counts30/24、-ActivateQueued按阶段更新时限，4h/批、无重试，结束后自动raw/start/staticNN/engine审计。不得重复启动。

本段后旧“当前运行/尚未准备”已过期。当前root两个新评估脚本和诊断扩展待提交；sourcefc38cd3全量1378通过，endpoint编排fix专测3通过；无需重复全量测试或训练。目标仍UNMET，不开放最终120清单。

# 最新：2026-11-04 11:46 真实学习已闭合，冻结复测修复子批次运行

原生5120请求课程已正常完成5396控制(overshoot276)/544决策/17rollout/170Adam。32个Adam均真实推进170，初始化全部权重/完整Adam/RNG相等、RMS精确冻结，训练raw/engine通过，模型13e5fa1e55ec784d0e5bf83f99e73f25493b43d2720dec4bb7295e1cc0cd1b23。学习系数从+.00030845变为-.000599594，负值保留，不预设反馈获益。公开 reports/native_reference_option_adaptation_learning_complete_20261104.json。训练worker5832/watch6060/协调器12704已正常关闭，不能重跑训练。

协调器原_same3_endpoint因训练专属freeze_observation_statistics未删除，在任何车辆控制前失败；全部failed数据保留，owned server已清空停止，确认0控制/0reset。修复只在endpoint准备函数pop该flag；针对性3测试通过6.59秒，无控制代码变更。新子批 D:/ThesisNativeArchive_20261104/20261104_option_native_PPO5120_H_same3_endpoint_normalization_fixed 已启动，launcher23644 creation1793763935.650，实际worker身份待查worker_process.json；模型/normalizer/source_hashes与原失败准备批完全一致。恢复注册位于execution/normalization_endpoint_recovery.json。新eval须手工raw/startNN/allStaticNN/engine核验后汇总，原coordinator failure.json保持真实，不要伪造complete。

source fc38cd3全量CPU1378通过/1skip/4CARLA deselect/361.97秒已保存，doc commit16ee9e9已推。现有root新增仅endpoint编排修复及测试，待提交。正式研究仍UNMET，不把3known条件或12随机训练回合当独立最终验证。

# 当前运行：2026-11-04 11:31 真实5120-control option适配已启动

当前唯一原生作业 D:/ThesisNativeArchive_20261104/20261104_option_native_PPO5120_H，固定source fc38cd3，471文件。协调器实际12704（身份D:/ThesisNativeArchive_20261104/option_native_coordinator_identity.json，venv外壳7496），worker5832 creation_epoch1793762826.276，watch6060；先复核OS身份/status，勿重复启动。协调执行 D:/ThesisNativeArchive_20261104/20261104_option_native_PPO5120_execution。脚本 `run_fixed_lowlevel_adaptation.py`在训练正常结束后自动逐帧审计、32Adam/RMS/初始RNG证明，再用训练冻结源码和同3条件生成_same3_endpoint并核验raw、全部NN、engine。无自动重试。

已实测初始化权重、完整Adam hash、Torch RNG与指定checkpoint完全相同；32个Adam状态初始步数分别330/1310（新增两参数/原骨干），H/initial_restore_verification.json留证。最新训练日志已真实执行>=472控制、首次320控制rollout后10Adam更新，第一完整训练回合case0 success412控制/21interventions。只是随机训练回合成功，非冻结策略提升；最终预算端点才作比较。源初始bf6ef925，低层和option语义固定，方法H seed22单条继续链、5120请求控制+最多319overshoot、零辅助监督/RMS冻结；非独立seed或最终测试。

全量CPU已关闭：1378通过、1skip、4 CARLA deselect、361.97秒（session1284 DONE0），XML D:/ThesisNativeArchive_20261104/option_training_full_cpu.xml。公开reports/native_reference_option_CPU_regression_20261104.json。76项恢复/门限等专项8.93秒；35项option专项5.85秒；43项含训练门限5.79秒；Py3.7编译通过。HEAD fc38cd3已实际推送。本段后的旧“未启动”状态已被覆盖，不可据旧文档重建/重跑。

# 最新关闭及准备：2026-11-04 11:26 显式option接口与5120原生适配

option_commitment_frozen_H_pilot worker18076/launcher10056已正常结束，原case0deadline900、case1success336、case4success350，1586控制/1678ticks/159NN/3起点raw、策略/参考重建、engine均通过。报告 `reports/native_reference_option_frozen_pilot_20261104.json`。新接口仅在PPO重复选择相同action标签、旧参考仍有效时保留原世界坐标目标；换标签/失效/新回合均重新规划，不自动替PPO作决定。旧策略仍失败，是后续真实学习前对照，不能宣称接口已修复任务。

新增 `scripts/prepare_fixed_lowlevel_adaptation.py`准备单H seed22继续5120真实控制(rollout overshoot<=319)，原3个known开发条件，固定32k低层与option语义、零辅助监督、RMS冻结；32参数/Adam/RNG原样恢复，新增worker初始指纹覆盖继续训练并记录optimizer hash，训练后须验证32个Adam状态真实推进。训练门限只有完整注册允许；AST核验低层代码相对pilot只改变settings注册函数。`scripts/run_fixed_lowlevel_adaptation.py`可顺序执行训练、raw/engine/学习保存证明、同3场冻结端点、raw/NN/engine。当前尚未准备或启动训练，不得声称正在训练。

76项相关恢复/门限/option/缓存/STOP测试通过8.93秒；另43项通过5.79秒、Python3.7编译通过。原全量1361属于旧v1源。本次新增源尚待提交/准备后实际运行。旧full_transfer known24不得启动。

# 最新关闭：2026-11-04 11:15 32k三场pilot已全部审计

当前无原生作业需要等待：32k worker2088/launcher16444已正常退出并清空owned server。原case0 deadline900控制/66planfail/660interventions；原case1 success333、case4 success340。共1573控制/1665世界ticks(92reset)/158NN/3起点，raw、NN完整重建及engine全通过；58次精确失败缓存命中。公开 `reports/native_steering_budget32k_pilot_20261104.json`。case0起步22510节点成功，执行24个macro后因更换子目标规划失败停车；不接受为完整修复、不重算旧4/12。

关键新增诊断：D:/ThesisNativeArchive_20261104/steering_budget_case0_policy_switch.json，前25次NN与规划精确重建。decision23/24均放弃仍可复用参考；decision24旧目标(30.9139,7.48894,.25558)仍可复用，但当前所有state-relative动作都不解码到这个固定目标，31个context bits全0。不是简单再增大熵或规则锁死动作；可版本化重复选中同一option时保留已选世界坐标目标，让PPO通过显式重复动作选择继续参考，再在固定低层下真实适配。该新option语义尚未实现/训练，不得宣称学习收益。

之前全量1361通过属于v1；最新缓存/STOP等32项专项通过10.23秒。当前HEAD05c04e9已推送，400dfa4 hostedCI成功。不可重启已关闭批次；原full_transfer known24已abandoned_before_launch。

# 最新状态：2026-11-04 11:08 低预算回归已关闭；32k三场pilot运行

旧full_transfer协调器26208及static worker26140均已结束。主动STOP按guard正常关闭，完整保留40次同输入7200节点timeout/400制动静止控制；cooperative_stop_raw_audit.json验证全部400控制与engine回执。不是基础设施故障或完整case结果。known_validation prepared已标abandoned_before_launch，禁止误启动。公开报告 `reports/native_steering_transfer_regression_stop_20261104.json`。旧两场成功pilot仍保留，不能把它推广为完整方法成功。

同实测case0起点离线32k预算在22510节点、20.858秒找到路径；三种连接/起步原型仍在7200节点失败，均留D盘且从runtime移除。当前唯一原生任务 D:/ThesisNativeArchive_20261104/20261104_steering_budget32k_H_pilot，预注册原case0/1/4，worker2088 creation_epoch1793761636.161（身份见worker_process.json）、launcher16444。v2只增加节点上限至32000并按完整静态几何、配置、起终姿态、实测轮角和速度上限缓存精确确定性失败；不量化、不替换动作、不缓存成功、动态/墙钟预算旁路。用于避免重复40次无信息搜索，计算成本必须披露。32项缓存/STOP负检查等专项通过10.23秒，另组29通过6.04秒。全量1361通过是之前v1源，不能冒称v2全量通过。当前pilot尚无完成结果，须检查OS身份、日志后继续，禁止重复启动。

# 当前运行：2026-11-04 10:49 完整开发清单转角规划迁移

已启动 `scripts/run_steering_transfer_validation.py`：execution D:/ThesisNativeArchive_20261104/20261104_steering_full_transfer，实际协调器26208（身份 D:/ThesisNativeArchive_20261104/steering_full_coordinator_identity.json），venv外壳3600。两个batch均已准备：同前缀_static/known_validation，全部12/24原清单不删改，同一source_hashes、同一冻结bf6ef925模型、零新增PPO。先static，再known验证，然后每批raw/start/staticNN/engine核验；无自动重试。检查status.json、failure.json、complete.json及OS身份，不得重复启动。queued激活仅刷新对应批次时限，注册单批4h上限、D盘12GiB/内存4GiB/GPU85℃保护。

已修复实测转角略超命令max_steer时的输入误拒绝：保留有限且绝对值小于pi/2的实测角，目标命令仍max_steer限幅，完整路径/曲率检查保留；新增专项断言，27项通过5.95秒。此次仅输入域修复发生在全量1361通过之后，未虚称再次全量回归。两场成功pilot源已提交推送71e2031；完整清单源是本次输入域修复后的另一个冻结快照，不得混淆。

# 最新接续：2026-11-04 10:45 转角连续规划两场原生pilot成功

D:/ThesisNativeArchive_20261104/20261104_steering_transition_refinement_H_pilot 已正常关闭，8252worker/8828launcher均退出、server_cleanup=0。原case1成功333控制、原case4成功340控制，673控制/734ticks(61reset)/68NN/2起点全部raw、重建参考、NN和engine通过。报告 `reports/native_steering_transition_pilot_20261104.json`。新增转角连续规划从实测转角出发、按max(当前速度,所选速度,0.1)约束空间转角变化；关闭不满足该约束的analytic connectors并绕过pose-only memo。PPO冻结、实际道路/评分不变。并非学习收益或完整静态清单分数。27项专项通过；全量CPU 1361通过、1skip、4 CARLA deselect，261.06秒，XML D:/ThesisNativeArchive_20261104/steering_full_cpu.xml。

需继续完整开发清单检查。注意实测轮角可能略超过命令max_steer（case1终点0.50362rad>0.5），当前规划输入严格上限可能在其他场景触发；应接受有限且小于pi/2的实测角度而保持目标命令限幅，保留轨迹曲率验证，不能剪改实测状态。原case4旧失败和4/12均保留，未触碰最终确认集。

# 最新接续：2026-11-04 原生道路参考修正v1已关闭

原型真实运行case1成功332控制、case4仍越界258控制；590控制/651世界ticks/60NN/2起点全部raw、推理重建和engine核验通过。CPU1358通过/1skip/4CARLA deselect。完整报告 `reports/native_reference_road_refinement_pilot_20261104.json`。当前没有该批作业需要等待。原4/12不重算，未新增PPO。参考重规划初始曲率与实测转角不连续，是下一项待验证的执行接口假设；不能宣称修复。

# 当前接续状态（2026-11-04 10:05）

目标不变，研究验收UNMET。继续当前分支，不重建、不恢复瞬移、不新增子代理。以下是实际已启动状态；继续前须复核OS进程身份，不能仅凭状态文件重启任务。不要触碰Desktop/papper其他项目进程。

## 熵0.02课程全部关闭；当前只有模型场景预检

协调器72044、训练worker30936、静态19196、known67216及各owned server均已正常结束；三批require_finished_parent复核通过，无需重跑。execution `runs/reference_entropy002_memo_execution_20261103/complete.json`确认static4/12、known24/24。训练10485控制/1056决策/330Adam/18完整回合，11073世界步=10485+588reset，RMS不变、32Adam状态各推进330、ent_coef0.02。原始训练帧与engine全部通过。模型bf6ef9257ecbe619d7629c7c4baefeff4c60dd639da0fe01b74f0fe3a601f23a，source ac04d20。

完整评估15745控制/36起点/1276静态NN，全部raw/NN/engine通过；原v4口径保留。只比上一轮多成功case0，仍低于静态70%目标及零PPO BC教师5/12，不证明反馈或PPO优势。公开 `reports/native_reference_entropy002_endpoints_complete_20261104.json`。新课程训练耗时1565.887秒，原训练5005.225秒；策略与状态历史不同，不把时间差归因于缓存。实际训练253次cache_hit另列为计算记录。

case5原局部可达的4个子目标均不能两段接到终点；半分辨率/8倍节点上限也在38717节点open_set_exhausted，8个±0.4rad朝向提案均未得到全程路径。公开 `reports/native_reference_scale100_case5_model_reachability_20261103.json`，全部输入/注册参数保留在D:/ThesisNativeArchive_20261103。失败搜索不证明物理不可能，也不能假定只调熵就解决。冻结几何图 `D:/ThesisNativeArchive_20261103/case5_frozen_geometry_20261104.png`揭示障碍与分隔带之间狭窄通道；是模型图，不是RGB录像。

已完成 `scripts/preflight_native_static_placements.py`：12布局×9摆放=108提案，30个模型可行，分布case0/1/3/4/6/7/8/10/11；case2/5/9无通过者，保留全部拒绝。公开 `reports/native_static_placement_preflight_20261104.json`；D:/ThesisNativeArchive_20261104/static_placement_preflight_v1含完整注册、进度、全部路径。变换及相关hazard/memo专项30通过10.43秒。原清单不删除、不重算。

新增真实迁移pilot已关闭：D:/ThesisNativeArchive_20261104/20261104_case7_placement10_frozen_H_transfer_pilot。取case7首个模型可行提案(distance10m,lateral0)，原是13m；冻结同一bf6ef925模型，不新学习。327控制/33NN/357世界ticks、success、0planner failures、2interventions，raw/startNN/allNN/engine均通过，owned server正常清空退出。公开 `reports/native_case7_placement_transfer_pilot_20261104.json`。这是不同物理条件的工程可执行性复验，不是原case7重算或RL收益，原4/12仍保留。当前无本任务CARLA或预检作业需等待，不得重启旧PID。

新30清单迁移验证已全部关闭：新增batch `D:/ThesisNativeArchive_20261104/20261104_model_supported_placements29_frozen_H`，旧pilot复用1场。29场22成功/5deadline/2bounds，加pilot为23/30、2/30bounds；合计12402控制/1252NN/30起点全部raw、NN、engine通过，原4/12不改。报告 `reports/native_model_supported_placements_complete_20261104.json`。新29中的索引2/7/15/19/22/23/27与旧静态12条件重复，已明确标为重复测量并计费，不能当新增独立条件；整批均已知开发、零新最终条件、零PPO更新。无活跃本任务CARLA进程需等待。

原生SDK只读参考检查：`scripts/probe_reference_road37.py`使用冻结MapSHA/lane_projection，校准旧case4非Driving点；无Client/tick。30参考共6226pose/74712足迹，名义均Driving；横向±.05m、heading±.05rad九组合共672408点，9参考出现非Driving，包括case4两摆放各8点。报告 `reports/native_placement_reference_road_probe_20261104.json`。输出记录实际CARLA float32查询点与请求double点，原始输入/精确重查输出均D盘。有限采样不证明连续安全；不能在观察失败后过滤这30场或重算评分。

完成已注册case4模型诊断：D:/ThesisNativeArchive_20261104/case4_robust_planning_model_prototype。仅将原v4判定为非Driving的点对应的规划栅格约束加入模型，得到新路径，其九组合SDK重查0miss；仅增碰撞包络(.06m纵向/.25m横向)仍1miss；点约束加包络0miss；全cell约束加包络无路径。四种结果全保留，见 `reports/native_case4_planning_constraint_prototype_20261104.json`。未部署、未新物理执行，不宣称修复或RL收益。

下一工程验证应将已验证非Driving点作为新版本的规划约束并有限次重规划，实际任务v4边界/评分保留原语义；规划视图与实际bounds必须分离，不能因模型约束而伪造传感器或改旧成绩。各方法须共享该能力、披露额外计算，冻结PPO替换低层仅列迁移，再按需版本化重训。先验证原case4实际闭环，而不是再盲扫熵/参考系数或过滤失败。最终集仍未开启。

新增注册/映射/nominal足迹专项21通过3.01秒，XML D:/ThesisNativeArchive_20261104/placement_inventory_final_cpu.xml。新增脚本 `scripts/prepare_native_placement_inventory.py`只取所有独立模型可行提案；复用完整pilot不要求成功；源码与样本/模型/回执全部绑定。当前全部新任务已关闭，不要重复启动65264/66668等旧PID。

## 旧静态对照已全部结束，禁止重跑

协调器38744及worker52508/guard64696/server33076均已结束；execution `runs/static_remainder_after_reference_v4_execution_20261103/complete.json`存在，H0缺项8场和H_teacher完整12场的raw、全部起点NN、严格发动机回执全部通过。旧H0补齐后为4/12；教师5/12（6deadline、1bounds）、7339控制。新报告 `reports/native_static_H_H0_teacher_complete_20261103.json`，脚本可选 --teacher；原双策略报告原样保留。

教师是896步行为克隆监督初始化、零PPO的神经网络策略，不是直接规则B3。与旧H/H0共享冻结源码、校准、环境、全部12物理条件；教师模型/归一化与原注册哈希完全一致，零新增学习。没有证据支持PPO收益，不得把该单seed开发对照包装为正式主结果。只读真实分析执行通过；篡改raw绑定、出现PPO更新、缺严格发动机回执三种负检查均被拒绝，无新驾驶。

原损坏H0仅case0..3有完整证据，剩余case4..11已补齐。损坏文件及未知成本继续保留，没有合成evaluation.json，没有重跑已验证成功回合。公开 `reports/native_static_interrupted_prefix_20261103.json`。

## 新版关闭端点的公开预发布与完整公开复现已完成

v0.3.0-dev.20261103-reference-context，source commit f0fcba1e82ab0545989da00408a08644c5f77ad5，URL https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/tag/v0.3.0-dev.20261103-reference-context 。isPrerelease=true/isDraft=false，8资产均上传并经匿名HTTPS下载逐字节SHA核对，总52081782字节。模型/norm/raw原字节不变，辅助JSON路径转换7/6项有台账；没有CARLA安装/完整地图/SDK/环境。

两个capsule分别616/638文件、21691425/28814284字节，SHA static=87a766612046fc5f6a3f72918eb32fff7cd4a46c722eacceae2fb183826129c2，known=cc47c7d24320e1428467b3982447a59b7cbe7c386870c687fdd029b59091618f。本地预发布重建通过后，从新的GitHub clone f0fcba1及匿名下载件再完整重建：32060控制/72起点/2611静态NN/strict engine/两PNG逐字节一致，复现receipt本身也与已发布receipt逐字节一致。不是新驾驶或新增训练seed。完整新本地环境重装未重复执行；依赖安装由clean GitHub CPU CI另验，当前使用既有locked3.11runtime。

源码修复了旧复现器固定走4705审计器的缺口：按声明的context版本显式选reference审计；未知版本拒绝，额外重建全部静态动作、RMS不变及发动机回执。旧schema分支仍用原审计器。专项43通过7.39秒，CI41628896576成功1311通过/18跳过/4CARLA排除179.71秒。完整实例复现已完成，不要重复旧broader B1或这些端点全量重建。

本地资产 D:/ThesisNativeArchive_20261103/reference_context_closed_endpoints_20261103；匿名下载 reference_context_public_download_20261103；新clone reference_context_public_checkout_20261103；完整公开重建 reference_context_public_static12_rebuild_20261103及reference_context_public_known24_rebuild_20261103，两个工具session6879/74600已退出0。公开凭证 `reports/native_reference_context_public_download_verification_20261103.json`。这些端点capsule不重复上游training-only raw/失败祖先，祖先成本仍在learning报告；scale100新H不包含在该预发布内，其关闭结果见下，研究验收仍UNMET。

## 旧H/H0静态全12结果与独立失败解剖（17:25）

`reports/native_static_H_H0_recovered_complete_20261103.json` 及 `scripts/analyze_native_static_recovered_pair.py` 已实际完成。H3/12（8deadline/1bounds），H04/12（7deadline/1bounds），完整case控制8306/7761。H0仅组合原已核验case0..3与新case4..11，模型/归一化字节、校准、所有冻结源、route/hazard一致；原4起点独立NN重建也通过。H0已知消耗至少7850控制（含未完整保留的尾段），未知尾部仍披露。当前是旧4705输入模型，不是新context模型，不证明反馈收益。

共同case4中，两方法101帧state及command逐位相同，11次action均26，首次plan成功后10次reference_reused，MPC全部solved，无planner/MPC失败或干预，但v4Driving足迹探测一角点miss。不能将全部失败归为NN切换参考。进一步离线复核：模型道路余量约1.194m，该实际角点与模型车身角点仅3.14微米差，却被栅格标可行驶；同哈希本地OpenDRIVE和同一冻结lane_projection算法复现该点非Driving、五个栅格采样均Driving，单格17x17采样中15点非Driving。扩大同lane局部s搜索未找回该点，不能归因于小范围投影搜索。

只读引擎射线在同点和四周±2cm共5点均返回Roads；已核验当前owned server33076创建时间/进程、端口及Map哈希，未调用tick/control/spawn。它说明lane近似与road mesh存在分歧，不证明车道合法性或连续扫掠安全。原v4 out_of_bounds统计和所有失败均不改。不得通过增大容差或重算旧分数达标。若后续修复geometry，须新环境版本、全方法共享、另行实际验证，并区分工程收益与PPO收益。上述context评估均按v4原几何冻结执行，关闭结果见下。

可复核证据 `reports/native_static_shared_road_disagreement_20261103.json`，便携输入 `protocols/native_static_case4_road_disagreement_input_20261103.json`，脚本 `scripts/diagnose_native_road_boundary37.py`（只加载本地Map无Client）、`scripts/probe_native_ground_surface37.py`（保留只读owned句柄，仅语义射线）。完整原始探针与密集采样在D:/ThesisNativeArchive_20261103/road_boundary_probe_20261103，未改CARLA安装资源、环境或当前物理actor。

## scale100单H训练与全部端点评估已结束，禁止重跑

协调器48660、训练worker18496、静态worker4864、known worker57848均正常结束，无当前物理作业。execution `runs/reference_scale100_after_scale1_endpoints_execution_20261103/course/complete.json`存在；三批均guard_final exit0、owned_empty_server_stopped，2100..2102无监听。旧PID不可用于新任务观察或清理。

实际训练10501控制/1056决策/330Adam/18完整回合，世界11089=10501控制+588reset；请求10240，overshoot261<=319，未遍历全部48条件。32个Adam状态各推进330、RMS不变，全部训练raw及engine通过。模型15ba2e1b17a99086d07eb031babe0ce5f008afb0516c208928f5035992969753，norm680faf1c43af1f83391d0fcaddd4f6d0aae44e8007ae075ad0a3c6bf02e3d2e3。

完整静态12场为3成功/8deadline/1bounds，8303控制/832NN；known24为24成功、7990控制/498静态NN。合计16293控制/36起点/1330静态NN全部raw、RMS不变及严格engine核验通过。报告 `native_reference_scale100_static_complete_20261103.json`、`native_reference_scale100_endpoints_complete_20261103.json`。scale1 H同为3/12、H04/12、零PPO BC教师5/12；这不是新增H0scale100实验，不能归因或宣称优势，不继续盲扫参考系数。

首场完整90决策中，重建至首个失败共24次NN/规划完全匹配；仍可复用参考被放弃次数由旧5次变1次，但decision23还是action2/no_path、有效logit0.0792低于该状态额外所需约1.0123。不同权重和状态历史不构成固定状态因果比较。case2/5各90次均选26且全部plan_failed，初始参考context为空；7200节点预算下其他6/8个动作能返回几何路径，不代表可执行或任务成功。case4仍按原v4口径越界，11次NN/参考重建匹配，无planner/MPC失败。

`native_reference_scale100_failure_diagnosis_20261103.json`绑定原始前6场全部4场失败。不可变.inputs.json保留在D:/ThesisNativeArchive_20261103/reference_scale100_case{0,2,4,5}_switch_diagnostic_20261103.inputs.json。既有精确失败缓存的离线180次搜索重放：178命中、2实际搜索，状态/原因/cost/空路径及原节点数全匹配；报告 `native_reference_scale100_exact_failure_memo_replay_20261103.json`。这仅支持节省重复计算，不提高成绩。原训练5005.225秒、桥接等待98.800秒，离线单起点观测profile在D盘，不将它当原生性能基准。

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

最长8小时等前序确切退出0，要求static12/known24双方法完整、固定模型/raw绑定、全部NN重建与engine通过；注册的460个源码与全部初始化文件字节须不变，变化或前序失败即保留失败并停止，零自动重试。满足后才新建新截止时限的单H10240请求控制course，每320控制保存，继而完整12/24开发评估及自动raw/NN/engine审计；不新增监督、不改最终条件、不重跑H0训练。前序完整结果另存predecessor_complete.json。该源码逐字节门槛已在19:22发车前通过。发车后核验batch/source对应protocol.source_hashes，不能再把Root与旧等待注册快照比较后误判运行失败。在线worker仍使用原冻结460源。f0fcba1仅更新离线打包/复现/推理RMS核验，未改navstack或nativeworker；后续评估准备器会显式更新离线auditor的SHA，不改物理或策略。

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

旧适配H/H0在已知24场各24/24是不同模型、不同阶段，不可写成当前context版本成绩。五seed旧H120/120 vsH0119/120、PPO对零PPO教师120vs117差值区间都跨0；强B3已知24/24，尚无稳定新增反馈或PPO收益。旧静态三策略、context双方法静态及known24全部收齐；scale100新H课程已关闭，新探索课程状态见最上方。保持目标：至少5独立真实适配策略、健康充分训练B1、公平H/H0/B1/B2/B3及消融；至少120NEW去重最终条件、各核心30、未见布局/车辆；轻量85%/核心75%，CARLA80%/核心70%/不安全≤5%，H对B1预定差值与CI、PPO对强B3及协作贡献、原始数据/模型/失败成本/视频/公开复现。目标不降低，最终集未触碰，不能把当前单seed开发研究当最终完成。

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

# 低速车辆分层决策与控制研究

最新真实 PPO 训练已完成：共同停车辅助及同步控制版本全程固定，新增 10,240 个物理训练步，35 组 Adam 状态各实际增加 318 步；四条已知基础道路最终 4/4 成功、零碰撞/越界。11,639 帧控制及 1,399 个终点策略动作全部核验。训练中 26 次成功、3 次越界及预算截断尾段均保留。初始冻结模型原本已为 4/4，因此这里只说明 PPO 训练后保住基础能力，尚无 PPO 新增收益或五种子结论；决策周期仍有超时。
[训练、接管成本与时延](shared_parking.md#actual-ppo-with-the-wrapper-fixed)。

共享终端停车辅助的工程复验已完成：B1、H、强规则 B3 在同四条已知训练道路上均为 4/4 成功、零碰撞/越界。B1 的 1,409 个控制步中有 6 步被停车辅助改写；H、B3 各为 8 步。目标、成功判定不变，末帧仍须未被接管。这是共同低层能力的冻结策略迁移，不代表新增 PPO 学习收益、五种子基线或最终测试达标。
[实现、失败记录和核验证据](shared_parking.md)。

[公开预发布](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/tag/v0.2.0-dev.20261101-shared-parking)已从独立 checkout 无需登录下载并重建：2,816 个文件、18,936 帧控制及四张图表全部核验一致。下载所得 B1 模型又完成一次真实 CARLA 闭环复现：356 控制步、3 次停车辅助、成功停车，完整动作和实际控制回执核验通过。Linux CI 为 1,110 项通过，本机全量 CPU 回归为 1,127 项通过。这些是工程与复现结果，研究验收仍未达标。

此前的失败版本：B1 偏差响应监督后的驾驶复验为 2/4 成功、两次越界；当时追加的真实 PPO 完成 10,240 物理训练步和 320 次实际 Adam 更新，最终仍为 2/4、两次越界，未晋级。该版本的 11,933 帧全部通过原始状态与同步控制核验，1,693 个评估动作可从对应测量历史精确重建。研究目标保持不变，仍未达标。
[诊断、拒绝模型与实验依据](native_flat_feedback.md)。

新增实际控制回执审计发现：此前12,219帧中有51帧返回上一帧请求的控制量。已增加明确版本化的同步发送与逐帧核验；新1,271帧全部匹配，但相同冻结B1模型仍3/4成功、一次越界，未通过基础门槛。旧物理结果和公开日志保留；不能把时序修复当成PPO学习优势。
[发送契约与旧数据说明](native_control_delivery.md)、[同模型迁移结果](../reports/native_sync_control_delivery_transfer_20261101.json)。

本批独立公开复现已完成：563文件、12,219控制帧、四个初态动作及结果图逐字节核验通过；35组PPO Adam状态各增加320步，初始干预可精确重建、最终归一化统计保持不变。当前全量CPU回归1,101通过；最近完整GitHub CI1,084通过。研究总体验收仍未通过。 [公开复现证据](../reports/native_B1_exploration_PPO_public_verification_20261101.json).

原始训练道路的恢复采集虽为混合驾驶4/4成功，纯策略仍未通过：有训练误差投影约束时2/4、一次越界；固定数据、起始模型及512次更新预算，仅去掉该约束后1/4、两次越界。三个恢复方案共9,010真实控制均已审计，失败模型未晋级。另一次真实PPO探索试验也已完成10,240训练步、320次实际Adam更新，纯策略为2/4成功、一次超时、一次越界；12,219控制帧全部审计，模型未晋级，当前已无本批后台训练。
[原始道路结果](../reports/native_B1_original_TRAIN_recovery_collection_and_probe_20261101.json)、[投影约束对照](../reports/native_B1_original_TRAIN_unprojected_probe_20261101.json)、[真实PPO完整结果](../reports/native_B1_exploration_PPO_pilot_20261101.json)、[公开证据包复现](native_B1_exploration_reproduction.md)。

真实恢复采集与独立纯策略复验已完成：新增训练道路采集1,612控制步，其中教师850、策略762；全部状态输入、动作、接管选择及CARLA帧均已重建核验。固定512次辅助actor Adam提议后，纯B1复验为2/4成功、一次超时、一次越界，全部1,657复验控制帧核验通过，模型未晋级。此轮没有新增PPO更新或最终测试条件。本机全量CPU回归1,091通过、1跳过、4项CARLA排除。
[完整失败与成本记录](../reports/native_B1_executed_recovery_collection_and_probe_20261101.json)、[采集、拟合与复验命令](native_recovery_collection.md)。

真实PPO录像及完整模型/控制日志证据包已[公开预发布](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/tag/v0.2.0-dev.20261101)。独立checkout已无需登录下载两个证据包及四个视频，核验1,828控制帧、恢复两份策略、逐字节重建两幅图，并解码全部1,832视频帧。包含B1失败，尚未满足研究验收。
[实际公开下载与独立复现凭证、SHA256](../reports/native_PPO_RGB_public_verification_20261101.json)。
[两个公开模型/录像包的复现命令](native_PPO_RGB_reproduction.md)。

99标签B1修正已完成实测：2/4成功、两次越界；复用的重放对照为3/4、一次越界，该修正被否决。后续三项TRAIN保留/上下文拟合也均未通过预设门槛，未追加驾驶评测。当前数据修复区分真实执行教师标签和旧反事实标签，尚未得到健康基线。
[99标签实测](../reports/native_B1_three_episode_training_coverage_probe_20261101.json)、[保留门槛失败及成本](../reports/native_B1_retention_and_failed_gates_20261101.json)、[教师参考上下文命令核验](../reports/native_teacher_reference_context_20261101.json)。
仅真实教师标签修复也已完成：3/4成功、一次越界；1182控制帧及四个初态动作均已核验，仍未通过健康门槛。
[修复后完整失败结果](../reports/native_B1_executed_teacher_only_repair_probe_20261101.json)。

五个“仅教师初始化H”对照已全部完成：117/120成功，追加PPO后H为120/120，双方无碰撞/越界。配对平均+2.50个百分点，95%t区间[-2.13,+7.13]，p=.5；强规则仍为24/24，尚未证明稳定的PPO增益。42,109个新增教师评测控制帧均已核验，完整归因报告由独立目录逐字节重建。
[完整归因](../reports/native_broader_PPO_teacher_rule_attribution_20261101.json)、[独立本地输入重建](../reports/native_broader_teacher_attribution_independent_verification_20261101.json)。

一组B1已训模型的配对闭环诊断已完成：两组各追加512次actor监督更新，仅一组加入自身训练轨迹上的17个有效规划/MPC标签。两组均3/4成功、一次越界；训练拟合误差下降未转化为相对重放对照的成功率或安全改善，均未通过健康基线门槛。
[配方、成本、队列接替与待验证范围](../reports/native_B1_trained_trajectory_correction_preflight_20261101.json)。
[完整配对驾驶结果](../reports/native_B1_matched_trained_actor_correction_20261101.json)。

五个B1保留状态续训种子全部结束：分别3/4、2/4、2/4、0/4、2/4成功，总计9/20成功、十次越界、一次超时。追加51,200真实控制步、1,595次实际Adam更新，只有第一个通过基础门槛。完整任务阶段按原定规则跳过，不选择性放行。该方案尚未修复健康B1。
[第四个种子的完整失败轨迹与测量状态诊断](../reports/native_B1_seed44_lower_rate_failure_diagnosis_20261101.json)。
[实际门槛与未解决停滞](../reports/native_B1_first_lower_rate_basic_gate_and_stall_20261101.json)。首阶段五种子3/4、2/4、2/4、0/4、2/4的全部失败和成本继续保留。

新的同执行版本H/H0五种子训练及完整公开复现已结束：36个TRAIN条件、24个已知开发验证条件，各方法各种子请求20,480控制步。H共120/120成功，H0为119/120；双方无碰撞/越界，H0第44种子的静态条件15超时。按独立训练种子配对，平均差值+0.83个百分点，95%t区间[-1.48,+3.15]，探索性符号翻转p=1.0，尚无稳定反馈优势。仍是24个已知条件，不能算120组新的最终条件。
[完整配对统计](../reports/native_matched_broader_feedback_cohort_20261101.json)。

H/H0实际训练控制步为103,197/102,711，Adam更新为3,270/3,250；H第44、55种子各660次，其余端点各650次。评测干预总数2,020/2,730，主要差异来自一次H0超时；各个种子双方均成功的条件中，H都略慢。所有失败和成本保留，不以汇总成功率掩盖代价。
[完整成本](../reports/native_matched_broader_feedback_cohort_costs_20261101.json)、[独立目录逐字节重建](../reports/native_matched_broader_feedback_costs_independent_verification_20261101.json)、[超时及反馈敏感性诊断](../reports/native_matched_broader_seed44_failure_diagnostics_20261101.json)。

五个完整模型/日志包均已[公开预发布](https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/tag/v0.2.0-dev.20261101-feedback)。无需登录的公开下载共核验4,652个文件、287,811个训练/评测控制帧、240个测量初态动作及五幅完整条件图；完整五组统计字段、十份实际Adam/冻结归一化证明、每种方法五份不同策略和主图也已重建通过。
[完整公开复现凭证](../reports/native_public_matched_broader_feedback_five_seed_verification_20261101.json)、[完整配方](../protocols/public_native_matched_broader_feedback_cohort_20261101.json)。当前checkout可执行：

```powershell
.venv/Scripts/python.exe scripts/reproduce_public_feedback_cohort.py --recipe protocols/public_native_matched_broader_feedback_cohort_20261101.json --output runs/public_matched_broader_five_seed_rebuilt
```

被哈希引用的报告和清单在Git中保留原始字节，仓库配方与实际公开执行配方一致；修复前的导出仍保留。
[实际配方导出核验](../reports/native_matched_broader_exact_metadata_export_verification_20261101.json)。最近已完成CPU CI为1036项通过，新增导出保护相关44项测试通过。当前仍未取得健康B1、PPO独立价值、稳定反馈优势及新最终确认，不能宣布研究完成。

额外执行预测网络被否决：策略种子和道路同时隔离后，Brier为0.0518，不如简单历史失败基线的0.0463；独立目录重放所有预测完全一致。它未接入PPO、动作掩码或控制器。
[负结果](../reports/native_execution_prediction_cross_seed_probe_20261101.json)、[独立复验](../reports/native_execution_prediction_independent_verification_20261101.json)。

真实原生PPO录像已完成：四段、1,832个传感器帧、1,828个核验控制帧；H两条已知道路均成功，B1一条成功、一条越界。模型保持冻结，无录像阶段PPO更新。全部视频解码/哈希通过，已查看每段首、中、末帧；原场景3有树冠遮挡。这是单种子示例，不是最终对比。
[真实结果、初次启动失败及修复](../reports/native_PPO_actual_recordings_and_subset_fix_20261101.json)、[批次/模型/seed/原场景索引](../reports/native_PPO_video_catalog_20261101.json)。

前三个已完成基础课程种子的40个实际决策点已重建。两次失败在进入停车位置时仍为1.44/1.50m/s（目标上限0.25m/s），初始模型在同一测量状态下也缺少足够制动；另三次失败尚未进入停车位置，不能归结为单一停车问题。这是动作诊断，不能替代反事实驾驶或证明因果。
[原始核验、全部关键点与限制](../reports/native_B1_basic_motion_landmark_diagnosis_20261101.json)。

第11种子未达标课程的完整证据包现已公开下载并独立重建：553个文件、11,966控制帧、4个初始动作及参考图，包含越界失败和全部检查点；明确标注安全门槛未通过。

B1优化器诊断已完成真实训练数据的受控重放：同一1,280控制步，在原学习率下重现原检查点参数的最大误差约5.22e-8（数值一致，非逐位一致）；学习率从3e-5降至3e-6后，Adam更新从1次增至40次，KL及critic训练误差均更低。
独立目录重建的数据数组、全部比较结果字段完全一致，183项测试通过。
[独立核验](../reports/native_B1_matched_rate_replay_independent_verification_20261101.json)。
这尚不能证明驾驶改善。原课程与统一低学习率续训现均已全部结束；续训各追加10,240步，未增加监督拟合、选择种子或放宽零越界门槛，仍只有一个种子通过基础门槛。完整任务阶段未启动，全部失败和成本保留。

H/B2十个公开包的整体验证已实际通过：全部配对统计字段一致，10份Adam状态及冻结归一化核验、每种方法5份不同策略参数与主图逐字节一致；独立178项CPU测试通过。
[整体验证凭证](../reports/native_public_H_B2_five_seed_independent_verification_20261101.json)。
B1基础课程第11种子完成10,240控制步、9次实际Adam更新，评估3/4成功、1次越界，门槛失败；其余种子继续。不能把这一版本称为健康基线，也未降低标准来启动完整任务阶段。

B2五种子公开复现现已全部通过：133,687控制帧、60个初始动作及全部参考图已核验。
旧的基础课程直接转完整任务队列已在门槛失败后被替代。当前队列先统一低学习率追加基础训练；仅全部5种子达标后才进入完整任务训练，保留权重、Adam与RNG，最终策略交互量将为每种子40,960步。
[当前队列与状态保留核验](../reports/native_B1_lower_rate_extension_activation_20261101.json)。尚未启动完整任务子批次。

07:30更新：原课程进程已外部中断，留下2,408控制步、1次Adam更新，但无已保存的训练权重。原日志保留、损失单列；新独立批次已从原初始化实际启动，并改为每1,280控制步保存检查点。尚未完成基线验证。

11月1日更新：旧版追加拟合的5个B1种子完整结果为9/4/5/5/7（各12条件），
4个种子的基本运动门槛失败。23,610评估控制帧及5,120额外拟合更新均保留。
虽然五种子的TRAIN误差都下降，这没有带来稳定的实际驾驶能力，故不再追加同类拟合。
新版本已开始真正的基础运动PPO课程：五种子各10,240控制步，四个已知无障碍训练条件，
不增加监督拟合；在学习后检查至少3/4成功且零碰撞、越界。
这是基本学习健全性检查，尚不是健康的完整任务基线，也不能直接用于H/B1等预算优势主张。
[失败、成本与实际启动凭证](../reports/native_B1_maturation_rejection_and_basic_PPO_activation_20261101.json)。
B2第11/22/33种子的公开下载及独立原始数据、模型、参考图重建已经完成；第44/55种子随后也已全部完成。


10月31日23:22进展：H/H0五种子开发结果为58/60对57/60；均无碰撞或越界，
配对差值均值+1.67个百分点，95%区间[-11.82,+15.16]，尚不能证明反馈收益。
所需六个证据包现已全部公开。新补齐的H22包已实际公开下载并独立重建，
核验24,858控制帧、12个初始动作及完全相同的图；中断H0的68个文件和原始不完整GZIP均保留。
[公开复现凭证](../reports/native_seed22_completed_H_retained_H0_public_verification_20261031.json)。
从公开资产重建全部五组结果（使用全新输出目录，至少4GiB可用磁盘）：

```powershell
.\.venv\Scripts\python.exe scripts/reproduce_public_feedback_cohort.py --output runs/FRESH_public_feedback_cohort
```

全部五组结果的独立重建现已实际通过：六份此前公开下载包的凭证、清单与整包SHA256
重新核验，所有统计字段、10个Adam状态及冻结归一化、每种方法5份不同策略参数均通过，
主图逐字节一致。明确复用已验证的公开下载缓存；此前两个网络失败目录保留。
独立目录150项相关CPU测试及托管工作流均通过。
[完整独立复现凭证](../reports/native_public_feedback_five_seed_independent_verification_20261031.json)。

B2五个种子全部完成9/12、8/12、10/12、10/12、7/12。H/B2共58/60对44/60，
配对均值差+23.33个百分点，95% t区间[+3.43,+43.24]，精确符号置换p=.125。
这是12个重复开发条件的探索比较，两种统计方法的证据不同，不能冒充正式确认。
五个B2批次另有行人生成高度修正，H使用旧版本；该差值不能完全归因于规划器。
[完整配对与所有回合](../reports/native_H_B2_five_seed_development_comparison_20261031.json)。
B1新五种子的基本运动门槛检查已经自动开始；统一制动和频率消融继续排队。
这些仍是已知开发条件；独立最终测试和研究主张尚未达标。

以下为此前开发快照，按各报告的实验版本理解。

10月31日最新完整结果：v4几何上的真实 PPO 对照为 **H 12/12、H0 11/12**，
两者均无碰撞、越界，分别完成20,563/20,562训练控制步和650次Adam。
全部50,348训练与评估控制帧、24个初始策略动作已审计，服务器正常清理。
H0仅静态条件5超时；单种子一例差异尚不能证明反馈机制有效（精确检验p=1.0）。
[完整配对报告](../reports/native_v4_trained_seed11_feedback_pair_20261031.json)。

H的5个训练种子现已全部完成：累计102,688次真实训练控制、3,250次实际Adam更新，
开发评估58/60；全部125,451训练/评估帧已核验，5个初始Torch随机状态及最终策略参数均不同。
[5种子H训练凭证](../reports/native_H_five_seed_development_completion_20261031.json)。
这是12个已知开发条件各评估5次，不能称为60个未见条件或最终验收；H0预算配对重跑和B1队列仍未齐全。

后续高学习率seed33现已完整结束：H为8/12、四次超时；H0为6/12、六次超时，均无碰撞/越界。
[H完整端点](../reports/native_H_v4_seed33_high_lr_endpoint_20261031.json)、
[H0完整端点](../reports/native_H0_v4_seed33_high_lr_endpoint_20261031.json)。
核查发现该批保存的学习率是3e-4，参考版本为3e-5，
此前只核对环境配置的校验漏掉了这一差异。原结果保留，不混入同配置主队列；
已准备显式3e-5的33/44/55批次，并核对完整优化器参数及初始权重、归一化和随机状态。
[差异记录](../reports/native_H_seed33_initialization_gate_vs_high_lr_20261031.json)
与[准备核验](../reports/native_canonical_H_H0_lr_profile_preparation_20261031.json)。
修正后的seed33整批现已审计完成：**H10/12，H012/12**，均无碰撞/越界，
各20,561真实训练控制步、650次Adam；全部50,843原始控制帧、24个初始动作已核验，
服务器正常清理，第44种子自动开始。[完整配对](../reports/native_v4_seed33_canonical_feedback_pair_20261031.json)。
H条件5和7静态超时，差值-16.67个百分点、精确p=.5，不能宣称反馈收益稳定。
两次失败分别有81/68次规划失败和810/680次制动干预；H0均成功且无这些干预。
[完整状态核验](../reports/native_seed33_canonical_feedback_failure_diagnostics_20261031.json)
重建547/435次决策；固定H权重隐藏反馈只改变19次动作，在两次失败的180个已访问决策中均不改变动作。
这不是反事实驾驶结果，也不能把低碰撞率当作任务完成。此前等待审计的记录作为历史保留。
该批[完整公开数据包](../reports/native_seed33_canonical_public_verification_20261031.json)
已上传、无登录下载并独立复现：全部50,843原始控制帧、24个初始动作、逐字节一致的轨迹图，
以及982次反馈决策的全部非耗时字段。成功和失败均完整保留；
[哈希及复现命令](model_card.md#public-canonical-seed33-pair)。
第44种子也已完整审计：H12/12、H011/12，均无碰撞/越界，实际训练控制20,502/20,586次、
各650次Adam。[配对记录](../reports/native_v4_seed44_canonical_feedback_pair_20261031.json)。
11/33/44三个完整配对的[描述性中期结果](../reports/native_canonical_feedback_three_seed_interim_20261031.json)
平均成功率差为0个百分点，尚不能宣称反馈收益稳定。

B2已[准备5个种子的零PPO初始化](../reports/native_B2_five_pristine_training_preparations_20261031.json)：
与对应H的初始权重、归一化、随机状态逐字节相同，仅替换二维A*及合理的车体可行样条后处理；
保持奖励、MPC、信息、教师先验和每种子20,480物理控制预算。
[5次冻结代码实际恢复](../reports/native_B2_frozen_pristine_restore_verification_20261031.json)
验证了不同初始策略参数、空优化器与冻结观测统计。B2接续进程已启动，
[正等待现有B1队列完成](../reports/native_B2_five_seed_queue_activation_20261031.json)，尚未开始B2训练，
不能把初始化或旧的冻结权重换规划器结果称为已训练B2成绩。

[统一制动器接续队列](../reports/native_shared_brake_cohort_activation_20261031.json)已实际启动，
等待B2完成后评估H/B1各5种子、关闭/开启共20个冻结策略部署对照；目前6个批次准备好，尚无新驾驶结果。
保留原有动态轨迹检查，仅追加同一执行制动器，避免原shared_shield开关同时移除H动态检查的混杂。
[恢复核验](../reports/native_shared_brake_frozen_restore_20261031.json)确认同一模型两组参数一致、归一化未变；
这是部署干预，不是更换训练安全包装后重训，也不构成安全达标证明。

新增训练数据的[公开复现凭证](../reports/native_training36_measured_views_public_verification_20261031.json)
已完成：无登录下载18个分片、965个文件，在独立checkout重新核验20,521次训练控制，
覆盖36个条件、51个训练回合，重建4,156条标签且数据包逐字节一致，用时496秒。
失败、末尾未完成回合、原验证中断及H0未开始状态均保留。该证据不代表B1驾驶能力提升。
下载及一条离线重建命令见[英文README](../README.md)，无需启动CARLA。

另一条36训练/24隔离开发验证条件的分支现已完成：H为24/24，三类各8/8，无碰撞/越界。
行人初始生成高度故障修复后，使用原训练检查点完整重评，未重复训练H；原中断消耗保留。
H0对照也已完成：21/24，三次失败均为静态障碍超时，无碰撞/越界；20,561训练控制步、
650次Adam，30,292训练/验证帧与24个初始动作全部审计。配对差距12.5个百分点，
精确双侧p=0.25，尚不能证明反馈优势。[配对报告](../reports/native_v4_broader_feedback_pair_20261031.json)。
H0完整数据包已公开，并经无登录下载、独立环境重现全部30,292帧、24个初始动作和逐字节一致的
全结果轨迹图。[公开复现凭证](../reports/H0_v4_broader_public_verification_20261031.json)。
该结果不能与B1的另一份12条件清单直接比较，也不计入最终120组。
[恢复与完整验证](../reports/native_broader_walker_spawn_recovery_20261031.md)。
完整反馈输入核验重建了1,808次决策：固定权重与物理掩码，隐藏反馈后H有24/823次动作改变，
H0为0/985。这是输入敏感性证据，尚非成功率收益的因果证明。
[完整核验](../reports/native_broader_feedback_reliance_summary_20261031.json)。

B1已在相同36训练/24验证清单中完成20,480真实控制步、640次Adam，验证为9/24：
弯道4/8、静态2/8、动态3/8；9次越界、3次碰撞、3次超时，安全与泛化仍不合格。
全部30,839原始帧和24个初始动作已审计并经公开下载独立复现，全部失败保留。
[公开凭证](../reports/B1_v4_broader_public_verification_20261031.json)
和[配对分析及限制](../reports/native_v4_broader_H_B1_pair_20261031.json)。
保留原小学习率试验的初始权重、归一化与随机状态，学习率3e-6。
[实际初始化核验](../reports/native_B1_broader36_validation24_preparation_20261031.json)。
其余四个B1种子的v4初始化也已准备并核验，权重、归一化及随机状态逐字节保留；
尚未启动，不计为四个已训练策略。[核验记录](../reports/native_B1_remaining_seed_v4_preparations_20261031.json)。

旧权重单独迁移到v4后为H9/12、H010/12，旧版正差距并不稳定。B1在v4上的真实PPO
仍为 **7/12、零碰撞、零越界、5次超时**，其公开模型/原始数据已独立复现27,737控制帧。
四次失败提前停在距终点1.45–2.37米处，原成功距离阈值未改。两臂等量监督拟合现已完成：
重放8/12、3次越界；加入训练阶段接近终点样本后10/12、2次越界。二者均零PPO更新，
不能称为PPO收益或合格安全基线。全部10,193评估控制帧及两份公开数据包已独立复现。
相同20,480控制步的真实PPO对照现已完成：重放初始化为4/12，接近终点初始化为6/12，
两者均4次越界、1次碰撞，仍不合格。两份公开训练包及完整对照已从公开下载数据独立复现，
全部53,324控制帧和失败均保留。较小学习率的新试验现已完成20,480真实训练步、610次Adam：
10/12成功，2次越界、零碰撞/超时，基础场景4/4、静态与动态各3/4。
安全仍未达标，也未超过监督初始化的10/12成功数，不能据此宣称PPO价值已验证。
[完整小学习率结果](../reports/native_B1_v4_approach_lower_lr_complete_20261031.md)。
[完整PPO对照](../reports/native_B1_v4_actor_fit_PPO_pair_20261031.md)。
[完整对照](../reports/native_B1_v4_approach_pair_complete_20261031.json)。
这些均为开发条件；五个独立训练种子、健康完整任务B1及全新最终条件等验收仍未完成。

以下为较早的执行记录：

2026-10-31最新：[完整原生诊断报告](../reports/completed_native_diagnostics_20261031.md)。B1固定奖励尺度版本20,480真实训练步、208次Adam后的冻结成绩为 **2/12**，5次越界、1次碰撞，仍不合格；27,897训练/评估帧全部审计。真实CARLA固定参考对照 **MPC4/4、Pursuit/PI4/4**，3,811帧与原始RPC完全绑定；本次Pursuit的跟踪误差和jerk更低，不能宣称MPC全面占优。两份原始数据包均已公开，主方法仍保留PPO→Hybrid A*→MPC。H冻结重评已完成9/12，三个核心类各3/4，零碰撞越界；B1的1280步采样对照已实际训练，后续受限任务排队。健康B1、五个独立训练种子及120组全新最终条件等研究验收仍未完成；以下较早的运行状态均属历史。


2026-10-28 开发结果更新：B1 累计102,400真实控制步后的冻结评估为 **2/12**，含7次越界和1次碰撞，仍不具备健康基线资格。[完整失败报告](../reports/native_B1_long_PPO_endpoint_20261028.json)。B2 body-spline 部署评估 **7/12**，没有新增训练，不能据此声称等计算量对比。新共享物理筛选机制的 H/H0 监督初始化各 **11/12**，尚无新增 PPO；五个固定种子的15个初始化已准备完毕，不能算作五组完成训练的策略。critic 学习率单变量对照已登记排队，结果待实测。

[十二条件开发对照已完成](../reports/native_twelve_case_completed_20261027.json)：同四条已知道路包含弯道停车、静态绕障、行人横穿。规则 B3 为 **12/12**；额外教师监督初始化、**PPO 更新为零**的 H、H0 各为 **11/12**；直接控制 B1 为 **4/12**，其中六次碰撞或越界。B1 仍不合格，这些结果尚未证明 PPO 价值、分层优势或反馈贡献。

[真实 CARLA PPO 双方法批次](../reports/native_twelve_case_PPO_complete_20261027.json)已完整结束：**H11/12、H0为10/12**，实际训练20,541/20,623控制步，各650次Adam，合计 **51,429原始控制帧**审计通过。H保持监督成绩，H0退化一个条件；单种子差异不能证明反馈贡献，也未胜过12/12的强规则。[保留候选顺序的B1](../reports/native_twelve_case_ordered_B1_endpoint_20261027.json)为 **3/12**；[改变拟合划分后](../reports/native_twelve_case_phase4_B1_endpoint_20261027.json)为 **6/12**，均未通过原完整任务初始化门槛。纠偏拟合版现已完整补评：7/12，但四次越界，仍不适合作稳定初始化。另行披露的基本运动训练入口允许B1进行[真实PPO学习](../reports/native_B1_basic_motion_PPO_endpoint_20261027.json)：20,480控制步、640次Adam；冻结评估退化至4/12，不能作为健康基线。[相同初始化和预算的KL停止诊断](../reports/native_B1_KL_PPO_endpoint_20261028.json)现已结束，仍为4/12；累计102,400步的B1续训正在实际运行，尚不能称为合格最终基线。

[此前较小探索强度 H/H0 PPO 对照已全部结束](../reports/native_shared_small_exploration_complete_20261027.json)，均为 **2/4**，合计 **46,493** 个真实训练与评估控制帧已审计，未形成总体反馈优势。[较小噪声 B1 PPO](../reports/native_B1_small_exploration_20261027.json)仍为 2/4，其[公开模型及原始数据包](model_card.md#public-smaller-exploration-b1)已实际下载并独立复现全部 23,025 控制帧。

[早期混合教师数据的完整独立重建](../reports/native_mixed_measured_full_reconstruction_20261027.json)现已完成：3,617 控制帧、五个数组及整个 NPZ 文件逐字节一致，新增驾驶、拟合、PPO 更新均为零。原 600 秒失败记录保留；新尝试事先登记 900 秒预算，实际耗时 809.75 秒。[动态高层观测核验](../reports/native_hierarchical_dynamic_views_20261027.json)。

上述均为开发条件。五个独立训练种子、120组全新配对最终条件、健康且充分训练的B1、完整消融、相对强规则的PPO价值及未见迁移仍待完成。新静态障碍距起点13米，不能据此宣称原10米失败已修复。源码ae3d238的GitHub CI已通过 **573项CPU测试**、15跳过/4项CARLA排除；20项续跑检查也在独立固定环境通过。CPU验证与CARLA实验单独报告。

[执行及队列入口](native_execution.md) · [当前交接状态](handoff.md)

[真实PPO模型及原始数据包](model_card.md#public-twelve-condition-ppo)已实际上传、无登录下载，
在独立checkout复现全部51,429控制帧、24个策略起点动作和逐字节一致的包内结果图。
另外[三组规则/初始化证据包](model_card.md#public-twelve-condition-development-evidence)
已独立复现18,591控制帧、36个起点动作和两张结果图。

Earlier development snapshots follow.

最新[监督后PPO保持性实测](../reports/native_supervised_retention_20261027.md)：监督初始化直接模型从3/4退化至真实PPO训练20,480步后的0/4；23,427帧已审计，归一化统计完全未变。相同初始权重/RNG的共享奖励配置对照正在执行，尚无PPO改善结论。

最新[实测直接控制诊断](../reports/native_measured_bc_20261027.md)：额外教师监督初始化的连续直接网络在四条已知完整静态道路上为 **3/4**，1,976控制帧已审计，PPO更新为0。对应真实PPO续训已启动，结果单独核算。弧线候选PPO版本H/H0/B1均0/4，已否决改进结论并保留全部69,514帧；尚未通过正式研究验收。

最新[实测车身表示与共享奖励训练](../reports/native_static_body_learning_20261027.md)：保守矩形并集使规则B3静态已知道路从2/4升至3/4；终点筛选PPO经冻结补评仍0/4。新共享奖励H已完成20,761真实训练控制步，冻结已知道路2/4；H0恢复后累计20,480谱系控制步，冻结0/4；B1仍在训练。此次比较尚不完整，尚未证明PPO贡献或健康完整B1。此前全失败模型及七批规则原始数据已实际发布、公开下载并在独立检出复验，见[模型与复现入口](model_card.md#public-hazard-evidence-2026-10-27)。

最新[原生障碍与控制频率开发](../reports/native_hazards_cadence_20261026.md)：10Hz直接PPO完成20,765真实训练控制步，已知6m任务仍0/4。原生车辆/行人任务与逐帧证据已接入；参考复用后B3静态2/4，修正出生位置后B3行人4/4。全新障碍H/H0/B1仍各0/4，必要终点筛选正在隔离实测；研究验收尚未通过。

最新[完整道路评分头匹配开发对照](../reports/native_actor_head_20261026.md)：共享头 **1/4**、独立头 **3/4**，单个seed11、四条已知道路。13,259控制帧及500次冻结决策已核验；失败道路仍耗尽搜索预算并超周期，不代表正式优势、健康B1或实时性达标。

[当前状态、复现和后续目标](current_status_20261026.md) · [可直接复制的接续提示词](continuation_prompt.zh-CN.md)。完整原生道路模型/数据已公开发布并从公开下载独立复现；完整研究验收尚未通过。

[GitHub CPU回归已实际通过](https://github.com/Leo-7GOAT/Intelligent-driving-system/actions/runs/40281946359)：源码e0f9099在Ubuntu24.04锁定依赖环境302通过、3项平台/显式条件跳过、4项CARLA测试排除；独立固定Windows环境304通过、1跳过、4排除。CPU验证与实际CARLA实验分开报告，见[CI范围与复现](cpu_ci.md)。

新增默认关闭的[完整车体道路 SAT 批处理后端](../reports/road_collision_batch_20261025.md)：已保存开发状态的碰撞标志与完整搜索路径保持一致，离线搜索计算耗时下降；已完成的 CARLA PPO 批次使用冻结的原版 scalar 后端。离线耗时不能代替实际完整周期或实时性验证。

已补齐[初始权重与训练权重的实际对照入口](native_weight_contrast.md)及完整帧/决策审计器。本地完整CPU回归295通过、1跳过、4项CARLA排除；实际H0初始权重重建、保存/重载与训练日志核对通过。**该实际CARLA权重对照已启动但中断**：初始组0/4，训练组部分记录，缺少整批完成与正常清理证明，不能形成配对结论。两个权重组共用训练后的冻结归一化统计，因此仅检验该统计条件下的权重贡献，不是全新归一化的未训练系统成绩。

[完整全新原生道路开发批次](../reports/native_route_complete_20261025.md)已结束：四条已知训练道路上 **H0为3/4、H为2/4、B1为0/4**，训练控制步分别为20,702/20,665/20,621，各130次优化器更新。全部 **67,289个训练/评估控制帧**审计通过，534次确定性决策由保存模型重建。H存在策略主动等待失败；B1四路全部越界，尚非健康最终基线。训练中的H一次、B1的39次越界均保留。原worker/守护已结束，自有空场景服务已停止。这些单种子开发结果不支持反馈改进，也不能证明PPO贡献、未见泛化或分层优势。

[目标朝向搜索引导](../reports/native_dubins_guide_20261025.md)另在规则B3执行中完成4/4已知道路，1,319控制步审计通过。此前批处理实现的完整CPU回归267通过、5条件跳过，后续29项只读诊断/审计针对性测试在两处环境通过；当前CPU结果见上方链接。研究验收仍未完成，进程与结论边界见[handoff](handoff.md)。

2026-10-24 18:38 更新：[32/256 决策 rollout 对照](../reports/native_rollout_horizon_20261024.md)六支全部完成，均为 **0/4**。H0 平均进度 **1.84/1.15m**，H 为 **14.99/14.94m**，B1 为 **2.20/6.64m**。全部 **639,009 个本段训练及评估控制帧**已审计，超额 rollout、失败及历史中断成本均保留。随后实际执行的 [CARLA 规则对照](../reports/native_rule_protocol_20261024.md)在相同 H 低层与实测初态下完成 **3/4**，审计 **1,977 个控制帧**，没有 PPO 更新。第四个条件在没有可生成运动参考的候选状态下持续等待。这支持继续排查高层学习和终端候选覆盖，不构成分层优势或 PPO 价值证明。两批均已正常结束并清理自有服务；道路仍是开发用训练道路，六支 PPO 是 seed11 的相关分叉。[此前裁剪对照](../reports/native_clipping_pair_20261024.md)同样全部0/4，独立 actor/critic 裁剪未被采用。

源码 `ac59741` 完整回归 **203 项通过、5 项条件跳过**，17 项恢复及 rollout 测试也在独立检出通过；`530d168` 的三项离线动作可用性测试在两处通过。H0 配对结果、226 个物理回合奖励重建及 16 个实测状态的轨迹查询均独立逐字节复现。八个最终 H0 决策均选中规划失败动作，同时仍有可生成局部轨迹的备选动作；这不等于备选动作实际执行成功。此前约 100k 的 [完整原生模型与数据资产](model_card.md)已发布并从 GitHub 下载复验：895 个文件、三份原始审计、十二个起始动作和轨迹图均核对通过。该版 H0/H/B1 分别保留 **100,160 / 100,160 / 100,050** 训练控制步，均为 **0/4**。研究验收仍未完成。

以下为历史开发快照；其中当时的“运行中”状态不能覆盖 [handoff](handoff.md) 中最新核验的进程和批次。

2026-10-24 更新：[道路与候选动作表示实验](../reports/road_candidate_development_20261023.md)的 H0/H/B1 约 2 万步端点已完成，四条训练道路评估均为 **0/4**。逐方法原始帧审计覆盖 23,760 / 23,760 / 10,500 个训练及评估控制步。B1 中断恢复后的模型谱系为 20,061 步，已消费交互下界为 29,606 步，未知中断尾部仍保留，不能当作同成本优势比较。三方法接续至约 10 万谱系控制步的真实 CARLA 批次已经启动，设六小时上限和资源看门狗。源码 `4b7db60` 已推送并核对远端，独立检出的固定依赖环境 **163 项通过、5 项条件跳过**，三份原始帧审计逐字节一致。新模型和数据尚未上传发布资产。

归一化续训已完成双方累计 302,080 步：持续更新统计为 **4/10**，冻结统计为 **3/10**；冻结统计在 20 万步时的优势未延续到 30 万步。两分支来自同一监督初始化，不能算两个独立训练种子或纯 PPO 对比。完整研究验收仍未达到。

最新[恢复状态监督诊断](../reports/recovery_aggregation_20261023.md)：监督初始化后的普通 PPO 从 **4/10 退化到 0/10**；仅降低探索噪声最终也为 **0/10**，含六次碰撞。其余条件对应一致、训练期间冻结观测统计的分支完成 102,400 步后为 **4/10，停车探针 8/8**；静态、窄通道与未见布局仍全部失败。所有预定检查点、原始控制和监督成本保留。这支持继续验证能力保持，尚未修好完整任务基线，更不能当作纯强化学习对比或分层优势。源码 `eee6c8a` 在本地及独立检出均为 **139 项通过、5 项条件跳过**。

此前当前状态标签对照在相同初始化和19,400次监督更新下为1/10，对照为2/10，修改标签并未修复任务能力。全部32,538重标注控制步和10,397新评估控制步已核验。更早的教师混合执行研究两轮监督结果为4/10、1/10，随后102,400实际控制步PPO为4/10；全部检查点与成本保留。

历史源代码`6d351d5`已推送并核对远端，该版本本地及独立检出均为137项通过、5项条件跳过。独立固定依赖环境逐字节重建两轮独立驾驶聚合的核验汇总和图表，两模型重载预测可重复、统计冻结；核验覆盖28,586采集控制步和21,644评估控制步。更早标签对照的复现凭证仍保留。新模型/数据尚未上传发布资产，研究验收仍未完成。

源代码`ff6b578`已推送并核对远端。本地与独立检出的固定依赖环境均为136项通过、5项条件跳过；独立环境加载全部5个恢复诊断模型，归一化保持冻结，并逐字节重建核验汇总及结果图。这些新诊断模型/数据目前只在本地，尚未作为发布资产上传。下文较早测试数量属于其各自源代码版本。

2026-10-23更新：[恢复训练与学习稳定性报告](../reports/recovery_stability_20261023.md)。无界、tanh及softsign均值三个B1分支最终均为6/10，静态及未见布局仍失败。CARLA H恢复新增70,080控制步、B1完成100,270控制步，二者四条训练道路均为0/4；全部176,167训练及评估控制帧已匹配原始CARLA记录。损耗交互单独计入，不作为同预算优势比较。独立checkout另完成1,592个真实CARLA控制步复验，随后确认actor清空并关闭本任务服务器。本地与独立环境均为133项通过、5项条件跳过。完整研究验收仍未通过。

另见[示范辅助学习诊断](../reports/demonstration_diagnostic_20261023.md)：已采集32组成功的B3执行及直接控制重放，但仅行为克隆及后续25,600/102,400控制步PPO的完整任务均为0/10。四种权重与归一化交叉诊断也均为0/10。示范成功不等于策略成功；额外监督与控制频率变化单独披露。

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

早期记录中的133通过、5条件跳过属于历史快照，不代表当前版本；CPU验证范围见[CI说明](cpu_ci.md)。
新下载入口专项测试在本地和独立目录均21项通过。跳过项不计为实测通过。原生地图、执行器和完整物理帧实测见[原生道路报告](../reports/native_road_learning_20261022.md)及[恢复训练报告](../reports/recovery_stability_20261023.md)。

## 开发模型复现

当前原生B1的完整开发证据可用以下两步下载并复现，无需GitHub登录或重新启动CARLA：

```powershell
$capsuleUrl = 'https://github.com/Leo-7GOAT/Intelligent-driving-system/releases/download/v0.2.0-dev.20261022/native_B1_v4_approach_PPO_20261031_repro.zip.parts.json'
$capsuleSha = 'ebc35169fb1efee6841f21b595071de12a20cae85074aef574290c28469b58f7'
.\.venv\Scripts\python.exe scripts/download_native_asset.py --manifest-url $capsuleUrl --sha256 $capsuleSha --output runs/public_B1_approach_download
.\.venv\Scripts\python.exe scripts/reproduce_native_asset.py --archive runs/public_B1_approach_download/native_B1_v4_approach_PPO_20261031_repro.zip --sha256 $capsuleSha --output runs/public_B1_approach_reproduced
```

命令校验全部21个分片与整包哈希，再核对25,745个真实控制记录、12个起始动作并重建结果图。
该包保留6/12成功及全部失败，不代表合格基线或最终确认。下载器限制并发、时间和磁盘占用，
不覆盖已有目录；中断文件保留。每次复现使用新输出路径。此前轻量开发资产的复现方式保留如下。
[实际下载与独立复现记录](../reports/native_public_download_cli_verification_20261031.json)已核验整包哈希、原始控制、起始动作和逐字节相同的结果图。

将开发预发布资产解压到`runs/`，得到`runs/development_v8_seed11/`后执行：

```powershell
.\.venv\Scripts\python.exe scripts/reproduce_development_asset.py --asset runs/development_v8_seed11 --output runs/reproduced --full
```

该命令核对哈希、使用随包冻结源码、加载PPO及归一化状态、重跑全部20个开发回合，并重建端点图。数据包括失败，不是未见最终确认。模型内容和限制见[模型卡](model_card.md)。

CARLA需要独立安装服务端和匹配客户端，相关要求见[桥接说明](CARLA_bridge.md)。不把条件跳过的测试称作CARLA验证。运行中的批次、恢复入口和停止条件见[交接记录](handoff.md)。

# 分层 PPO—Hybrid A*—MPC 硕士项目（2026-10-16 历史说明）

本文保留旧批次操作与证据说明。2026-10-22 已进入新的自主研发流程，旧测试集降为历史评估集，旧六小时及禁止提交的授权限制已被新指令替代。当前状态见 [handoff.md](handoff.md)，完整验收缺口见 [research_acceptance.md](research_acceptance.md)。下文的“当前”“未开启新窗口”等描述均仅指历史批次。

研究目标：检验学习高层决策、车辆可行规划和跟踪控制是否改善非分层 PPO 的样本效率、任务质量、稳定性和泛化。**预注册的完整五种子主结论仍未完成。** 本轮6小时窗口内已实际修改、测试、训练并保存停止：五个B1均达1M；H的11/33/44种子达1M、22/55分别保存到842155/686986步。已完成的三个配对独立测试H均30/40、B1均0/40，B3也为30/40。CARLA两组约50k配对微调后H为9/30、10/30，B1均0/30。全部结果和失败保留，未开启新训练窗口。

本批目录为 `runs/20261016T015226Z_research_batch_3377580f/`。最新实测表见该目录的 `research_results.md`、`failure_analysis.md`、`analysis/`，CARLA细节见 [CARLA_bridge.md](docs/CARLA_bridge.md)。历史640步结果仍保留，不与新实验混合。所有请求的“训练步”均报告实际物理步及rollout超额；CARLA另列控制与初始化步。

旧工程完整保存在 `legacy_snapshot_20261016/`，只作审计，不作为新实验入口。旧 README 的成功率、RMSE 和“零样本”指标存在位置赋值、重复 reset、终点吸附和单位不一致问题，不可用于新论文结论。上一轮整改已建立 `feature/physical-hierarchical-ppo`；本轮沿用该分支，未重新初始化Git，未commit/push。

## 安装（Windows PowerShell，Python 3.11）

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
```

依赖只装入项目 `.venv`。`requirements.txt` 是核心直接依赖，锁文件包含本轮完整版本。无需 CARLA 即可运行轻量闭环；不会下载 CARLA 服务端。

## 单测与完整真实验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q -m "not training and not carla"
.\.venv\Scripts\python.exe -m pytest -q
```

完整测试包含真实 Hybrid A*、真实 OSQP QP、真实车辆积分、静态绕障和 H/B1 的两轮 PPO 优化。Mock 仅用于明确的失败注入。CARLA 条件测试缺资源时会 skip。

一条命令运行 H、B1、B2 的真实训练/重载/验证，B1 hold10 频率对照、B3 和固定参考 MPC 对比：

```powershell
.\.venv\Scripts\python.exe scripts\smoke_suite.py
```

每个训练配置640物理步，至少两轮优化，suite 每训练120 s墙钟上限。任务截止30 s，可完成默认23 m路线。它是工程 smoke，**不是训练收敛或优势验证**。所有输出在新建的 `runs/<UTC时间>_smoke_suite_<随机ID>/`，从不覆盖旧实验。

## 分别运行 H/B1 短训练及评估

```powershell
.\.venv\Scripts\python.exe -m navstack.cli train --method H --profile smoke --seed 7
$hRun = (Get-ChildItem -LiteralPath runs -Directory -Filter '*train_H_smoke*' | Sort-Object Name | Select-Object -Last 1).FullName
.\.venv\Scripts\python.exe -m navstack.cli evaluate --method H --run $hRun --split validation --episodes 4

.\.venv\Scripts\python.exe -m navstack.cli train --method B1 --profile smoke --seed 7
$b1Run = (Get-ChildItem -LiteralPath runs -Directory -Filter '*train_B1_smoke*' | Sort-Object Name | Select-Object -Last 1).FullName
.\.venv\Scripts\python.exe -m navstack.cli evaluate --method B1 --run $b1Run --split validation --episodes 4
```

训练的 `total_timesteps` 已按方法动作频率转换，停止条件使用累计实际物理步数，并报告 rollout 超额。每次保存 model.zip、normalization.pkl、配置、源码hash、原始轨迹、奖励分项、优化日志和确定性重载探针。旧 grid PPO 或其他层级签名不匹配会明确拒绝加载，必须重训。

默认规划器有0.1秒墙钟限额，同seed可能受机器负载影响。可复现实验可显式使用节点预算模式（训练与评估须保持同配置，不能悄悄替换底层）：

```powershell
.\.venv\Scripts\python.exe -m navstack.cli train --method H --config configs/offline_H.json --profile smoke --seed 7
```

## B2/B3、频率与控制失配

```powershell
.\.venv\Scripts\python.exe -m navstack.cli train --method B2 --profile smoke --seed 7
.\.venv\Scripts\python.exe -m navstack.cli baseline-smoke --method B2 --episodes 4
.\.venv\Scripts\python.exe -m navstack.cli baseline-smoke --method B3 --episodes 4
.\.venv\Scripts\python.exe -m navstack.cli train --method B1 --flat-hold 10 --profile smoke --seed 7
.\.venv\Scripts\python.exe -m navstack.cli benchmark --steps 240
```

B2 `train` 是真实 PPO；B2 `baseline-smoke` 只是明确标注 cycle 的固定目标接口检查，不能当成训练策略。B3 明确为规则对照。统一场景与车辆，flat PPO 不制造参考轨迹或 MPC RMSE。

## 当前批次与累计恢复

本轮用户已授权6小时本地批次，已实际启动。统一任务45秒，节点预算模式，独立训练种子11/22/33/44/55。`research_queue.py`在同一次训练的100k、300k、1M里程碑继续保存，不按每档重新初始化。源码快照用于避免后台训练受到后续编辑影响；优化器、归一化、物理状态和随机数保存，恢复时仅重建OSQP工作区。

```powershell
# 查看真实进度，不重复启动正在运行的队列
.\.venv\Scripts\python.exe scripts\batch_status.py --batch runs\20261016T015226Z_research_batch_3377580f
.\.venv\Scripts\python.exe scripts\research_report.py --batch runs\20261016T015226Z_research_batch_3377580f

# 确认相应队列已结束后，在本批剩余预算内恢复；已有检查点会累计接续
.\.venv\Scripts\python.exe scripts\parallel_research.py --batch runs\20261016T015226Z_research_batch_3377580f --queue configs\formal_v5_queue.json --source-root runs\20261016T015226Z_research_batch_3377580f\source_formal_v5 --workers 6
```

相同 seed 的不同候选 checkpoint 用 `select` 在验证集选择，禁止挑选最好的训练 seed：

```powershell
.\.venv\Scripts\python.exe -m navstack.cli select $hRun --episodes 16
```

该旧选择入口使用安全、成功、进度排序，只用于开发检查点。新主实验固定预算检查点，不选择最好训练seed，也不以较早检查点替换共同1M主指标。主次指标与20,000次训练seed配对bootstrap见 `configs/research_preregistration.json`。最终测试清单冻结，开发阶段不调用它。

正式方案为formal_v5：共同短路线课程、终点参考修复、beta=1和target_kl=.03；B1采用经开发验证的gSDE探索。所有验证/测试始终完整路线。v2提高beta的负结果及v3结果保留。详细配置与限制见 [experiment_protocol.md](docs/experiment_protocol.md)。

`final_evaluation_queue.py`已启动，每个1M检查点就绪后即运行固定最终/泛化清单；`carla_formal_pipeline.py`在已就绪的配对1M源模型中按固定种子优先级串行执行真实CARLA冻结迁移、双方累计50k/100k微调和最终测试；`supplementary_pipeline.py`在主队列完成后独立训练B2、hold10和共同安全盾。它们都遵守本批15:52截止。等待状态不等于对应实验已经完成，请读取各status.json。

```powershell
.\.venv\Scripts\python.exe scripts\publication_report.py --batch runs\20261016T015226Z_research_batch_3377580f
```

该命令从实际日志生成主/次指标、分场景结果、五seed区间与学习曲线；未运行项目保持缺失。完整正式统计位于批次 `analysis/formal_results.md`。

该报告入口同时检查冻结源码、清单hash和检查点计数；可单独运行 `scripts/verify_batch_integrity.py --batch ...`。CARLA动作、实际油门/制动、静止与接管审计见 `analysis/carla_decision_failure_audit.md`，实际同条件轨迹见 `analysis/carla_v5_matched50k_pair.png`。最新修复后开发微调为H 50346/B1 50592实际世界步，固定六条件成功1/6与0/6，仍不能代替正式多种子结论。

## 本机CARLA

```powershell
.\scripts\setup_carla_client.ps1
.\.venv\Scripts\python.exe -m navstack.cli carla-probe
# 仅在没有现有服务时启动；脚本会检查进程和端口
.\scripts\start_carla.ps1 -OutputDirectory runs\manual_carla_start
```

真实桥接只允许一个tick进程。不要在CARLA训练或评估运行时启动另一个环境。执行中的车辆位置来自CARLA；规则/固定参考诊断与PPO轨迹分别标注。预算守卫在本批6小时结束前要求保存，服务空闲且身份仍为本批启动时关闭该服务。

## 报告与入口

```powershell
.\.venv\Scripts\python.exe -m navstack.cli report
.\.venv\Scripts\python.exe Simulation.py --help
```

suite 的报告指定 `--root` 为打印出的 suite 目录。报告保留所有失败回合；旧8秒调试记录单独注明，不与30秒任务混算。CSV优化日志在 sb3/，逐轮 KL、clip_fraction、entropy、value_loss、explained_variance 在 optimization.jsonl。

主代码位于 navstack/：env.py 唯一环境；vehicle.py 唯一状态源；planning.py 两个规划器；trajectory.py 弧长/速度/时空检查；mpc.py 单周期QP；reward.py共享奖励；experiments.py训练评估。旧顶层模块成为显式新接口桥接，不再保留瞬移演示入口。

## CARLA 条件测试

```powershell
.\.venv\Scripts\python.exe -m navstack.cli carla-probe
$env:RUN_CARLA = '1'
.\.venv\Scripts\python.exe -m pytest -q -m carla -rs
Remove-Item Env:\RUN_CARLA
```

需要用户已部署的服务端、2000端口、同版本Python客户端、空闲spawn点。配置样例是 configs/carla.json。当前客户端缺失且端口不可达，明确 BLOCKED。适配器采用同步固定步长，不开自车autopilot、不瞬移，只清理自建actor并恢复world设置。条件测试目前只检查真实plant接口，**完整PPO→CARLA、车辆几何/纵向标定及三类场景尚未验证**，不能写成CARLA已完成。

实施说明见 `docs/design.md`；实测结果与未完成项见 `docs/change_report.md`；下一步可执行命令见 `docs/handoff.md`。


## 检查点随机状态修复（12:41，最终测试前）

实测发现旧累计训练在保存后用 PPO.load 验证模型，会重置 Python、NumPy、Torch 全局随机状态，且与保存的恢复状态不一致。现在验证过程用上下文保存并恢复 RNG；真实 B1 连续训练与两段恢复训练的最终参数完全一致，普通回归45通过、3项条件跳过。该测试不能推出重建OSQP工作区的H位级一致。

formal_v4及更早累计训练的观测结果全部保留，但仅作带此实现缺陷的历史开发记录。formal_v5在相同环境/PPO参数、相同5个种子和原测试清单下从头训练；没有续用v4权重。修改时未取得任何最终性能，主指标、统计规则及15:52硬截止不变。根因与修订见批次 checkpoint_rng_audit.json、checkpoint_rng_repair_amendment.json，旧冻结协议见 protocol_history。

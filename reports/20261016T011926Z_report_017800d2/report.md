# 实测运行报告

仅记录工程验证，不据短训练宣称分层优势。所有失败回合保留。

8 秒早期运行仅用于接口调试（路线无法在该截止时间完成），不与 30 秒任务合并统计。
|run_id / 方法|截止秒|物理步|优化阶段|Actor L2变化|重载一致|训练秒|
|---|---:|---:|---:|---:|---|---:|
|20261016T011608Z_train_H_smoke_seed7_a743d665 / H|30.0|640|2|0.048964|True|1.96|
|20261016T011615Z_train_B1_smoke_seed7_1d17f535 / B1|30.0|640|2|0.081081|True|0.77|
|20261016T011619Z_train_B2_smoke_seed7_a24f17f4 / B2|30.0|640|2|0.047859|True|1.20|
|20261016T011622Z_train_B1_smoke_seed7_6a42313c / B1|30.0|640|2|0.038899|True|0.18|

|run_id / 方法/策略|截止秒|划分|回合|成功率|失败原因|
|---|---:|---|---:|---:|---|
|20261016T011611Z_eval_H_learned_validation_seed7_4a283771 / H/learned|30.0|validation|4|0.250|{'collision': 0, 'out_of_bounds': 0, 'deadline': 3}|
|20261016T011616Z_eval_B1_learned_validation_seed7_565bfc12 / B1/learned|30.0|validation|4|0.000|{'collision': 0, 'out_of_bounds': 0, 'deadline': 4}|
|20261016T011620Z_eval_B2_learned_validation_seed7_6e2ce4f0 / B2/learned|30.0|validation|4|0.000|{'collision': 0, 'out_of_bounds': 0, 'deadline': 4}|
|20261016T011623Z_eval_B1_learned_validation_seed7_ae539fe2 / B1/learned|30.0|validation|4|0.000|{'collision': 0, 'out_of_bounds': 0, 'deadline': 4}|
|20261016T011624Z_eval_B3_rule_validation_seed7_d9e8798e / B3/rule|30.0|validation|4|0.750|{'collision': 0, 'out_of_bounds': 0, 'deadline': 1}|
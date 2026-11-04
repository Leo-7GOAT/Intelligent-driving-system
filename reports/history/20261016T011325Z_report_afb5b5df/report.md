# 实测运行报告

仅记录工程验证，不据短训练宣称分层优势。所有失败回合保留。

8 秒早期运行仅用于接口调试（路线无法在该截止时间完成），不与 30 秒任务合并统计。
|run_id / 方法|截止秒|物理步|优化阶段|Actor L2变化|重载一致|训练秒|
|---|---:|---:|---:|---:|---|---:|
|20261016T005729Z_train_H_smoke_seed7_90e101ed / H|8.0|640|2|0.038645|True|4.86|
|20261016T005929Z_train_B1_smoke_seed7_8caeb4cc / B1|8.0|640|2|0.125575|True|0.69|

|run_id / 方法/策略|截止秒|划分|回合|成功率|失败原因|
|---|---:|---|---:|---:|---|
|20261016T010110Z_eval_H_learned_validation_seed7_53a53141 / H/learned|8.0|validation|4|0.000|{'collision': 0, 'out_of_bounds': 0, 'deadline': 4}|
|20261016T010135Z_eval_B3_rule_validation_seed7_f7bd8b7e / B3/rule|30.0|validation|1|0.000|{'collision': 0, 'out_of_bounds': 0, 'deadline': 1}|
|20261016T010145Z_eval_B1_learned_validation_seed7_bc3b4b48 / B1/learned|8.0|validation|4|0.000|{'collision': 0, 'out_of_bounds': 0, 'deadline': 4}|
|20261016T010150Z_eval_B2_cycle_validation_seed7_da8066dd / B2/cycle|8.0|validation|4|0.000|{'collision': 0, 'out_of_bounds': 0, 'deadline': 4}|
|20261016T010238Z_eval_B3_rule_validation_seed7_70d100ba / B3/rule|30.0|validation|1|1.000|{'collision': 0, 'out_of_bounds': 0, 'deadline': 0}|
|20261016T010447Z_eval_B3_rule_validation_seed7_be141820 / B3/rule|30.0|validation|4|0.750|{'collision': 0, 'out_of_bounds': 0, 'deadline': 1}|
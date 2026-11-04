# 实测运行报告

仅记录工程验证，不据短训练宣称分层优势。所有失败回合保留。

8 秒早期运行仅用于接口调试（路线无法在该截止时间完成），不与 30 秒任务合并统计。
|run_id / 方法|截止秒|物理步|优化阶段|Actor L2变化|重载一致|训练秒|
|---|---:|---:|---:|---:|---|---:|
|20261016T011001Z_train_H_smoke_seed7_0b98e705 / H|30.0|640|2|0.048964|True|1.73|
|20261016T011007Z_train_B1_smoke_seed7_be8d8c3e / B1|30.0|640|2|0.081081|True|0.66|
|20261016T011011Z_train_B2_smoke_seed7_0de972ce / B2|30.0|640|2|0.047859|True|0.93|
|20261016T011013Z_train_B1_smoke_seed7_253de4a2 / B1|30.0|640|2|0.038899|True|0.16|

|run_id / 方法/策略|截止秒|划分|回合|成功率|失败原因|
|---|---:|---|---:|---:|---|
|20261016T011004Z_eval_H_learned_validation_seed7_aad83f2a / H/learned|30.0|validation|4|0.000|{'collision': 0, 'out_of_bounds': 0, 'deadline': 4}|
|20261016T011008Z_eval_B1_learned_validation_seed7_6c032b85 / B1/learned|30.0|validation|4|0.000|{'collision': 0, 'out_of_bounds': 0, 'deadline': 4}|
|20261016T011012Z_eval_B2_learned_validation_seed7_d3b5aa37 / B2/learned|30.0|validation|4|0.000|{'collision': 0, 'out_of_bounds': 0, 'deadline': 4}|
|20261016T011014Z_eval_B1_learned_validation_seed7_8ba69c0a / B1/learned|30.0|validation|4|0.000|{'collision': 0, 'out_of_bounds': 0, 'deadline': 4}|
|20261016T011015Z_eval_B3_rule_validation_seed7_0f649db1 / B3/rule|30.0|validation|4|0.750|{'collision': 0, 'out_of_bounds': 0, 'deadline': 1}|
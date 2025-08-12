# dialog_parser.py

def parse_command(user_input):
    """
    解析用户输入的命令，输出结构化任务指令。
    支持关键词自动匹配（中英文混输）。
    """
    cmd = user_input.strip().lower()

    # 训练
    if "train" in cmd or "训练" in cmd:
        return {"action": "train"}

    # 推理/测试/运行
    if "infer" in cmd or "run" in cmd or "test" in cmd or "推理" in cmd or "执行" in cmd or "轨迹" in cmd:
        return {"action": "infer"}

    # 可视化/画图
    if "visual" in cmd or "show" in cmd or "plot" in cmd or "display" in cmd \
       or "可视化" in cmd or "画图" in cmd or "显示" in cmd:
        return {"action": "visual"}

    # 退出
    if cmd in ["exit", "quit", "q", "bye", "退出", "结束"]:
        return {"action": "exit"}

    # 其他未识别命令
    return {"action": "unknown"}

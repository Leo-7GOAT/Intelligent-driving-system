import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_grid_path(grid_size, path, obstacles, start, goal, title="Path"):
    """
    原有网格与路径可视化函数，绘制方格背景、障碍、起点/终点及轨迹
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks(np.arange(0, grid_size))
    ax.set_yticks(np.arange(0, grid_size))
    ax.grid(True)

    # 绘制障碍
    if obstacles:
        obs = np.array(list(obstacles))
        ax.scatter(obs[:, 0], obs[:, 1], marker='s', s=200, c='black', label='Obstacle')

    # 绘制起点和终点
    ax.scatter(start[0], start[1], marker='o', s=200, c='green', label='Start')
    ax.scatter(goal[0], goal[1], marker='*', s=200, c='red', label='Goal')

    # 绘制路径
    pts = np.array(path)
    ax.plot(pts[:, 0], pts[:, 1], '-o', label='Path')

    ax.set_title(title)
    ax.legend()
    plt.gca().invert_yaxis()
    plt.show()


def plot_path_compare(a_star_path, smooth_xy, mpc_path, goal, title="Compare"):
    """
    绘制 A* 路径、平滑轨迹和 MPC 跟踪轨迹对比。
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    # A*
    ast = np.array(a_star_path)
    ax.plot(ast[:, 0], ast[:, 1], '-s', label='A* Path')
    # 平滑轨迹
    sm = np.array(smooth_xy)
    ax.plot(sm[:, 0], sm[:, 1], '--', label='Smoothed')
    # MPC
    mpc = np.array(mpc_path)
    ax.plot(mpc[:, 0], mpc[:, 1], '-.', label='MPC Track')

    ax.scatter(goal[0], goal[1], marker='*', s=200, c='red')
    ax.set_title(title)
    ax.legend()
    plt.gca().invert_yaxis()
    plt.show()


def plot_metrics(metrics_file="metrics.csv", figsize=(12, 6)):
    """
    从 metrics.csv（带表头）读取，绘制主要指标趋势和柱状对比。
    """
    import pandas as pd
    if not os.path.exists(metrics_file):
        print(f"[可视化] 未找到指标文件：{metrics_file}")
        return

    df = pd.read_csv(metrics_file)
    # 只保留float列
    sr = df['sr'].astype(float).values
    oratio = df['oratio'].astype(float).values
    rmse = df['rmse'].astype(float).values
    idx = np.arange(1, len(sr) + 1)

    # 折线图
    plt.figure(figsize=figsize)
    ax1 = plt.gca()
    ax1.plot(idx, sr, marker='o', label='Success Rate')
    ax1.plot(idx, oratio, marker='s', label='Optimality Ratio')
    ax1.set_xlabel('Experiment Index')
    ax1.set_ylabel('Rate / Ratio')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(idx, rmse, color='C3', marker='^', label='MPC RMSE')
    ax2.set_ylabel('MPC RMSE (m)')
    ax2.legend(loc='upper right')

    plt.title('Key Metrics Over Experiments')
    plt.tight_layout()
    plt.show()

    # 柱状图
    width = 0.25
    plt.figure(figsize=figsize)
    plt.bar(idx - width, sr, width, label='Success Rate')
    plt.bar(idx, oratio, width, label='Optimality Ratio')
    plt.bar(idx + width, rmse, width, label='MPC RMSE')
    plt.xlabel('Experiment Index')
    plt.legend()
    plt.title('Key Metrics Comparison')
    plt.tight_layout()
    plt.show()
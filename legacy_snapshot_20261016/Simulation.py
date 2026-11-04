# Simulation.py

import sys
import os
import ast
import math
import time
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root modules load first
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from env_gridworld import GridWorldGymEnv
from ppo_agent import (
    train_agent,
    load_agent,
    decide_local_goal,
    plot_training_curves,
    evaluate_success
)
from astar_agent import plan_with_astar
from mpc_agent import mpc_track_path, plot_path_compare
from visualization import plot_grid_path
from dialog_parser import parse_command

def infer_custom(env, model):
    """
    Run one rollout in env with model, return a dict of metrics:
      'success', 'optimality', 'astar_time', 'mpc_time', 'rmse'
    """
    # reset environment and record start position
    obs, _ = env.reset()
    agent_pos = tuple(env.agent_pos)
    start_pos = agent_pos  # record initial position for optimality calculation
    goal = tuple(env.goal_pos)

    plan_times = []
    ctrl_times = []
    mse_acc = 0.0
    pt_cnt = 0
    full_moves = 0
    success = False

    while full_moves < env.max_steps:
        state = np.array(obs, dtype=np.float32)
        action_arr, _ = model.predict(state, deterministic=True)
        action = int(action_arr)
        sub_goal = decide_local_goal(model, state, grid_size=env.grid_size)

        # A* planning
        t0 = time.perf_counter()
        path = plan_with_astar(agent_pos, sub_goal, set(env.obstacles), env.grid_size)
        dt_plan = time.perf_counter() - t0
        plan_times.append(dt_plan)
        if len(path) < 2:
            break

        # MPC tracking
        t1 = time.perf_counter()
        mpc_path = mpc_track_path(path)
        dt_ctrl = time.perf_counter() - t1
        ctrl_times.append(dt_ctrl)

        # accumulate full_moves
        full_moves += len(path) - 1

        # accumulate RMSE
        ref = np.array(path)
        for x, y in mpc_path:
            d2 = np.min((ref[:,0] - x)**2 + (ref[:,1] - y)**2)
            mse_acc += d2
            pt_cnt += 1

        # advance for next step
        agent_pos = path[-1]
        env.agent_pos = list(agent_pos)
        obs, info = env.reset()
        if agent_pos == goal:
            success = True
            break

    # compute metrics
    total_moves = full_moves
    # compute shortest path length from start to goal (Chebyshev distance)
    dx = abs(goal[0] - start_pos[0])
    dy = abs(goal[1] - start_pos[1])
    shortest = max(dx, dy)
    optimality = (shortest / total_moves) if total_moves > 0 else 0.0
    astar_time = np.mean(plan_times) if plan_times else 0.0
    mpc_time = np.mean(ctrl_times) if ctrl_times else 0.0
    rmse = math.sqrt(mse_acc / pt_cnt) if pt_cnt > 0 else 0.0

    return {
        'success': success * 1.0,
        'optimality': optimality,
        'astar_time': astar_time,
        'mpc_time': mpc_time,
        'rmse': rmse
    }

def plot_summary(summary, title):
    """
    Plot bar chart for a single scenario summary.
    summary: dict of metric->(mean,std)
    """
    metrics = ['success','optimality','rmse','astar_time','mpc_time']
    labels = ['Success Rate','Optimality Ratio','MPC RMSE','A* Time (s)','MPC Time (s)']
    means = [summary[m][0] for m in metrics]
    stds  = [summary[m][1] for m in metrics]
    x = np.arange(len(metrics))
    plt.figure(figsize=(10,4))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, labels)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def demo_infer():
    """
    Single-step demo: PPO -> A* planning in default 4x4 empty env
    """
    model = load_agent()
    env = GridWorldGymEnv()
    obs, _ = env.reset()
    agent_pos = tuple(env.agent_pos)
    goal = tuple(env.goal_pos)
    print(f"[DEMO_INFER] Agent position: {agent_pos}, Goal: {goal}")
    state = np.array(obs, dtype=np.float32)
    action_arr, _ = model.predict(state, deterministic=True)
    action = int(action_arr)
    sub_goal = decide_local_goal(model, state, grid_size=env.grid_size)
    print(f"[DEMO_INFER] PPO sub-goal: {sub_goal}")
    path = plan_with_astar(agent_pos, sub_goal, set(env.obstacles), env.grid_size)
    print(f"[DEMO_INFER] A* path: {path}")
    plot_grid_path(
        grid_size=env.grid_size,
        path=path,
        obstacles=set(env.obstacles),
        start=agent_pos,
        goal=sub_goal,
        title="PPO → A* Subgoal Planning"
    )



def demo_astar():
    """
    Manual A* demo: enter start and goal
    """
    s_start = input("Enter A* start (x,y): ")
    s_goal = input("Enter A* goal  (x,y): ")
    start = tuple(map(int, s_start.strip("()").split(",")))
    goal = tuple(map(int, s_goal.strip("()").split(",")))
    env = GridWorldGymEnv()
    path = plan_with_astar(start, goal, set(env.obstacles), env.grid_size)
    print(f"[DEMO_ASTAR] Path from {start} to {goal}: {path}")
    plot_grid_path(
        grid_size=env.grid_size,
        path=path,
        obstacles=set(env.obstacles),
        start=start,
        goal=goal,
        title="A* Path Planning"
    )


def demo_mpc():
    """
    MPC demo: smooth & track given discrete path
    """
    raw = input("Enter reference path [(x1,y1),(x2,y2),...]: ")
    try:
        ref_path = ast.literal_eval(raw)
    except Exception as e:
        print(f"[DEMO_MPC] Invalid input: {e}")
        return
    print(f"[DEMO_MPC] Reference path: {ref_path}")
    mpc_path = mpc_track_path(ref_path)
    plot_path_compare(
        a_star_path=ref_path,
        smooth_xy=ref_path.copy(),
        mpc_path=mpc_path,
        goal=ref_path[-1],
        title="A* vs Smoothed vs MPC"
    )


def demo_hierarchical():
    """
    End-to-end hierarchical demo: PPO -> A* -> MPC
    Shows step-by-step operations at each layer, then metrics and plots.
    """
    model = load_agent()
    env = GridWorldGymEnv()
    obs, _ = env.reset()
    agent_pos = tuple(env.agent_pos)
    goal = tuple(env.goal_pos)
    print(f"[DEMO_HIER] Start: {agent_pos}, Goal: {goal}")

    full_traj = [agent_pos]
    mpc_full = []
    plan_times, ctrl_times = [], []
    mse_accum = 0.0
    point_cnt = 0
    success = False

    for step in range(env.max_steps):
        state = np.array(obs, dtype=np.float32)
        action_arr, _ = model.predict(state, deterministic=True)
        action = int(action_arr)
        sub_goal = decide_local_goal(model, state, grid_size=env.grid_size)
        print(f"[DEMO_HIER] Step {step}: PPO action={action}, sub-goal={sub_goal}")

        # A* planning
        t0 = time.perf_counter()
        path = plan_with_astar(agent_pos, sub_goal, set(env.obstacles), env.grid_size)
        dt_plan = time.perf_counter() - t0
        plan_times.append(dt_plan)
        print(f"  A* path: {path}, plan time={dt_plan*1000:.2f}ms")
        if len(path) < 2:
            print("  A* failed, stopping.")
            break

        # MPC tracking
        t1 = time.perf_counter()
        mpc_path = mpc_track_path(path)
        dt_ctrl = time.perf_counter() - t1
        ctrl_times.append(dt_ctrl)
        print(f"  MPC points: {len(mpc_path)}, control time={dt_ctrl*1000:.2f}ms")

        # accumulate error
        ref = np.array(path)
        for px, py in mpc_path:
            d2 = np.min((ref[:,0]-px)**2 + (ref[:,1]-py)**2)
            mse_accum += d2
            point_cnt += 1

        # advance agent
        full_traj += path[1:]
        mpc_full += mpc_path
        agent_pos = path[-1]
        env.agent_pos = list(agent_pos)
        obs, info = env.reset()

        if agent_pos == goal:
            print("[DEMO_HIER] Goal reached!")
            success = True
            break

    # metrics calculation and display
    total_moves = len(full_traj) - 1
    dx = abs(full_traj[0][0] - goal[0])
    dy = abs(full_traj[0][1] - goal[1])
    shortest_steps = max(dx, dy)
    optimality = (shortest_steps / total_moves) if total_moves > 0 else 0.0
    rmse = math.sqrt(mse_accum/point_cnt) if point_cnt>0 else float('nan')
    avg_plan = np.mean(plan_times)*1000 if plan_times else 0.0
    avg_ctrl = np.mean(ctrl_times)*1000 if ctrl_times else 0.0
    sr = 1.0 if success else 0.0

    print(f"[METRICS] Success Rate      = {sr:.2%}")
    print(f"[METRICS] Optimality Ratio  = {optimality:.3f}")
    print(f"[METRICS] MPC RMSE         = {rmse:.3f}")
    print(f"[METRICS] Total Moves      = {total_moves}")
    print(f"[METRICS] Avg Plan Time    = {avg_plan:.2f} ms")
    print(f"[METRICS] Avg Control Time = {avg_ctrl:.2f} ms")

    # plot discrete trajectory
    plot_grid_path(
        grid_size=env.grid_size,
        path=full_traj,
        obstacles=set(env.obstacles),
        start=full_traj[0],
        goal=goal,
        title="RL + A* Combined Trajectory"
    )
    # plot MPC tracking trajectory
    plot_grid_path(
        grid_size=env.grid_size,
        path=mpc_full,
        obstacles=set(env.obstacles),
        start=full_traj[0],
        goal=goal,
        title="MPC Tracking Trajectory"
    )


def batch_test(env_params, trials=50):
    """
    Run infer_custom trials times with same env_params, collect metrics,
    return summary dict of metric->(mean, std).
    """
    model = load_agent()
    # 准备收集列表
    metrics = {
        'success':    [],
        'optimality': [],
        'astar_time': [],
        'mpc_time':   [],
        'rmse':       []
    }

    for _ in range(trials):
        env = GridWorldGymEnv(**env_params)
        # 这里 infer_custom 需要返回一个 dict，包含上述五个指标
        res = infer_custom(env, model)  # assume returns that dict
        for k in metrics:
            metrics[k].append(res[k])

    # 计算均值和标准差
    summary = {}
    for k, vals in metrics.items():
        arr = np.array(vals, dtype=float)
        summary[k] = (arr.mean(), arr.std())

    return summary


def compare_summaries(summaries, labels, title):
    """
    Plot grouped bar chart comparing metrics across scenarios.
    summaries: list of dict(metric->(mean,std))
    labels: list of scenario names
    """
    metrics = ['success','optimality','rmse','astar_time','mpc_time']
    metric_labels = ['Success Rate','Optimality Ratio','MPC RMSE','A* Time (s)','MPC Time (s)']
    n = len(summaries)
    x = np.arange(len(metrics))
    width = 0.8 / n
    plt.figure(figsize=(10,5))
    for i, summ in enumerate(summaries):
        means = [summ[m][0] for m in metrics]
        stds  = [summ[m][1] for m in metrics]
        plt.bar(x + i*width, means, width, yerr=stds, capsize=3, label=labels[i])
    plt.xticks(x + width*(n-1)/2, metric_labels)
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    print("Commands: train | infer | astar | mpc | hierarchical | larger | obstacle | compare | exit")
    while True:
        cmd = input(">>> ").strip().lower()
        if cmd == 'train':
            train_agent()
            print("[TRAIN] complete.")
            plot_training_curves(
                reward_log_path="rewards.npy",
                success_log_path="successes.npy",
                smooth_window=10
            )
            sr = evaluate_success(
                load_agent(), n_eval=100,
                grid_size=4, obstacle_num=0, max_steps=30
            )
            print(f"[TRAIN] Eval default success rate: {sr:.2%}")
        elif cmd == 'infer':
            demo_infer()
        elif cmd == 'astar':
            demo_astar()
        elif cmd == 'mpc':
            demo_mpc()
        elif cmd == 'hierarchical':
            demo_hierarchical()
        elif cmd in ('larger','larger map'):
            print("[BATCH] Larger map 8x8 empty")
            summary_large = batch_test({'grid_size':8,'obstacle_num':0,'max_steps':100}, trials=50)
            print(summary_large)
            plot_summary(summary_large, "Zero-shot on 8×8 empty map")
        elif cmd in ('obstacle','obstacle situation'):
            print("[BATCH] 4x4 with 2 obstacles")
            summary_obs = batch_test({'grid_size':4,'obstacle_num':2,'max_steps':50}, trials=50)
            print(summary_obs)
            plot_summary(summary_obs, "Zero-shot on 4×4 with 2 obstacles")
        elif cmd == 'compare':
            # plot train curves
            plot_training_curves(
                reward_log_path="rewards.npy",
                success_log_path="successes.npy",
                smooth_window=10
            )
            # batch tests
            summary_def   = batch_test({'grid_size':4,'obstacle_num':0,'max_steps':30}, trials=50)
            summary_large = batch_test({'grid_size':8,'obstacle_num':0,'max_steps':100}, trials=50)
            summary_obs   = batch_test({'grid_size':4,'obstacle_num':2,'max_steps':50}, trials=50)
            # compare
            compare_summaries(
                summaries=[summary_def, summary_large, summary_obs],
                labels=["Default 4×4","Larger 8×8","Obstacle 4×4"],
                title="Scenario Comparison"
            )
        elif cmd == 'exit':
            break
        else:
            print("Unknown command.")

if __name__ == '__main__':
    main()
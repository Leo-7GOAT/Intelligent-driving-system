# ppo_agent.py

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from env_gridworld import GridWorldGymEnv

# === 回调：收集每集 reward ===
class RewardHistoryCallback(BaseCallback):
    def __init__(self, reward_log_path="rewards.npy", verbose=0):
        super().__init__(verbose)
        self.reward_log_path = reward_log_path
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True

    def _on_training_end(self) -> None:
        np.save(self.reward_log_path, np.array(self.episode_rewards))
        print(f"[PPO_AGENT] rewards saved to {self.reward_log_path}")

# === 回调：收集每集 success (0/1) ===
class SuccessHistoryCallback(BaseCallback):
    def __init__(self, success_log_path="successes.npy", verbose=0):
        super().__init__(verbose)
        self.success_log_path = success_log_path
        self.episode_success = []
        self._current_success = 0  # 本集是否已到达过 goal

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("success", False):
                self._current_success = 1
            if "episode" in info:
                # episode 结束，记录一次
                self.episode_success.append(self._current_success)
                self._current_success = 0
        return True

    def _on_training_end(self) -> None:
        np.save(self.success_log_path, np.array(self.episode_success))
        print(f"[PPO_AGENT] successes saved to {self.success_log_path}")

# === 环境工厂 ===
def make_env(seed=None, grid_size=4, obstacle_num=0, max_steps=30):
    def _init():
        env = GridWorldGymEnv(
            grid_size=grid_size,
            obstacle_num=obstacle_num,
            max_steps=max_steps,
            seed=seed
        )
        return Monitor(env)
    return _init

# === 训练函数 ===
def train_agent(
    total_timesteps=1e5,
    n_envs=4,
    grid_size=4,
    obstacle_num=0,
    max_steps=30,
    save_path="ppo_gridworld.zip",
    reward_log_path="rewards.npy",
    success_log_path="successes.npy"
):
    # 创建并行环境
    env = SubprocVecEnv([
        make_env(seed=i, grid_size=grid_size, obstacle_num=obstacle_num, max_steps=max_steps)
        for i in range(n_envs)
    ])
    model = PPO(
        "MlpPolicy", env,
        learning_rate=5e-5, n_steps=512, batch_size=128, n_epochs=10,
        gamma=0.99, clip_range=0.05, ent_coef=0.01,
        normalize_advantage=True, verbose=1,
        tensorboard_log="./ppo_tb/",
        seed=0
    )
    # 挂载回调
    reward_cb = RewardHistoryCallback(reward_log_path=reward_log_path)
    success_cb = SuccessHistoryCallback(success_log_path=success_log_path)

    print("[PPO_AGENT] Start training...")
    model.learn(
        total_timesteps=int(total_timesteps),
        callback=[reward_cb, success_cb]
    )
    model.save(save_path)
    print(f"[PPO_AGENT] Model saved to {save_path}")
    return model

# === 加载模型 ===
def load_agent(model_path="ppo_gridworld.zip"):
    print(f"[PPO_AGENT] Loading model from {model_path}")
    return PPO.load(model_path)

# === 评估成功率 ===
def evaluate_success(model, n_eval=100, grid_size=4, obstacle_num=0, max_steps=30):
    """
    严格评估：在独立环境里运行 n_eval 次，统计 info['success'] 标志。
    返回成功率（0.0–1.0）。
    """
    env = GridWorldGymEnv(
        grid_size=grid_size,
        obstacle_num=obstacle_num,
        max_steps=max_steps
    )
    success_count = 0

    for _ in range(n_eval):
        obs, _ = env.reset()
        done = False
        while not done:
            action_arr, _ = model.predict(obs, deterministic=True)
            action = int(action_arr)        # 强制转换为 int
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if info.get("success", False):
            success_count += 1

    success_rate = success_count / n_eval
    print(f"[METRIC] Success Rate = {success_rate:.2%}  ({success_count}/{n_eval})")
    return success_rate

# === 决策子目标 (8 邻域) ===
import numpy as np

def decide_local_goal(model, state, grid_size=4):
    """
    高层 RL：决策下一个子目标点（任意网格点）。
    state: np.array([x, y, gx, gy])
    返回 (tx, ty)，其中 tx,ty ∈ [0, grid_size-1]
    """
    # 模型预测会返回一个形如 [a] 的 array
    action_arr, _ = model.predict(state, deterministic=True)
    a = int(action_arr)         # 0 .. grid_size*grid_size - 1

    # divmod 映射为二维坐标
    tx, ty = divmod(a, grid_size)

    # 保证在边界内（虽然 divmod 本身已经保证）
    tx = int(np.clip(tx, 0, grid_size - 1))
    ty = int(np.clip(ty, 0, grid_size - 1))
    return (tx, ty)


# === 画训练曲线 ===
def plot_training_curves(
    reward_log_path="rewards.npy",
    success_log_path="successes.npy",
    smooth_window=10
):
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d

    rewards = np.load(reward_log_path)
    successes = np.load(success_log_path)
    episodes = np.arange(1, len(rewards) + 1)

    # 奖励
    plt.figure(figsize=(8,4))
    plt.plot(episodes, rewards, alpha=0.3, label="Reward per Episode")
    if len(rewards) > smooth_window:
        sm = uniform_filter1d(rewards, size=smooth_window)
        plt.plot(episodes, sm, label=f"Reward MA({smooth_window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO Training Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 成功率
    plt.figure(figsize=(8,4))
    plt.plot(episodes, successes, alpha=0.3, label="Episode Success (0/1)")
    if len(successes) > smooth_window:
        ss = uniform_filter1d(successes, size=smooth_window)
        plt.plot(episodes, ss, label=f"Success Rate MA({smooth_window})")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.title("PPO Training Success Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === 直接运行时演示训练与画图 ===
if __name__ == "__main__":
    model = train_agent()
    plot_training_curves()

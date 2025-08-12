# env_gridworld.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GridWorldGymEnv(gym.Env):
    def __init__(self, grid_size=4, obstacle_num=0, max_steps=30, seed=None):
        super().__init__()
        self.grid_size = grid_size
        self.obstacle_num = obstacle_num
        self.max_steps = max_steps
        # obstacles list for compatibility
        self.obstacles = []

        # RNG seed storage
        self._seed = None
        if seed is not None:
            self.seed(seed)

        # action space: any grid cell as sub-goal
        self.action_space = spaces.Discrete(grid_size * grid_size)

        # state space: agent(x,y) + goal(x,y)
        low = np.array([0, 0, 0, 0], dtype=np.int32)
        high = np.array([grid_size-1, grid_size-1, grid_size-1, grid_size-1], dtype=np.int32)
        self.observation_space = spaces.Box(low, high, dtype=np.int32)

    def seed(self, seed=None):
        """
        Set random seed for reproducibility.
        """
        self._seed = seed
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        """
        Reset environment. Supports Gymnasium-style seed parameter.
        Returns (obs, info).
        """
        if seed is not None:
            self.seed(seed)
        # fixed start and goal positions
        self.agent_pos = (0, 0)
        self.goal_pos = (self.grid_size - 1, self.grid_size - 1)
        self.steps = 0
        obs = np.array([*self.agent_pos, *self.goal_pos], dtype=np.int32)
        info = {}
        return obs, info

    def step(self, action):
        # map action to target cell
        tx, ty = divmod(int(action), self.grid_size)
        # set agent_pos to sub-goal (for A* planning externally)
        self.agent_pos = (tx, ty)
        self.steps += 1

        terminated = False
        truncated = False
        success = False
        if self.agent_pos == self.goal_pos:
            reward = 100
            terminated = True
            success = True
        else:
            reward = -1
            if self.steps >= self.max_steps:
                truncated = True

        obs = np.array([*self.agent_pos, *self.goal_pos], dtype=np.int32)
        return obs, reward, terminated, truncated, {"success": success}

    def render(self, mode='human'):
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        grid[ax, ay] = 'A'
        grid[gx, gy] = 'G'
        for (ox, oy) in self.obstacles:
            grid[ox, oy] = 'X'
        print("\n".join("".join(row) for row in grid))

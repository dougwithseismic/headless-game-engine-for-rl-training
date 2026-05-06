import random

import gymnasium as gym
import numpy as np
import ghostlobby


class GhostLobbyGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config_path, scenario="fps", agent_id=0, frame_skip=4, max_steps=2048):
        super().__init__()
        self.config_path = config_path
        self.scenario = scenario
        self.agent_id = agent_id
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_ticks = 0

        self.env = ghostlobby.GhostLobbyEnv(config_path, scenario=scenario)
        space_info = self.env.action_space()
        self.action_size = space_info["total_size"]

        obs, _ = self.env.reset()
        sample_obs = self._flatten_obs(obs[self.agent_id])
        self.obs_size = len(sample_obs)

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_size,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32
        )

    def _flatten_obs(self, agent_obs):
        parts = []
        for key in sorted(agent_obs.keys()):
            parts.extend(agent_obs[key])
        return np.array(parts, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_ticks = 0
        self.agent_id = random.randint(0, 1)
        self.env = ghostlobby.GhostLobbyEnv(self.config_path, scenario=self.scenario)
        obs, info = self.env.reset()
        return self._flatten_obs(obs[self.agent_id]), {}

    def step(self, action):
        action_list = action.tolist() if hasattr(action, "tolist") else list(action)
        total_reward = 0.0
        terminated = False
        truncated = False

        for _ in range(self.frame_skip):
            obs, rewards, term, trunc, infos = self.env.step({self.agent_id: action_list})
            total_reward += rewards.get(self.agent_id, 0.0)
            self.episode_ticks += 1
            terminated = term.get(self.agent_id, False)
            if terminated:
                break

        self.current_step += 1
        self.episode_reward += total_reward
        if self.current_step >= self.max_steps:
            truncated = True

        flat_obs = self._flatten_obs(obs[self.agent_id])

        info = {}
        if terminated or truncated:
            info["episode_reward"] = self.episode_reward
            info["episode_ticks"] = self.episode_ticks
            info["episode_steps"] = self.current_step

        return flat_obs, total_reward, terminated, truncated, info

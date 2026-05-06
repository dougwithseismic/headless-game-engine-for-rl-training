"""
Self-play Gymnasium wrapper for GhostLobby.

Agent 0 is controlled by the learning policy (SB3).
Agent 1 is controlled by a frozen opponent policy that gets swapped periodically.
Falls back to scripted AI actions when no opponent policy is set.
"""

import random

import gymnasium as gym
import numpy as np
import ghostlobby


class SelfPlayGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config_path, scenario="fps", frame_skip=4, max_steps=2048):
        super().__init__()
        self.config_path = config_path
        self.scenario = scenario
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_ticks = 0
        self.agent_id = 0
        self.opp_id = 1
        self.opponent_policy = None
        self.opponent_obs = None
        self.opponent_lstm_states = None
        self.telemetry_sink = None

        self.env = ghostlobby.GhostLobbyEnv(config_path, scenario=scenario)
        space_info = self.env.action_space()
        self.action_size = space_info["total_size"]

        obs, _ = self.env.reset()
        sample_obs = self._flatten_obs(obs[0])
        self.obs_size = len(sample_obs)

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_size,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32
        )

    def set_opponent(self, policy):
        self.opponent_policy = policy
        self.opponent_lstm_states = None

    def _flatten_obs(self, agent_obs):
        parts = []
        for key in sorted(agent_obs.keys()):
            parts.extend(agent_obs[key])
        return np.array(parts, dtype=np.float32)

    def _get_opponent_action(self, obs_dict):
        if self.opponent_policy is None or self.opp_id not in obs_dict:
            return None
        opp_obs = self._flatten_obs(obs_dict[self.opp_id])
        episode_start = np.array([self.opponent_lstm_states is None])
        action, self.opponent_lstm_states = self.opponent_policy.predict(
            opp_obs, state=self.opponent_lstm_states,
            episode_start=episode_start, deterministic=False,
        )
        return action.tolist() if hasattr(action, "tolist") else list(action)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_ticks = 0
        self.agent_id = random.randint(0, 1)
        self.opp_id = 1 - self.agent_id
        self.env = ghostlobby.GhostLobbyEnv(self.config_path, scenario=self.scenario)
        obs, info = self.env.reset()
        self.opponent_obs = obs
        self.opponent_lstm_states = None
        return self._flatten_obs(obs[self.agent_id]), {}

    def step(self, action):
        action_list = action.tolist() if hasattr(action, "tolist") else list(action)
        total_reward = 0.0
        terminated = False
        truncated = False

        opp_action = self._get_opponent_action(self.opponent_obs)

        for _ in range(self.frame_skip):
            actions = {self.agent_id: action_list}
            if opp_action is not None:
                actions[self.opp_id] = opp_action
            obs, rewards, term, trunc, infos = self.env.step(actions)
            telemetry = self.env.drain_telemetry()
            if self.telemetry_sink is not None:
                for event_json in telemetry:
                    self.telemetry_sink.append(event_json)
            total_reward += rewards.get(self.agent_id, 0.0)
            self.episode_ticks += 1
            terminated = term.get(self.agent_id, False)
            if terminated:
                break

        self.opponent_obs = obs
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

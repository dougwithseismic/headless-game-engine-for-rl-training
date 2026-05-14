"""Fixed-capacity replay buffer for storing environment transitions.

Stores (obs, action, reward, next_obs, done) tuples in pre-allocated
numpy arrays with FIFO eviction when capacity is reached.

Used by the Dyna-style world model augmentation to accumulate real
transitions for training the dynamics ensemble.

Typical usage::

    buf = ReplayBuffer(capacity=500_000, obs_dim=250, act_dim=4)
    buf.add(obs, action, reward, next_obs, done)
    batch = buf.sample(256)
    buf.save("replay.npz")
"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Fixed-capacity FIFO replay buffer backed by numpy arrays.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions stored.
    obs_dim : int
        Observation vector dimensionality.
    act_dim : int
        Action vector dimensionality (number of heads for MultiDiscrete).
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self._pos = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def full(self) -> bool:
        return self._size >= self.capacity

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float | np.ndarray,
        next_obs: np.ndarray,
        done: bool | float | np.ndarray,
    ) -> None:
        """Add a single transition or a batch of transitions.

        When inputs are 1-D (single transition) they are stored directly.
        When inputs are 2-D (batch), each row is stored sequentially.
        """
        obs = np.asarray(obs, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        reward = np.asarray(reward, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        done = np.asarray(done, dtype=np.float32)

        if obs.ndim == 1:
            self.observations[self._pos] = obs
            self.actions[self._pos] = action
            self.rewards[self._pos] = reward
            self.next_observations[self._pos] = next_obs
            self.dones[self._pos] = done
            self._pos = (self._pos + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)
        elif obs.ndim == 2:
            n = obs.shape[0]
            for i in range(n):
                self.observations[self._pos] = obs[i]
                self.actions[self._pos] = action[i]
                self.rewards[self._pos] = reward[i] if reward.ndim > 0 else reward
                self.next_observations[self._pos] = next_obs[i]
                self.dones[self._pos] = done[i] if done.ndim > 0 else done
                self._pos = (self._pos + 1) % self.capacity
                self._size = min(self._size + 1, self.capacity)
        else:
            raise ValueError(f"Expected 1-D or 2-D obs, got {obs.ndim}-D")

    def add_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Add a batch of transitions efficiently."""
        n = obs.shape[0]
        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32).ravel()
        next_obs = np.asarray(next_obs, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32).ravel()

        if self._pos + n <= self.capacity:
            self.observations[self._pos : self._pos + n] = obs
            self.actions[self._pos : self._pos + n] = actions
            self.rewards[self._pos : self._pos + n] = rewards
            self.next_observations[self._pos : self._pos + n] = next_obs
            self.dones[self._pos : self._pos + n] = dones
            self._pos = (self._pos + n) % self.capacity
        else:
            first = self.capacity - self._pos
            self.observations[self._pos :] = obs[:first]
            self.actions[self._pos :] = actions[:first]
            self.rewards[self._pos :] = rewards[:first]
            self.next_observations[self._pos :] = next_obs[:first]
            self.dones[self._pos :] = dones[:first]

            rest = n - first
            self.observations[:rest] = obs[first:]
            self.actions[:rest] = actions[first:]
            self.rewards[:rest] = rewards[first:]
            self.next_observations[:rest] = next_obs[first:]
            self.dones[:rest] = dones[first:]
            self._pos = rest

        self._size = min(self._size + n, self.capacity)

    def sample(
        self, batch_size: int, rng: np.random.Generator | None = None
    ) -> dict[str, np.ndarray]:
        """Sample a random batch of transitions.

        Returns
        -------
        dict with keys: observations, actions, rewards, next_observations, dones
        """
        if self._size == 0:
            raise ValueError("Cannot sample from empty buffer")
        if rng is None:
            rng = np.random.default_rng()

        idx = rng.integers(0, self._size, size=min(batch_size, self._size))
        return {
            "observations": self.observations[idx].copy(),
            "actions": self.actions[idx].copy(),
            "rewards": self.rewards[idx].copy(),
            "next_observations": self.next_observations[idx].copy(),
            "dones": self.dones[idx].copy(),
        }

    def sample_states(
        self, n: int, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Sample random observation states (for synthetic rollout starting points)."""
        if self._size == 0:
            raise ValueError("Cannot sample from empty buffer")
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.integers(0, self._size, size=min(n, self._size))
        return self.observations[idx].copy()

    def all_data(self) -> dict[str, np.ndarray]:
        """Return all stored transitions (up to current size)."""
        s = self._size
        return {
            "observations": self.observations[:s].copy(),
            "actions": self.actions[:s].copy(),
            "rewards": self.rewards[:s].copy(),
            "next_observations": self.next_observations[:s].copy(),
            "dones": self.dones[:s].copy(),
        }

    def save(self, path: str) -> None:
        """Save buffer contents to a compressed .npz file."""
        s = self._size
        np.savez_compressed(
            path,
            observations=self.observations[:s],
            actions=self.actions[:s],
            rewards=self.rewards[:s],
            next_observations=self.next_observations[:s],
            dones=self.dones[:s],
            _pos=np.array([self._pos]),
            _size=np.array([self._size]),
            _capacity=np.array([self.capacity]),
            _obs_dim=np.array([self.obs_dim]),
            _act_dim=np.array([self.act_dim]),
        )

    @classmethod
    def load(cls, path: str, capacity: int | None = None) -> "ReplayBuffer":
        """Load buffer from a .npz file.

        Parameters
        ----------
        path : str
            Path to the .npz file.
        capacity : int or None
            Override capacity. If None, uses the original capacity.
        """
        data = np.load(path)
        obs_dim = int(data["_obs_dim"][0])
        act_dim = int(data["_act_dim"][0])
        orig_cap = int(data["_capacity"][0])
        size = int(data["_size"][0])

        cap = capacity if capacity is not None else max(orig_cap, size)
        buf = cls(capacity=cap, obs_dim=obs_dim, act_dim=act_dim)

        n = min(size, cap)
        buf.observations[:n] = data["observations"][:n]
        buf.actions[:n] = data["actions"][:n]
        buf.rewards[:n] = data["rewards"][:n]
        buf.next_observations[:n] = data["next_observations"][:n]
        buf.dones[:n] = data["dones"][:n]
        buf._size = n
        buf._pos = n % cap

        return buf

"""Model Predictive Control planner using the dynamics ensemble.

At each decision step, the planner:
  1. Samples N candidate action sequences (horizon H)
  2. Rolls each forward through the dynamics model
  3. Scores by cumulative discounted reward
  4. Returns the first action of the best sequence

This is the "gun cleric" — the agent literally simulates combat
outcomes before choosing what to do.

Two planning strategies:
  - Random shooting: sample uniformly, pick best
  - CEM (Cross-Entropy Method): iteratively refine the action
    distribution toward high-reward sequences

Typical usage::

    planner = MPCPlanner(ensemble, act_nvec=[12,2,2,3], horizon=5)
    action = planner.plan(obs)  # returns best first action

    # Or wrap as a gym-compatible policy:
    policy = MPCPolicy(planner)
    action, _ = policy.predict(obs)
"""

from __future__ import annotations

import numpy as np

from training.dynamics_model import DynamicsEnsemble


class MPCPlanner:
    """Model Predictive Control planner for discrete action spaces.

    Parameters
    ----------
    ensemble : DynamicsEnsemble
        Trained dynamics model.
    act_nvec : list[int]
        Cardinality per discrete head (e.g. [12, 2, 2, 3]).
    horizon : int
        Planning horizon (number of steps to simulate forward).
    n_candidates : int
        Number of random action sequences to evaluate.
    gamma : float
        Discount factor for future rewards.
    n_elites : int
        Number of top sequences for CEM refinement (0 = random shooting only).
    cem_iterations : int
        CEM refinement iterations (only used if n_elites > 0).
    disagree_penalty : float
        Penalty coefficient for ensemble disagreement (0 = no penalty).
    base_policy : object or None
        If provided, uses the base policy to generate candidates instead of
        uniform random. This biases planning toward likely-good actions.
    """

    def __init__(
        self,
        ensemble: DynamicsEnsemble,
        act_nvec: list[int],
        horizon: int = 5,
        n_candidates: int = 256,
        gamma: float = 0.99,
        n_elites: int = 32,
        cem_iterations: int = 2,
        disagree_penalty: float = 0.0,
        base_policy=None,
    ):
        self.ensemble = ensemble
        self.act_nvec = list(act_nvec)
        self.act_dim = len(act_nvec)
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.gamma = gamma
        self.n_elites = n_elites
        self.cem_iterations = cem_iterations
        self.disagree_penalty = disagree_penalty
        self.base_policy = base_policy

    def plan(self, obs: np.ndarray) -> np.ndarray:
        """Plan the best action for a single observation.

        Parameters
        ----------
        obs : array, shape (obs_dim,)

        Returns
        -------
        action : array, shape (act_dim,) -- discrete action indices
        """
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        if self.n_elites > 0 and self.cem_iterations > 0:
            return self._plan_cem(obs)
        else:
            return self._plan_random_shooting(obs)

    def plan_batch(self, obs: np.ndarray) -> np.ndarray:
        """Plan actions for a batch of observations.

        Parameters
        ----------
        obs : array, shape (batch, obs_dim)

        Returns
        -------
        actions : array, shape (batch, act_dim)
        """
        batch = obs.shape[0]
        actions = np.zeros((batch, self.act_dim), dtype=np.float32)
        for i in range(batch):
            actions[i] = self.plan(obs[i])
        return actions

    def _sample_action_sequences(self, n: int, probs: list | None = None) -> np.ndarray:
        """Sample random action sequences.

        Returns shape (n, horizon, act_dim).
        """
        seqs = np.zeros((n, self.horizon, self.act_dim), dtype=np.float32)
        for h in range(self.act_dim):
            if probs is not None and probs[h] is not None:
                for t in range(self.horizon):
                    seqs[:, t, h] = np.random.choice(
                        self.act_nvec[h], size=n, p=probs[h][t]
                    )
            else:
                seqs[:, :, h] = np.random.randint(
                    0, self.act_nvec[h], size=(n, self.horizon)
                )
        return seqs

    def _evaluate_sequences(
        self, obs: np.ndarray, sequences: np.ndarray
    ) -> np.ndarray:
        """Evaluate action sequences by rolling through the dynamics model.

        Parameters
        ----------
        obs : array, shape (1, obs_dim)
        sequences : array, shape (n_candidates, horizon, act_dim)

        Returns
        -------
        returns : array, shape (n_candidates,) -- discounted cumulative reward
        """
        n = sequences.shape[0]
        current_obs = np.tile(obs, (n, 1))
        total_return = np.zeros(n, dtype=np.float32)
        discount = 1.0

        for t in range(self.horizon):
            actions = sequences[:, t, :]

            member_idx = np.random.randint(0, self.ensemble.n_models)
            next_obs, rewards, disagreement = self.ensemble.predict_next_obs(
                current_obs, actions, member_idx=member_idx
            )

            step_value = rewards - self.disagree_penalty * disagreement
            total_return += discount * step_value
            discount *= self.gamma

            current_obs = next_obs

        return total_return

    def _plan_random_shooting(self, obs: np.ndarray) -> np.ndarray:
        """Plan via random shooting: sample N sequences, pick best."""
        sequences = self._sample_action_sequences(self.n_candidates)
        returns = self._evaluate_sequences(obs, sequences)
        best_idx = np.argmax(returns)
        return sequences[best_idx, 0, :]

    def _plan_cem(self, obs: np.ndarray) -> np.ndarray:
        """Plan via Cross-Entropy Method: iteratively refine toward best."""
        probs = None

        for iteration in range(self.cem_iterations + 1):
            sequences = self._sample_action_sequences(self.n_candidates, probs=probs)
            returns = self._evaluate_sequences(obs, sequences)

            elite_idx = np.argsort(returns)[-self.n_elites:]
            elites = sequences[elite_idx]

            if iteration < self.cem_iterations:
                probs = []
                for h in range(self.act_dim):
                    head_probs = []
                    for t in range(self.horizon):
                        counts = np.bincount(
                            elites[:, t, h].astype(int),
                            minlength=self.act_nvec[h],
                        )
                        p = counts / counts.sum()
                        p = 0.8 * p + 0.2 / self.act_nvec[h]
                        head_probs.append(p)
                    probs.append(head_probs)

        best_idx = elite_idx[np.argmax(returns[elite_idx])]
        return sequences[best_idx, 0, :]

    def evaluate_planning_quality(
        self,
        obs_batch: np.ndarray,
        base_policy,
        n_comparisons: int = 100,
    ) -> dict:
        """Compare MPC actions vs base policy actions.

        Returns metrics on how much planning improves predicted returns.
        """
        n = min(n_comparisons, obs_batch.shape[0])
        obs = obs_batch[:n]

        mpc_returns = np.zeros(n, dtype=np.float32)
        base_returns = np.zeros(n, dtype=np.float32)

        for i in range(n):
            o = obs[i:i+1]
            mpc_action = self.plan(o)
            base_action, _ = base_policy.predict(o, deterministic=True)
            base_action = np.asarray(base_action, dtype=np.float32).reshape(1, -1)

            mpc_seq = np.tile(mpc_action, (1, self.horizon, 1))
            base_seq = np.tile(base_action, (1, self.horizon, 1))

            mpc_returns[i] = self._evaluate_sequences(o, mpc_seq)[0]
            base_returns[i] = self._evaluate_sequences(o, base_seq)[0]

        improvement = mpc_returns - base_returns

        return {
            "mpc_mean_return": float(mpc_returns.mean()),
            "base_mean_return": float(base_returns.mean()),
            "mean_improvement": float(improvement.mean()),
            "pct_improved": float((improvement > 0).mean() * 100),
            "max_improvement": float(improvement.max()),
        }


class MPCPolicy:
    """Wraps MPCPlanner as a gym-compatible policy for evaluation."""

    def __init__(self, planner: MPCPlanner):
        self.planner = planner

    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        if obs.ndim == 1:
            action = self.planner.plan(obs)
        else:
            action = self.planner.plan_batch(obs)
        return action, None

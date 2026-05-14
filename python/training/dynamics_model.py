"""Probabilistic MLP ensemble for learning environment dynamics.

Predicts (next_obs_delta, reward) from (obs, one_hot_action) using an
ensemble of independently trained MLPs with Gaussian NLL loss.

The ensemble provides both predictions and uncertainty estimates via
inter-model disagreement -- used for rollout truncation and model
exploitation detection.

Typical usage::

    model = DynamicsEnsemble(obs_dim=250, act_dim=4, act_nvec=[12,2,2,3])
    model.train_on_batch(obs, actions, next_obs, rewards)
    pred_delta, pred_reward, disagreement = model.predict(obs, actions)

Architecture choices:
  - Predicts state deltas (s' - s), not absolute states
  - Gaussian NLL loss (outputs mean + log_var per feature)
  - Bootstrap aggregation: each member trains on random 80% of data
  - One-hot encoding for MultiDiscrete actions
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot_actions(actions: np.ndarray | torch.Tensor, nvec: list[int]) -> torch.Tensor:
    """Convert MultiDiscrete actions to concatenated one-hot vectors.

    Parameters
    ----------
    actions : array-like, shape (batch, num_heads) or (num_heads,)
        Integer action indices per head.
    nvec : list[int]
        Cardinality of each discrete head (e.g. [12, 2, 2, 3]).

    Returns
    -------
    torch.Tensor, shape (batch, sum(nvec))
    """
    if isinstance(actions, np.ndarray):
        actions = torch.from_numpy(actions)
    actions = actions.long()
    if actions.ndim == 1:
        actions = actions.unsqueeze(0)

    parts = []
    for i, n in enumerate(nvec):
        head_vals = actions[:, i].clamp(0, n - 1)
        parts.append(F.one_hot(head_vals, num_classes=n).float())
    return torch.cat(parts, dim=-1)


class _EnsembleMember(nn.Module):
    """Single probabilistic MLP predicting (delta_obs, reward)."""

    def __init__(self, input_dim: int, obs_dim: int, hidden: int, n_layers: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        self.trunk = nn.Sequential(*layers)
        self.delta_mean = nn.Linear(hidden, obs_dim)
        self.delta_logvar = nn.Linear(hidden, obs_dim)
        self.reward_mean = nn.Linear(hidden, 1)
        self.reward_logvar = nn.Linear(hidden, 1)

        self.delta_logvar.weight.data.fill_(0.0)
        self.delta_logvar.bias.data.fill_(-2.0)
        self.reward_logvar.weight.data.fill_(0.0)
        self.reward_logvar.bias.data.fill_(-2.0)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return (
            self.delta_mean(h),
            self.delta_logvar(h),
            self.reward_mean(h),
            self.reward_logvar(h),
        )


class DynamicsEnsemble(nn.Module):
    """Ensemble of probabilistic MLPs for dynamics prediction.

    Parameters
    ----------
    obs_dim : int
        Observation vector dimensionality.
    act_dim : int
        Number of action heads.
    act_nvec : list[int]
        Cardinality per discrete action head (e.g. [12, 2, 2, 3]).
    n_models : int
        Number of ensemble members (default 5).
    hidden : int
        Hidden layer width (default 256).
    n_layers : int
        Number of hidden layers per member (default 3).
    lr : float
        Learning rate (default 1e-3).
    device : str
        Torch device.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_nvec: list[int],
        n_models: int = 5,
        hidden: int = 256,
        n_layers: int = 3,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_nvec = list(act_nvec)
        self.n_models = n_models
        self.hidden = hidden
        self.n_layers = n_layers
        self.device = device

        act_input_dim = sum(act_nvec)
        input_dim = obs_dim + act_input_dim

        self.members = nn.ModuleList(
            [
                _EnsembleMember(input_dim, obs_dim, hidden, n_layers)
                for _ in range(n_models)
            ]
        )
        self.to(device)

        self.optimizers = [
            torch.optim.Adam(m.parameters(), lr=lr) for m in self.members
        ]

        self._obs_mean = torch.zeros(obs_dim, device=device)
        self._obs_std = torch.ones(obs_dim, device=device)
        self._fitted = False

    def fit_normalizer(self, observations: np.ndarray) -> None:
        """Fit observation normalizer from data."""
        obs_t = torch.from_numpy(observations).float()
        self._obs_mean = obs_t.mean(dim=0).to(self.device)
        std = obs_t.std(dim=0).to(self.device)
        self._obs_std = torch.clamp(std, min=1e-6)
        self._fitted = True

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self._obs_mean) / self._obs_std

    def _prepare_input(
        self, obs: np.ndarray, actions: np.ndarray
    ) -> torch.Tensor:
        obs_t = torch.from_numpy(obs).float().to(self.device)
        act_oh = one_hot_actions(actions, self.act_nvec).to(self.device)
        obs_norm = self._normalize_obs(obs_t)
        return torch.cat([obs_norm, act_oh], dim=-1)

    def _compute_targets(
        self, obs: np.ndarray, next_obs: np.ndarray, rewards: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta = torch.from_numpy(next_obs - obs).float().to(self.device)
        delta_norm = delta / self._obs_std
        rew = torch.from_numpy(rewards).float().to(self.device).unsqueeze(-1)
        return delta_norm, rew

    def train_on_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        next_obs: np.ndarray,
        rewards: np.ndarray,
        bootstrap_ratio: float = 0.8,
    ) -> list[float]:
        """Train each ensemble member on a bootstrapped subset.

        Returns per-member losses.
        """
        n = obs.shape[0]
        x = self._prepare_input(obs, actions)
        delta_target, rew_target = self._compute_targets(obs, next_obs, rewards)

        losses = []
        for i, (member, opt) in enumerate(zip(self.members, self.optimizers)):
            mask = torch.rand(n, device=self.device) < bootstrap_ratio
            if mask.sum() < 2:
                mask[:2] = True

            x_b = x[mask]
            dt_b = delta_target[mask]
            rt_b = rew_target[mask]

            d_mean, d_logvar, r_mean, r_logvar = member(x_b)

            d_logvar = d_logvar.clamp(-10, 2)
            r_logvar = r_logvar.clamp(-10, 2)

            delta_loss = self._gaussian_nll(d_mean, d_logvar, dt_b)
            reward_loss = self._gaussian_nll(r_mean, r_logvar, rt_b)
            loss = delta_loss + reward_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(member.parameters(), 5.0)
            opt.step()

            losses.append(loss.item())

        return losses

    @staticmethod
    def _gaussian_nll(
        mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        var = logvar.exp()
        return (0.5 * (logvar + (target - mean).pow(2) / var)).mean()

    @torch.no_grad()
    def predict(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        member_idx: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict next-state delta and reward.

        Parameters
        ----------
        obs : array, shape (batch, obs_dim)
        actions : array, shape (batch, act_dim)
        member_idx : int or None
            If None, uses ensemble mean. If int, uses that member only.

        Returns
        -------
        pred_delta : array, shape (batch, obs_dim) -- in original (unnormalized) scale
        pred_reward : array, shape (batch,)
        disagreement : array, shape (batch,) -- ensemble std across members
        """
        x = self._prepare_input(obs, actions)

        if member_idx is not None:
            d_mean, _, r_mean, _ = self.members[member_idx](x)
            pred_delta = (d_mean * self._obs_std).cpu().numpy()
            pred_reward = r_mean.squeeze(-1).cpu().numpy()
            disagreement = np.zeros(obs.shape[0], dtype=np.float32)
        else:
            all_deltas = []
            all_rewards = []
            for member in self.members:
                d_mean, _, r_mean, _ = member(x)
                all_deltas.append(d_mean)
                all_rewards.append(r_mean.squeeze(-1))

            stacked_d = torch.stack(all_deltas, dim=0)
            stacked_r = torch.stack(all_rewards, dim=0)

            mean_d = stacked_d.mean(dim=0)
            mean_r = stacked_r.mean(dim=0)

            disagreement_d = stacked_d.std(dim=0).mean(dim=-1)
            disagreement_r = stacked_r.std(dim=0)
            disagreement = (disagreement_d + disagreement_r).cpu().numpy()

            pred_delta = (mean_d * self._obs_std).cpu().numpy()
            pred_reward = mean_r.cpu().numpy()

        return pred_delta, pred_reward, disagreement

    @torch.no_grad()
    def predict_next_obs(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        member_idx: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict next observation (obs + delta).

        Returns (next_obs, pred_reward, disagreement).
        """
        pred_delta, pred_reward, disagreement = self.predict(
            obs, actions, member_idx=member_idx
        )
        next_obs = obs + pred_delta
        return next_obs, pred_reward, disagreement

    @torch.no_grad()
    def rollout(
        self,
        start_obs: np.ndarray,
        policy,
        horizon: int,
        deterministic: bool = True,
        disagree_threshold: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate imagined rollouts from starting states.

        Parameters
        ----------
        start_obs : array, shape (batch, obs_dim)
        policy : object with predict(obs, deterministic) -> (actions, _)
        horizon : int
            Number of imagination steps.
        deterministic : bool
            Whether to use deterministic policy.
        disagree_threshold : float or None
            If set, truncate rollouts where ensemble disagreement exceeds this.

        Returns
        -------
        dict with keys:
            observations: (batch, horizon+1, obs_dim)
            actions: (batch, horizon, act_dim)
            rewards: (batch, horizon)
            dones: (batch, horizon) -- True where truncated by disagreement
            disagreements: (batch, horizon)
        """
        batch = start_obs.shape[0]
        all_obs = np.zeros((batch, horizon + 1, self.obs_dim), dtype=np.float32)
        all_act = np.zeros((batch, horizon, self.act_dim), dtype=np.float32)
        all_rew = np.zeros((batch, horizon), dtype=np.float32)
        all_done = np.zeros((batch, horizon), dtype=np.float32)
        all_disagree = np.zeros((batch, horizon), dtype=np.float32)

        obs = start_obs.copy()
        all_obs[:, 0] = obs
        alive = np.ones(batch, dtype=bool)

        for t in range(horizon):
            actions, _ = policy.predict(obs, deterministic=deterministic)
            actions = np.asarray(actions, dtype=np.float32)
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)

            member_idx = np.random.randint(0, self.n_models)
            next_obs, rewards, disagreement = self.predict_next_obs(
                obs, actions, member_idx=member_idx
            )

            if disagree_threshold is not None:
                truncated = disagreement > disagree_threshold
                alive &= ~truncated
                all_done[:, t] = (~alive).astype(np.float32)

            all_act[:, t] = actions
            all_rew[:, t] = rewards * alive
            all_disagree[:, t] = disagreement

            obs = next_obs
            all_obs[:, t + 1] = obs

        return {
            "observations": all_obs,
            "actions": all_act,
            "rewards": all_rew,
            "dones": all_done,
            "disagreements": all_disagree,
        }

    def save(self, path: str) -> None:
        """Save ensemble weights and normalizer."""
        torch.save(
            {
                "members": [m.state_dict() for m in self.members],
                "obs_mean": self._obs_mean,
                "obs_std": self._obs_std,
                "fitted": self._fitted,
                "config": {
                    "obs_dim": self.obs_dim,
                    "act_dim": self.act_dim,
                    "act_nvec": self.act_nvec,
                    "n_models": self.n_models,
                    "hidden": self.hidden,
                    "n_layers": self.n_layers,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu", **kwargs) -> "DynamicsEnsemble":
        """Load ensemble from checkpoint."""
        data = torch.load(path, map_location=device, weights_only=False)
        cfg = data["config"]
        model = cls(
            obs_dim=cfg["obs_dim"],
            act_dim=cfg["act_dim"],
            act_nvec=cfg["act_nvec"],
            n_models=cfg["n_models"],
            hidden=cfg.get("hidden", 256),
            n_layers=cfg.get("n_layers", 3),
            device=device,
            **kwargs,
        )
        for member, state in zip(model.members, data["members"]):
            member.load_state_dict(state)
        model._obs_mean = data["obs_mean"].to(device)
        model._obs_std = data["obs_std"].to(device)
        model._fitted = data["fitted"]
        return model

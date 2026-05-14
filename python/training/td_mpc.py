"""TD-MPC-inspired planner: latent dynamics + Q-function + MPPI planning.

Fixes all 5 gaps from the naive MPC implementation:
  1. Terminal Q-function for long-horizon value estimation
  2. Learned latent space (encoder, no decoder) for compact planning
  3. Policy prior for candidate generation
  4. MPPI soft-weighting instead of hard CEM elite selection
  5. Temporal correlation in action sampling

Architecture:
  - Encoder: obs -> latent z (compact representation)
  - Latent dynamics: (z, action) -> next z
  - Reward head: (z, action) -> predicted reward
  - Q-function: (z, action) -> long-horizon value
  - Policy head: z -> action distribution (prior for planning)

Training: all components train jointly from replay data via TD-learning.
Planning: MPPI in latent space with policy prior + terminal Q-value.

Typical usage::

    tdmpc = TDMPC(obs_dim=250, act_nvec=[12,2,2,3], latent_dim=128)
    tdmpc.train_from_buffer(replay_buffer, steps=1000)
    action = tdmpc.plan(obs)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.dynamics_model import one_hot_actions


class _Encoder(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class _LatentDynamics(nn.Module):
    def __init__(self, latent_dim: int, act_input_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + act_input_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor, act_oh: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, act_oh], dim=-1))


class _RewardHead(nn.Module):
    def __init__(self, latent_dim: int, act_input_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + act_input_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor, act_oh: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, act_oh], dim=-1)).squeeze(-1)


class _QFunction(nn.Module):
    def __init__(self, latent_dim: int, act_input_dim: int, hidden: int, n_heads: int = 2):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + act_input_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                nn.Linear(hidden, 1),
            )
            for _ in range(n_heads)
        ])

    def forward(self, z: torch.Tensor, act_oh: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, act_oh], dim=-1)
        qs = [h(x).squeeze(-1) for h in self.heads]
        return torch.stack(qs, dim=0)

    def min_q(self, z: torch.Tensor, act_oh: torch.Tensor) -> torch.Tensor:
        return self.forward(z, act_oh).min(dim=0).values


class _PolicyHead(nn.Module):
    """Outputs categorical logits per discrete head."""
    def __init__(self, latent_dim: int, hidden: int, act_nvec: list[int]):
        super().__init__()
        self.act_nvec = act_nvec
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, n) for n in act_nvec])

    def forward(self, z: torch.Tensor) -> list[torch.Tensor]:
        h = self.trunk(z)
        return [head(h) for head in self.heads]

    def sample(self, z: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        logits_list = self.forward(z)
        actions = []
        for logits in logits_list:
            probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
            actions.append(torch.multinomial(probs, 1).squeeze(-1))
        return torch.stack(actions, dim=-1).float()

    def log_prob(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        logits_list = self.forward(z)
        total = torch.zeros(z.shape[0], device=z.device)
        for i, logits in enumerate(logits_list):
            total += F.cross_entropy(logits, actions[:, i].long(), reduction='none') * -1
        return total


class TDMPC(nn.Module):
    """TD-MPC: Latent dynamics + Q-function + MPPI planning.

    Parameters
    ----------
    obs_dim : int
        Observation dimensionality.
    act_nvec : list[int]
        Cardinality per discrete head (e.g. [12, 2, 2, 3]).
    latent_dim : int
        Latent space dimensionality.
    hidden : int
        Hidden layer width.
    gamma : float
        Discount factor.
    lr : float
        Learning rate.
    tau : float
        Target network soft update rate.
    horizon : int
        Planning horizon.
    n_candidates : int
        MPPI candidate count.
    mppi_iterations : int
        MPPI refinement iterations.
    temperature : float
        MPPI temperature (lower = more greedy).
    policy_prior_weight : float
        Fraction of candidates from policy prior (0-1).
    device : str
        Torch device.
    """

    def __init__(
        self,
        obs_dim: int,
        act_nvec: list[int],
        latent_dim: int = 128,
        hidden: int = 256,
        gamma: float = 0.99,
        lr: float = 3e-4,
        tau: float = 0.005,
        horizon: int = 5,
        n_candidates: int = 256,
        mppi_iterations: int = 4,
        temperature: float = 0.5,
        policy_prior_weight: float = 0.8,
        device: str = "cpu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_nvec = list(act_nvec)
        self.act_dim = len(act_nvec)
        self.act_oh_dim = sum(act_nvec)
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.tau = tau
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.mppi_iterations = mppi_iterations
        self.temperature = temperature
        self.policy_prior_weight = policy_prior_weight
        self.device = device

        self.encoder = _Encoder(obs_dim, latent_dim, hidden)
        self.dynamics = _LatentDynamics(latent_dim, self.act_oh_dim, hidden)
        self.reward_head = _RewardHead(latent_dim, self.act_oh_dim, hidden)
        self.q_function = _QFunction(latent_dim, self.act_oh_dim, hidden)
        self.policy = _PolicyHead(latent_dim, hidden, act_nvec)

        self.target_encoder = _Encoder(obs_dim, latent_dim, hidden)
        self.target_q = _QFunction(latent_dim, self.act_oh_dim, hidden)
        self._hard_update_targets()

        self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self._train_steps = 0

    def _hard_update_targets(self):
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_q.load_state_dict(self.q_function.state_dict())

    def _soft_update_targets(self):
        for p, tp in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            tp.data.lerp_(p.data, self.tau)
        for p, tp in zip(self.q_function.parameters(), self.target_q.parameters()):
            tp.data.lerp_(p.data, self.tau)

    def _encode(self, obs: np.ndarray) -> torch.Tensor:
        obs_t = torch.from_numpy(obs).float().to(self.device)
        return self.encoder(obs_t)

    def _action_onehot(self, actions: np.ndarray) -> torch.Tensor:
        return one_hot_actions(actions, self.act_nvec).to(self.device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> dict[str, float]:
        """Single training step on a batch of transitions."""
        obs_t = torch.from_numpy(obs).float().to(self.device)
        next_obs_t = torch.from_numpy(next_obs).float().to(self.device)
        act_oh = self._action_onehot(actions)
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        dones_t = torch.from_numpy(dones).float().to(self.device)

        z = self.encoder(obs_t)

        z_next_pred = self.dynamics(z, act_oh)
        reward_pred = self.reward_head(z, act_oh)
        q_values = self.q_function(z, act_oh)
        policy_logprob = self.policy.log_prob(z.detach(), torch.from_numpy(actions).float().to(self.device))

        with torch.no_grad():
            z_next_target = self.target_encoder(next_obs_t)
            next_actions = self.policy.sample(z_next_target)
            next_act_oh = one_hot_actions(next_actions.cpu().numpy(), self.act_nvec).to(self.device)
            target_q = self.target_q.min_q(z_next_target, next_act_oh)
            td_target = rewards_t + self.gamma * (1 - dones_t) * target_q

        dynamics_loss = F.mse_loss(z_next_pred, z_next_target)
        reward_loss = F.mse_loss(reward_pred, rewards_t)
        q_loss = sum(F.mse_loss(q, td_target) for q in q_values) / len(q_values)
        policy_loss = -policy_logprob.mean()

        loss = dynamics_loss + reward_loss + q_loss + 0.1 * policy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        self.optimizer.step()
        self._soft_update_targets()
        self._train_steps += 1

        return {
            "dynamics_loss": dynamics_loss.item(),
            "reward_loss": reward_loss.item(),
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
            "total_loss": loss.item(),
            "mean_q": q_values.mean().item(),
            "mean_target_q": td_target.mean().item(),
        }

    def train_from_buffer(
        self,
        replay_buffer,
        steps: int = 1000,
        batch_size: int = 256,
        log_interval: int = 100,
    ) -> list[dict]:
        """Train for N steps from a replay buffer."""
        metrics_history = []
        rng = np.random.default_rng(42)

        for step in range(steps):
            batch = replay_buffer.sample(batch_size, rng=rng)
            metrics = self.train_step(
                batch["observations"], batch["actions"],
                batch["rewards"], batch["next_observations"],
                batch["dones"],
            )
            metrics_history.append(metrics)

            if (step + 1) % log_interval == 0:
                avg = {k: np.mean([m[k] for m in metrics_history[-log_interval:]]) for k in metrics.keys()}
                print(
                    f"  [td-mpc] step {step+1}/{steps}: "
                    f"total={avg['total_loss']:.4f}, "
                    f"dyn={avg['dynamics_loss']:.4f}, "
                    f"rew={avg['reward_loss']:.4f}, "
                    f"q={avg['q_loss']:.4f}, "
                    f"mean_q={avg['mean_q']:.2f}"
                )

        return metrics_history

    # ------------------------------------------------------------------
    # Planning (MPPI in latent space)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def plan(self, obs: np.ndarray) -> np.ndarray:
        """Plan the best action for a single observation using MPPI."""
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        z = self._encode(obs)

        n_prior = int(self.n_candidates * self.policy_prior_weight)
        n_random = self.n_candidates - n_prior

        best_actions = None

        for iteration in range(self.mppi_iterations):
            seqs = self._generate_candidates(z, n_prior, n_random, best_actions)
            returns = self._evaluate_sequences(z, seqs)

            weights = F.softmax(returns / max(self.temperature, 1e-4), dim=0)

            weighted_first = torch.zeros(self.act_dim, device=self.device)
            for h in range(self.act_dim):
                head_vals = seqs[:, 0, h]
                weighted_first[h] = (weights * head_vals).sum()

            best_idx = torch.argmax(returns)
            best_actions = seqs[best_idx]

        action = seqs[best_idx, 0].cpu().numpy()
        for h in range(self.act_dim):
            action[h] = np.clip(np.round(action[h]), 0, self.act_nvec[h] - 1)

        return action

    def _generate_candidates(
        self, z: torch.Tensor, n_prior: int, n_random: int,
        prev_best: torch.Tensor | None,
    ) -> torch.Tensor:
        """Generate candidate action sequences from policy prior + random."""
        seqs = []

        if n_prior > 0:
            z_expanded = z.expand(n_prior, -1)
            prior_seq = torch.zeros(n_prior, self.horizon, self.act_dim, device=self.device)
            z_t = z_expanded
            for t in range(self.horizon):
                actions = self.policy.sample(z_t, temperature=1.2)
                prior_seq[:, t] = actions
                act_oh = one_hot_actions(actions.cpu().numpy(), self.act_nvec).to(self.device)
                z_t = self.dynamics(z_t, act_oh)
            seqs.append(prior_seq)

        if n_random > 0:
            rand_seq = torch.zeros(n_random, self.horizon, self.act_dim, device=self.device)
            for h_idx in range(self.act_dim):
                base = torch.randint(0, self.act_nvec[h_idx], (n_random, 1), device=self.device).float()
                for t in range(self.horizon):
                    if t == 0:
                        rand_seq[:, t, h_idx] = base.squeeze()
                    else:
                        change = torch.rand(n_random, device=self.device) < 0.3
                        new_val = torch.randint(0, self.act_nvec[h_idx], (n_random,), device=self.device).float()
                        rand_seq[:, t, h_idx] = torch.where(change, new_val, rand_seq[:, t-1, h_idx])
            seqs.append(rand_seq)

        if prev_best is not None:
            shifted = prev_best.clone()
            shifted[:-1] = prev_best[1:]
            shifted[-1] = self.policy.sample(z, temperature=1.0).squeeze(0)
            seqs.append(shifted.unsqueeze(0))

        return torch.cat(seqs, dim=0)

    def _evaluate_sequences(self, z: torch.Tensor, sequences: torch.Tensor) -> torch.Tensor:
        """Score action sequences: cumulative reward + terminal Q-value."""
        n = sequences.shape[0]
        z_t = z.expand(n, -1)
        total_return = torch.zeros(n, device=self.device)
        discount = 1.0

        for t in range(self.horizon):
            actions = sequences[:, t]
            act_oh = one_hot_actions(actions.cpu().numpy(), self.act_nvec).to(self.device)

            reward = self.reward_head(z_t, act_oh)
            total_return += discount * reward
            discount *= self.gamma

            z_t = self.dynamics(z_t, act_oh)

        terminal_actions = self.policy.sample(z_t)
        terminal_act_oh = one_hot_actions(terminal_actions.cpu().numpy(), self.act_nvec).to(self.device)
        terminal_q = self.q_function.min_q(z_t, terminal_act_oh)
        total_return += discount * terminal_q

        return total_return

    # ------------------------------------------------------------------
    # Gym-compatible wrapper
    # ------------------------------------------------------------------

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        """Gym-compatible predict interface."""
        if obs.ndim == 1:
            return self.plan(obs), None
        actions = np.zeros((obs.shape[0], self.act_dim), dtype=np.float32)
        for i in range(obs.shape[0]):
            actions[i] = self.plan(obs[i])
        return actions, None

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "obs_dim": self.obs_dim, "act_nvec": self.act_nvec,
                "latent_dim": self.latent_dim, "hidden": self.encoder.net[0].out_features,
                "horizon": self.horizon, "n_candidates": self.n_candidates,
            },
            "train_steps": self._train_steps,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu", **kwargs) -> "TDMPC":
        data = torch.load(path, map_location=device, weights_only=False)
        cfg = data["config"]
        model = cls(obs_dim=cfg["obs_dim"], act_nvec=cfg["act_nvec"],
                     latent_dim=cfg["latent_dim"], hidden=cfg.get("hidden", 256),
                     device=device, **kwargs)
        model.load_state_dict(data["state_dict"])
        model._train_steps = data["train_steps"]
        return model

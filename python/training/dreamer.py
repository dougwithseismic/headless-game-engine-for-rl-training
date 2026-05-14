"""DreamerV3-lite: actor-critic trained in imagination.

Simplified DreamerV3 adapted for GhostLobby's state-vector observations
and discrete action spaces. Trains from a replay buffer:
  1. World model (RSSM): learns latent dynamics from real transitions
  2. Actor: trained entirely on imagined trajectories (no real env interaction)
  3. Critic: estimates value of imagined states

At inference: single forward pass through encoder + GRU + actor.
No planning, no search. All intelligence is baked into the policy.

Key DreamerV3 techniques:
  - Categorical latent state (32 categories x 32 classes)
  - Symlog predictions for rewards and values
  - KL balancing between prior and posterior
  - Percentile return normalization
  - Straight-through gradients for discrete latents

Usage::

    dreamer = Dreamer(obs_dim=250, act_nvec=[12,2,2,3])
    dreamer.train_from_buffer(replay_buffer, steps=5000)
    action = dreamer.act(obs)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical

from training.dynamics_model import one_hot_actions


def symlog(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * (x.abs() + 1).log()


def symexp(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * (x.abs().exp() - 1)


class RSSM(nn.Module):
    """Recurrent State-Space Model (DreamerV3 style).

    State = (h, z) where h is deterministic GRU hidden and z is stochastic.
    z is 32 categorical distributions x 32 classes = 1024 discrete dims.
    """

    def __init__(self, obs_dim: int, act_oh_dim: int, h_dim: int = 512,
                 z_cats: int = 32, z_classes: int = 32, mlp_dim: int = 256):
        super().__init__()
        self.h_dim = h_dim
        self.z_cats = z_cats
        self.z_classes = z_classes
        self.z_dim = z_cats * z_classes

        self.gru_input = nn.Linear(self.z_dim + act_oh_dim, h_dim)
        self.gru_cell = nn.GRUCell(h_dim, h_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(h_dim, mlp_dim), nn.SiLU(),
            nn.Linear(mlp_dim, z_cats * z_classes),
        )

        self.posterior_net = nn.Sequential(
            nn.Linear(h_dim + obs_dim, mlp_dim), nn.SiLU(),
            nn.Linear(mlp_dim, z_cats * z_classes),
        )

        self.obs_decoder = nn.Sequential(
            nn.Linear(h_dim + self.z_dim, mlp_dim), nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim), nn.SiLU(),
            nn.Linear(mlp_dim, obs_dim),
        )

        self.reward_head = nn.Sequential(
            nn.Linear(h_dim + self.z_dim, mlp_dim), nn.SiLU(),
            nn.Linear(mlp_dim, 1),
        )

        self.continue_head = nn.Sequential(
            nn.Linear(h_dim + self.z_dim, mlp_dim), nn.SiLU(),
            nn.Linear(mlp_dim, 1),
        )

    def initial_state(self, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch, self.h_dim, device=device)
        z = torch.zeros(batch, self.z_dim, device=device)
        return h, z

    def _sample_z(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample categorical z with straight-through gradients + unimix."""
        logits = logits.reshape(-1, self.z_cats, self.z_classes)
        probs = F.softmax(logits, dim=-1)
        probs = 0.99 * probs + 0.01 / self.z_classes
        dist = OneHotCategorical(probs=probs)
        sample = dist.sample()
        sample = sample + probs - probs.detach()
        return sample.reshape(-1, self.z_dim)

    def step(self, h: torch.Tensor, z: torch.Tensor, action_oh: torch.Tensor,
             obs: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """One RSSM step.

        Returns (next_h, next_z, prior_logits, posterior_logits).
        If obs is None, uses prior (imagination mode).
        """
        gru_in = self.gru_input(torch.cat([z, action_oh], dim=-1))
        gru_in = F.silu(gru_in)
        next_h = self.gru_cell(gru_in, h)

        prior_logits = self.prior_net(next_h)
        posterior_logits = None

        if obs is not None:
            posterior_logits = self.posterior_net(torch.cat([next_h, obs], dim=-1))
            next_z = self._sample_z(posterior_logits)
        else:
            next_z = self._sample_z(prior_logits)

        return next_h, next_z, prior_logits, posterior_logits

    def decode(self, h: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = torch.cat([h, z], dim=-1)
        obs_pred = self.obs_decoder(feat)
        reward_pred = self.reward_head(feat).squeeze(-1)
        continue_pred = self.continue_head(feat).squeeze(-1)
        return obs_pred, reward_pred, continue_pred

    def state_features(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([h, z], dim=-1)


class DreamerActor(nn.Module):
    """Actor that outputs discrete action distributions."""

    def __init__(self, feat_dim: int, hidden: int, act_nvec: list[int]):
        super().__init__()
        self.act_nvec = act_nvec
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, n) for n in act_nvec])

    def forward(self, feat: torch.Tensor) -> list[torch.Tensor]:
        h = self.trunk(feat)
        return [head(h) for head in self.heads]

    def sample(self, feat: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Sample actions with straight-through gradients (differentiable)."""
        logits_list = self.forward(feat)
        actions = []
        for logits in logits_list:
            logits_t = logits / max(temperature, 1e-4)
            probs = F.softmax(logits_t, dim=-1)
            probs = 0.99 * probs + 0.01 / logits.shape[-1]
            idx = torch.multinomial(probs, 1).squeeze(-1)
            actions.append(idx)
        return torch.stack(actions, dim=-1).float()

    def sample_onehot(self, feat: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Sample concatenated one-hot actions with straight-through gradients.

        Returns differentiable tensor of shape (batch, sum(act_nvec)).
        Use this during imagination to maintain gradient flow.
        """
        logits_list = self.forward(feat)
        parts = []
        for logits in logits_list:
            logits_t = logits / max(temperature, 1e-4)
            oh = F.gumbel_softmax(logits_t, tau=1.0, hard=True)
            parts.append(oh)
        return torch.cat(parts, dim=-1)

    def log_prob(self, feat: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        logits_list = self.forward(feat)
        total = torch.zeros(feat.shape[0], device=feat.device)
        for i, logits in enumerate(logits_list):
            probs = F.softmax(logits, dim=-1)
            probs = 0.99 * probs + 0.01 / logits.shape[-1]
            idx = actions[:, i].long().clamp(0, logits.shape[-1] - 1)
            total += probs.gather(1, idx.unsqueeze(1)).squeeze(1).log()
        return total

    def entropy(self, feat: torch.Tensor) -> torch.Tensor:
        logits_list = self.forward(feat)
        total = torch.zeros(feat.shape[0], device=feat.device)
        for logits in logits_list:
            probs = F.softmax(logits, dim=-1)
            probs = 0.99 * probs + 0.01 / logits.shape[-1]
            total += -(probs * probs.log()).sum(dim=-1)
        return total


class DreamerCritic(nn.Module):
    """Critic with symlog predictions."""

    def __init__(self, feat_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)


class Dreamer(nn.Module):
    """DreamerV3-lite for GhostLobby.

    Parameters
    ----------
    obs_dim : int
        Observation dimensionality.
    act_nvec : list[int]
        Cardinality per discrete head.
    h_dim : int
        GRU hidden dimensionality.
    z_cats, z_classes : int
        Categorical latent dimensions.
    mlp_dim : int
        MLP hidden width.
    imagine_horizon : int
        Steps to imagine forward during actor-critic training.
    gamma : float
        Discount factor.
    lambda_ : float
        GAE lambda for return computation.
    lr_world : float
        World model learning rate.
    lr_actor : float
        Actor learning rate.
    lr_critic : float
        Critic learning rate.
    kl_balance : float
        KL balancing ratio (fraction of gradient to prior).
    free_nats : float
        Minimum KL before penalizing.
    entropy_coef : float
        Entropy bonus for actor.
    device : str
        Torch device.
    """

    def __init__(
        self,
        obs_dim: int,
        act_nvec: list[int],
        h_dim: int = 512,
        z_cats: int = 32,
        z_classes: int = 32,
        mlp_dim: int = 256,
        imagine_horizon: int = 15,
        gamma: float = 0.997,
        lambda_: float = 0.95,
        lr_world: float = 1e-4,
        lr_actor: float = 3e-5,
        lr_critic: float = 3e-5,
        kl_balance: float = 0.8,
        free_nats: float = 1.0,
        entropy_coef: float = 3e-4,
        device: str = "cpu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_nvec = list(act_nvec)
        self.act_dim = len(act_nvec)
        self.act_oh_dim = sum(act_nvec)
        self.imagine_horizon = imagine_horizon
        self.gamma = gamma
        self.lambda_ = lambda_
        self.kl_balance = kl_balance
        self.free_nats = free_nats
        self.entropy_coef = entropy_coef
        self.device = device

        feat_dim = h_dim + z_cats * z_classes

        self.rssm = RSSM(obs_dim, self.act_oh_dim, h_dim, z_cats, z_classes, mlp_dim)
        self.actor = DreamerActor(feat_dim, mlp_dim, act_nvec)
        self.critic = DreamerCritic(feat_dim, mlp_dim)
        self.slow_critic = DreamerCritic(feat_dim, mlp_dim)
        self.slow_critic.load_state_dict(self.critic.state_dict())

        self.to(device)

        world_params = list(self.rssm.parameters())
        self.world_opt = torch.optim.Adam(world_params, lr=lr_world, eps=1e-8)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-8)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-8)

        self._train_steps = 0
        self._return_ema_low = 0.0
        self._return_ema_high = 1.0

    # ------------------------------------------------------------------
    # World model training
    # ------------------------------------------------------------------

    def _world_model_loss(
        self, obs_seq: torch.Tensor, act_seq: torch.Tensor,
        rew_seq: torch.Tensor, done_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute world model loss on a sequence batch.

        obs_seq: (seq_len, batch, obs_dim)
        act_seq: (seq_len, batch, act_dim)
        rew_seq: (seq_len, batch)
        done_seq: (seq_len, batch)
        """
        seq_len, batch = obs_seq.shape[:2]
        h, z = self.rssm.initial_state(batch, self.device)

        recon_loss = 0.0
        reward_loss = 0.0
        kl_loss = 0.0
        continue_loss = 0.0

        for t in range(seq_len):
            act_oh = one_hot_actions(act_seq[t].cpu().numpy(), self.act_nvec).to(self.device)
            h, z, prior_logits, post_logits = self.rssm.step(h, z, act_oh, obs_seq[t])

            obs_pred, rew_pred, cont_pred = self.rssm.decode(h, z)

            recon_loss += F.mse_loss(obs_pred, symlog(obs_seq[t]))
            reward_loss += F.mse_loss(rew_pred, symlog(rew_seq[t]))
            continue_loss += F.binary_cross_entropy_with_logits(
                cont_pred, 1.0 - done_seq[t]
            )

            prior = F.softmax(prior_logits.reshape(-1, self.rssm.z_cats, self.rssm.z_classes), dim=-1)
            post = F.softmax(post_logits.reshape(-1, self.rssm.z_cats, self.rssm.z_classes), dim=-1)

            dyn_loss = torch.sum(post.detach() * (post.detach().log() - prior.log()), dim=-1).sum(dim=-1).mean()
            rep_loss = torch.sum(post * (post.log() - prior.detach().log()), dim=-1).sum(dim=-1).mean()

            dyn_loss = torch.clamp(dyn_loss, min=self.free_nats)
            rep_loss = torch.clamp(rep_loss, min=self.free_nats)

            kl_loss += 0.5 * dyn_loss + 0.1 * rep_loss

        n = seq_len
        total = (recon_loss + reward_loss + continue_loss + kl_loss) / n

        return total, {
            "recon_loss": (recon_loss / n).item(),
            "reward_loss": (reward_loss / n).item(),
            "kl_loss": (kl_loss / n).item(),
            "continue_loss": (continue_loss / n).item(),
        }

    # ------------------------------------------------------------------
    # Imagination + actor-critic
    # ------------------------------------------------------------------

    def _imagine(self, start_h: torch.Tensor, start_z: torch.Tensor) -> dict:
        """Imagine H-step trajectories from starting states."""
        batch = start_h.shape[0]
        h, z = start_h, start_z

        feats = []
        rewards = []
        continues = []

        for t in range(self.imagine_horizon):
            feat = self.rssm.state_features(h, z)
            feats.append(feat)

            act_oh = self.actor.sample_onehot(feat)

            h, z, _, _ = self.rssm.step(h, z, act_oh, obs=None)

            _, rew_pred, cont_pred = self.rssm.decode(h, z)
            rewards.append(symexp(rew_pred))
            continues.append(torch.sigmoid(cont_pred))

        feats.append(self.rssm.state_features(h, z))

        return {
            "feats": torch.stack(feats, dim=0),
            "rewards": torch.stack(rewards, dim=0),
            "continues": torch.stack(continues, dim=0),
        }

    def _compute_returns(self, rewards: torch.Tensor, continues: torch.Tensor,
                         values: torch.Tensor) -> torch.Tensor:
        """Compute lambda-returns."""
        H = rewards.shape[0]
        returns = torch.zeros_like(rewards)
        last = values[-1]

        for t in reversed(range(H)):
            last = rewards[t] + self.gamma * continues[t] * (
                (1 - self.lambda_) * values[t + 1] + self.lambda_ * last
            )
            returns[t] = last

        return returns

    def _actor_critic_loss(self, start_h: torch.Tensor, start_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Compute actor and critic losses from imagined trajectories.

        Uses dynamics gradients: gradients flow through imagination back to actor
        via differentiable Gumbel-Softmax sampling.
        """
        imag = self._imagine(start_h.detach(), start_z.detach())

        with torch.no_grad():
            all_feats_detached = imag["feats"].detach()
            Hp1, B_s, Fd = all_feats_detached.shape
            values = self.slow_critic(all_feats_detached.reshape(Hp1 * B_s, Fd)).reshape(Hp1, B_s)
        returns = self._compute_returns(imag["rewards"], imag["continues"], values)

        low = self._return_ema_low
        high = self._return_ema_high
        scale = max(high - low, 1.0)
        normed_returns = (returns - low) / scale

        feats_no_last = imag["feats"][:-1]
        H, B, F_dim = feats_no_last.shape
        feats_flat = feats_no_last.reshape(H * B, F_dim)
        returns_flat = returns.reshape(H * B)
        normed_flat = normed_returns.reshape(H * B)

        critic_values_flat = self.critic(feats_flat.detach())
        critic_loss = F.mse_loss(critic_values_flat, symlog(returns_flat.detach()))

        normed_values_flat = (symexp(critic_values_flat).detach() - low) / scale
        advantage = normed_flat - normed_values_flat

        entropy = self.actor.entropy(feats_flat.detach())
        actor_loss = -(advantage + self.entropy_coef * entropy).mean()

        with torch.no_grad():
            alpha = 0.02
            self._return_ema_low += alpha * (float(returns.quantile(0.05)) - self._return_ema_low)
            self._return_ema_high += alpha * (float(returns.quantile(0.95)) - self._return_ema_high)

        return actor_loss, critic_loss, {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "mean_return": returns.mean().item(),
            "mean_value": critic_values_flat.mean().item(),
            "mean_entropy": entropy.mean().item(),
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_step(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        seq_len: int = 16,
    ) -> dict:
        """One training step: world model + imagination + actor-critic."""
        batch = obs.shape[0] // seq_len
        if batch < 1:
            batch = 1
            seq_len = obs.shape[0]

        n = batch * seq_len
        obs_t = torch.from_numpy(obs[:n]).float().to(self.device).reshape(seq_len, batch, -1)
        act_t = torch.from_numpy(actions[:n]).float().to(self.device).reshape(seq_len, batch, -1)
        rew_t = torch.from_numpy(rewards[:n]).float().to(self.device).reshape(seq_len, batch)
        done_t = torch.from_numpy(dones[:n]).float().to(self.device).reshape(seq_len, batch)

        wm_loss, wm_metrics = self._world_model_loss(obs_t, act_t, rew_t, done_t)
        self.world_opt.zero_grad()
        wm_loss.backward()
        nn.utils.clip_grad_norm_(self.rssm.parameters(), 100.0)
        self.world_opt.step()

        with torch.no_grad():
            h, z = self.rssm.initial_state(batch, self.device)
            t_mid = seq_len // 2
            for t in range(t_mid):
                act_oh = one_hot_actions(act_t[t].cpu().numpy(), self.act_nvec).to(self.device)
                h, z, _, _ = self.rssm.step(h, z, act_oh, obs_t[t])

        actor_loss, critic_loss, ac_metrics = self._actor_critic_loss(h, z)

        self.critic_opt.zero_grad()
        critic_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.critic_opt.step()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_opt.step()

        with torch.no_grad():
            for p, sp in zip(self.critic.parameters(), self.slow_critic.parameters()):
                sp.data.lerp_(p.data, 0.02)

        self._train_steps += 1

        return {**wm_metrics, **ac_metrics, "wm_loss": wm_loss.item()}

    def train_from_buffer(
        self,
        replay_buffer,
        steps: int = 5000,
        batch_size: int = 512,
        seq_len: int = 16,
        log_interval: int = 100,
    ) -> list[dict]:
        """Train Dreamer from replay buffer."""
        metrics_history = []
        rng = np.random.default_rng(42)

        for step in range(steps):
            batch = replay_buffer.sample(batch_size, rng=rng)
            metrics = self.train_step(
                batch["observations"], batch["actions"],
                batch["rewards"], batch["next_observations"],
                batch["dones"], seq_len=seq_len,
            )
            metrics_history.append(metrics)

            if (step + 1) % log_interval == 0:
                avg = {k: np.mean([m[k] for m in metrics_history[-log_interval:]]) for k in metrics.keys()}
                print(
                    f"  [dreamer] step {step+1}/{steps}: "
                    f"wm={avg['wm_loss']:.4f}, "
                    f"actor={avg['actor_loss']:.4f}, "
                    f"critic={avg['critic_loss']:.4f}, "
                    f"ret={avg['mean_return']:.2f}, "
                    f"ent={avg['mean_entropy']:.2f}"
                )

        return metrics_history

    # ------------------------------------------------------------------
    # Inference (reactive policy — no planning)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(self, obs: np.ndarray, h: torch.Tensor | None = None,
            z: torch.Tensor | None = None, temperature: float = 0.5,
           ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Select action from learned policy. Returns (action, h, z)."""
        obs_t = torch.from_numpy(obs).float().to(self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)

        batch = obs_t.shape[0]
        if h is None or z is None:
            h, z = self.rssm.initial_state(batch, self.device)

        dummy_act = torch.zeros(batch, self.act_oh_dim, device=self.device)
        h, z, _, _ = self.rssm.step(h, z, dummy_act, obs_t)

        feat = self.rssm.state_features(h, z)
        action = self.actor.sample(feat, temperature=temperature)

        return action.squeeze(0).cpu().numpy().astype(int), h, z

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        """Gym-compatible predict (stateless — resets GRU each call)."""
        temp = 0.1 if deterministic else 0.5
        action, _, _ = self.act(obs, temperature=temp)
        return action, None

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "obs_dim": self.obs_dim, "act_nvec": self.act_nvec,
                "h_dim": self.rssm.h_dim, "z_cats": self.rssm.z_cats,
                "z_classes": self.rssm.z_classes,
                "mlp_dim": self.rssm.prior_net[0].out_features,
                "imagine_horizon": self.imagine_horizon,
            },
            "train_steps": self._train_steps,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "Dreamer":
        data = torch.load(path, map_location=device, weights_only=False)
        cfg = data["config"]
        model = cls(
            obs_dim=cfg["obs_dim"], act_nvec=cfg["act_nvec"],
            h_dim=cfg["h_dim"], z_cats=cfg["z_cats"], z_classes=cfg["z_classes"],
            mlp_dim=cfg["mlp_dim"], imagine_horizon=cfg["imagine_horizon"],
            device=device,
        )
        model.load_state_dict(data["state_dict"])
        model._train_steps = data["train_steps"]
        return model

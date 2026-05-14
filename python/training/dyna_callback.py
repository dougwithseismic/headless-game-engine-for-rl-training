"""Dyna-style SB3 callback for world model augmentation.

Integrates the dynamics ensemble into PPO training:
  1. Collects real transitions from PPO rollouts into a replay buffer
  2. Periodically trains the dynamics ensemble on accumulated data
  3. At rollout end, uses model predictions to shape rewards:
     - Model-based reward shaping: adds discounted predicted future reward
     - Curiosity bonus: rewards states where ensemble disagrees (exploration)
  4. Logs dynamics metrics to TensorBoard

This stays PPO-compatible (on-policy) by modifying rewards on real
transitions rather than injecting off-policy synthetic data.

Typical usage::

    dyna = DynaCallback(
        obs_dim=250, act_dim=4, act_nvec=[12, 2, 2, 3],
        buffer_capacity=500_000, model_train_freq=2048,
        reward_shaping=True, shaping_coef=0.1,
    )
    model.learn(total_timesteps=3_000_000, callback=dyna)
"""

from __future__ import annotations

import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from training.replay_buffer import ReplayBuffer
from training.dynamics_model import DynamicsEnsemble
from training.dynamics_trainer import DynamicsTrainer


class DynaCallback(BaseCallback):
    """Dyna-style world model augmentation for SB3 PPO.

    Parameters
    ----------
    obs_dim : int
        Observation dimensionality.
    act_dim : int
        Number of action heads.
    act_nvec : list[int]
        Cardinality per discrete head (e.g. [12, 2, 2, 3]).
    buffer_capacity : int
        Replay buffer capacity.
    n_models : int
        Ensemble size (default 5).
    hidden : int
        Hidden layer width (default 256).
    n_layers : int
        Hidden layers per member (default 3).
    model_train_freq : int
        Train dynamics every N env steps (default 2048).
    model_train_steps : int
        Gradient steps per training call (default 10).
    model_batch_size : int
        Training batch size (default 256).
    warmup_steps : int
        Minimum buffer transitions before first model training (default 5000).
    model_lr : float
        Dynamics model learning rate (default 1e-3).
    save_dir : str or None
        Directory to save model checkpoints and replay buffer.
    verbose : int
        Verbosity level.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_nvec: list[int],
        buffer_capacity: int = 500_000,
        n_models: int = 5,
        hidden: int = 256,
        n_layers: int = 3,
        model_train_freq: int = 2048,
        model_train_steps: int = 10,
        model_batch_size: int = 256,
        warmup_steps: int = 5000,
        model_lr: float = 1e-3,
        reward_shaping: bool = True,
        shaping_coef: float = 0.1,
        shaping_horizon: int = 3,
        curiosity_coef: float = 0.01,
        save_dir: str | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_nvec = act_nvec
        self.model_train_freq = model_train_freq
        self.model_train_steps = model_train_steps
        self.warmup_steps = warmup_steps
        self.save_dir = save_dir
        self.reward_shaping = reward_shaping
        self.shaping_coef = shaping_coef
        self.shaping_horizon = shaping_horizon
        self.curiosity_coef = curiosity_coef

        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity, obs_dim=obs_dim, act_dim=act_dim
        )
        self.ensemble = DynamicsEnsemble(
            obs_dim=obs_dim,
            act_dim=act_dim,
            act_nvec=act_nvec,
            n_models=n_models,
            hidden=hidden,
            n_layers=n_layers,
            lr=model_lr,
        )
        self.trainer = DynamicsTrainer(
            ensemble=self.ensemble,
            replay_buffer=self.replay_buffer,
            batch_size=model_batch_size,
        )

        self._last_train_step = 0
        self._train_count = 0
        self._last_obs: np.ndarray | None = None

    def _on_training_start(self) -> None:
        if self.verbose:
            print(
                f"  [dyna] Initialized: obs_dim={self.obs_dim}, "
                f"act_nvec={self.act_nvec}, "
                f"ensemble={self.ensemble.n_models}x models, "
                f"buffer_cap={self.replay_buffer.capacity:,}, "
                f"shaping={self.reward_shaping} (coef={self.shaping_coef}, "
                f"horizon={self.shaping_horizon}, curiosity={self.curiosity_coef})"
            )

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        """Collect transitions into replay buffer."""
        infos = self.locals.get("infos", [])
        new_obs = self.locals.get("new_obs")
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        actions = self.locals.get("actions")

        if new_obs is None or self._last_obs is None:
            self._last_obs = self.locals.get("obs_tensor")
            if self._last_obs is not None and hasattr(self._last_obs, "cpu"):
                self._last_obs = self._last_obs.cpu().numpy()
            return True

        obs = self._last_obs
        if hasattr(obs, "cpu"):
            obs = obs.cpu().numpy()
        new_obs_np = np.asarray(new_obs, dtype=np.float32)
        actions_np = np.asarray(actions, dtype=np.float32)
        rewards_np = np.asarray(rewards, dtype=np.float32)
        dones_np = np.asarray(dones, dtype=np.float32)

        self.replay_buffer.add_batch(
            obs, actions_np, rewards_np, new_obs_np, dones_np
        )

        self._last_obs = new_obs_np.copy()

        if (
            len(self.replay_buffer) >= self.warmup_steps
            and self.num_timesteps - self._last_train_step >= self.model_train_freq
        ):
            self._train_dynamics()
            self._last_train_step = self.num_timesteps

        return True

    def _on_rollout_end(self) -> None:
        """Shape rewards using model predictions and log dynamics metrics."""
        if self._train_count == 0:
            return

        if self.reward_shaping:
            self._apply_reward_shaping()

        if self._train_count % 5 == 0:
            metrics = self.trainer.evaluate_accuracy(n_samples=500)
            self.logger.record("dynamics/state_mse", metrics["state_mse"])
            self.logger.record("dynamics/reward_mse", metrics["reward_mse"])
            self.logger.record("dynamics/disagreement", metrics["ensemble_disagreement"])
            median_r2 = float(np.median(metrics["r_squared"]))
            self.logger.record("dynamics/median_r_squared", median_r2)

        self.logger.record("dynamics/buffer_size", len(self.replay_buffer))
        self.logger.record("dynamics/train_count", self._train_count)

    def _apply_reward_shaping(self) -> None:
        """Add model-based reward shaping to the PPO rollout buffer.

        Two components:
        1. Future reward prediction: for each (obs, action) in the buffer,
           the model imagines `shaping_horizon` steps forward and adds the
           discounted imagined reward as a bonus. This teaches the agent
           about future consequences of current actions.
        2. Curiosity bonus: states where ensemble members disagree get a
           small bonus, encouraging exploration of uncertain regions.
        """
        buf = self.model.rollout_buffer
        n_steps, n_envs = buf.observations.shape[:2]
        obs_flat = buf.observations.reshape(-1, self.obs_dim)
        act_flat = buf.actions.reshape(-1, self.act_dim)

        _, pred_reward, disagreement = self.ensemble.predict(obs_flat, act_flat)

        shaping_bonus = np.zeros(obs_flat.shape[0], dtype=np.float32)

        if self.shaping_horizon > 1:
            current_obs = obs_flat.copy()
            current_act = act_flat.copy()
            gamma = self.model.gamma if hasattr(self.model, 'gamma') else 0.99
            discount = 1.0

            for h in range(self.shaping_horizon):
                _, step_reward, step_disagree = self.ensemble.predict(
                    current_obs, current_act
                )
                discount *= gamma
                shaping_bonus += discount * step_reward

                next_obs, _, _ = self.ensemble.predict_next_obs(
                    current_obs, current_act,
                    member_idx=np.random.randint(0, self.ensemble.n_models),
                )
                current_obs = next_obs

                with np.errstate(invalid='ignore'):
                    current_act_raw = self.model.policy.predict(
                        current_obs, deterministic=False
                    )[0]
                    current_act = np.asarray(current_act_raw, dtype=np.float32)
                    if current_act.ndim == 1:
                        current_act = np.tile(current_act, (current_obs.shape[0], 1))
        else:
            shaping_bonus = pred_reward

        curiosity_bonus = disagreement

        total_bonus = (
            self.shaping_coef * shaping_bonus
            + self.curiosity_coef * curiosity_bonus
        )

        total_bonus_2d = total_bonus.reshape(n_steps, n_envs)
        buf.rewards += total_bonus_2d

        mean_shaping = float(np.mean(np.abs(self.shaping_coef * shaping_bonus)))
        mean_curiosity = float(np.mean(self.curiosity_coef * curiosity_bonus))
        mean_total = float(np.mean(np.abs(total_bonus)))

        self.logger.record("dyna_shaping/mean_shaping_bonus", mean_shaping)
        self.logger.record("dyna_shaping/mean_curiosity_bonus", mean_curiosity)
        self.logger.record("dyna_shaping/mean_total_bonus", mean_total)
        self.logger.record("dyna_shaping/mean_disagreement", float(np.mean(disagreement)))

        if self.verbose and self._train_count % 10 == 0:
            print(
                f"  [dyna-shaping] shaping={mean_shaping:.4f}, "
                f"curiosity={mean_curiosity:.4f}, "
                f"total={mean_total:.4f}"
            )

    def _train_dynamics(self) -> None:
        """Train the dynamics ensemble on replay data."""
        t0 = time.perf_counter()
        metrics = self.trainer.train(
            n_steps=self.model_train_steps,
            refit_normalizer=(self._train_count == 0 or self._train_count % 10 == 0),
        )
        elapsed = time.perf_counter() - t0
        self._train_count += 1

        if self.verbose:
            print(
                f"  [dyna] Train #{self._train_count} at step "
                f"{self.num_timesteps:,} -- "
                f"loss={metrics['mean_loss']:.4f}, "
                f"val_loss={metrics['val_loss']:.4f}, "
                f"buffer={len(self.replay_buffer):,}, "
                f"time={elapsed:.1f}s"
            )

    def get_ensemble(self) -> DynamicsEnsemble:
        """Access the trained dynamics ensemble."""
        return self.ensemble

    def get_replay_buffer(self) -> ReplayBuffer:
        """Access the replay buffer."""
        return self.replay_buffer

    def get_trainer(self) -> DynamicsTrainer:
        """Access the dynamics trainer."""
        return self.trainer

    def save_checkpoint(self, path: str) -> None:
        """Save dynamics model and replay buffer."""
        import os
        os.makedirs(path, exist_ok=True)
        self.ensemble.save(os.path.join(path, "dynamics_ensemble.pt"))
        self.replay_buffer.save(os.path.join(path, "replay_buffer.npz"))

    def load_checkpoint(self, path: str) -> None:
        """Load dynamics model and replay buffer."""
        import os
        model_path = os.path.join(path, "dynamics_ensemble.pt")
        buf_path = os.path.join(path, "replay_buffer.npz")
        if os.path.exists(model_path):
            self.ensemble = DynamicsEnsemble.load(model_path, device=self.ensemble.device)
            self.trainer.ensemble = self.ensemble
        if os.path.exists(buf_path):
            self.replay_buffer = ReplayBuffer.load(buf_path)
            self.trainer.buffer = self.replay_buffer

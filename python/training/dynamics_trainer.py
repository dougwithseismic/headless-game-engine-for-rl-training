"""Training orchestrator for the dynamics ensemble.

Manages the training loop: batched gradient steps on replay data,
validation tracking, normalizer fitting, and TensorBoard logging.

Typical usage::

    trainer = DynamicsTrainer(ensemble, replay_buffer)
    metrics = trainer.train(n_steps=10, batch_size=256)
"""

from __future__ import annotations

import numpy as np

from training.replay_buffer import ReplayBuffer
from training.dynamics_model import DynamicsEnsemble


class DynamicsTrainer:
    """Orchestrates dynamics ensemble training on replay data.

    Parameters
    ----------
    ensemble : DynamicsEnsemble
        The dynamics model to train.
    replay_buffer : ReplayBuffer
        Source of real transitions.
    batch_size : int
        Training batch size (default 256).
    bootstrap_ratio : float
        Fraction of batch each member sees (default 0.8).
    val_ratio : float
        Fraction of buffer held out for validation (default 0.05).
    """

    def __init__(
        self,
        ensemble: DynamicsEnsemble,
        replay_buffer: ReplayBuffer,
        batch_size: int = 256,
        bootstrap_ratio: float = 0.8,
        val_ratio: float = 0.05,
    ):
        self.ensemble = ensemble
        self.buffer = replay_buffer
        self.batch_size = batch_size
        self.bootstrap_ratio = bootstrap_ratio
        self.val_ratio = val_ratio

        self._rng = np.random.default_rng(42)
        self._total_updates = 0

    def train(
        self,
        n_steps: int = 10,
        refit_normalizer: bool = True,
        tb_writer=None,
        global_step: int | None = None,
    ) -> dict[str, float]:
        """Run n_steps gradient updates on the ensemble.

        Parameters
        ----------
        n_steps : int
            Number of gradient steps per member.
        refit_normalizer : bool
            Whether to refit the observation normalizer before training.
        tb_writer : SummaryWriter or None
            TensorBoard writer for logging.
        global_step : int or None
            Global training step for TensorBoard x-axis.

        Returns
        -------
        dict with keys: mean_loss, val_loss, per_member_loss
        """
        if len(self.buffer) < self.batch_size:
            return {"mean_loss": float("nan"), "val_loss": float("nan"), "per_member_loss": []}

        if refit_normalizer:
            data = self.buffer.all_data()
            self.ensemble.fit_normalizer(data["observations"])

        all_losses: list[list[float]] = [[] for _ in range(self.ensemble.n_models)]

        for _ in range(n_steps):
            batch = self.buffer.sample(self.batch_size, rng=self._rng)
            losses = self.ensemble.train_on_batch(
                batch["observations"],
                batch["actions"],
                batch["next_observations"],
                batch["rewards"],
                bootstrap_ratio=self.bootstrap_ratio,
            )
            for i, loss in enumerate(losses):
                all_losses[i].append(loss)
            self._total_updates += 1

        mean_per_member = [np.mean(ml) for ml in all_losses]
        mean_loss = float(np.mean(mean_per_member))

        val_loss = self._validate()

        if tb_writer is not None and global_step is not None:
            tb_writer.add_scalar("dynamics/train_loss", mean_loss, global_step)
            tb_writer.add_scalar("dynamics/val_loss", val_loss, global_step)
            for i, ml in enumerate(mean_per_member):
                tb_writer.add_scalar(f"dynamics/member_{i}_loss", ml, global_step)
            tb_writer.add_scalar(
                "dynamics/total_updates", self._total_updates, global_step
            )
            tb_writer.add_scalar(
                "dynamics/buffer_size", len(self.buffer), global_step
            )

        return {
            "mean_loss": mean_loss,
            "val_loss": val_loss,
            "per_member_loss": mean_per_member,
        }

    def _validate(self) -> float:
        """Compute validation loss on a held-out sample."""
        n_val = max(int(len(self.buffer) * self.val_ratio), self.batch_size)
        n_val = min(n_val, len(self.buffer))
        batch = self.buffer.sample(n_val, rng=self._rng)

        pred_delta, pred_reward, _ = self.ensemble.predict(
            batch["observations"], batch["actions"]
        )

        actual_delta = batch["next_observations"] - batch["observations"]
        delta_mse = float(np.mean((pred_delta - actual_delta) ** 2))
        reward_mse = float(np.mean((pred_reward - batch["rewards"]) ** 2))

        return delta_mse + reward_mse

    def evaluate_accuracy(
        self,
        n_samples: int = 1000,
    ) -> dict[str, float | np.ndarray]:
        """Evaluate prediction accuracy with detailed metrics.

        Returns
        -------
        dict with keys:
            state_mse: overall state prediction MSE
            reward_mse: reward prediction MSE
            per_feature_mse: per-feature state MSE, shape (obs_dim,)
            r_squared: per-feature R-squared, shape (obs_dim,)
            ensemble_disagreement: mean disagreement across samples
        """
        n = min(n_samples, len(self.buffer))
        batch = self.buffer.sample(n, rng=self._rng)

        pred_delta, pred_reward, disagreement = self.ensemble.predict(
            batch["observations"], batch["actions"]
        )
        pred_next = batch["observations"] + pred_delta
        actual_next = batch["next_observations"]

        errors = (pred_next - actual_next) ** 2
        per_feature_mse = errors.mean(axis=0)
        state_mse = float(per_feature_mse.mean())
        reward_mse = float(np.mean((pred_reward - batch["rewards"]) ** 2))

        actual_var = np.var(actual_next - batch["observations"], axis=0)
        actual_var = np.maximum(actual_var, 1e-8)
        delta_errors = (pred_delta - (actual_next - batch["observations"])) ** 2
        r_squared = 1.0 - delta_errors.mean(axis=0) / actual_var

        return {
            "state_mse": state_mse,
            "reward_mse": reward_mse,
            "per_feature_mse": per_feature_mse,
            "r_squared": r_squared,
            "ensemble_disagreement": float(np.mean(disagreement)),
        }

    def evaluate_nstep(
        self,
        policy,
        horizon: int = 10,
        n_rollouts: int = 100,
    ) -> dict[str, np.ndarray]:
        """Evaluate N-step prediction divergence against real trajectories.

        Uses the replay buffer: picks starting states, rolls out with the
        dynamics model, and compares to actual next-states in the buffer
        (1-step only is exact; multi-step uses self-feeding model rollouts).

        Returns
        -------
        dict with keys:
            step_mse: MSE at each rollout step, shape (horizon,)
            step_max_err: max error at each step, shape (horizon,)
            step_disagreement: mean disagreement at each step, shape (horizon,)
        """
        starts = self.buffer.sample_states(n_rollouts, rng=self._rng)

        rollout = self.ensemble.rollout(
            starts, policy, horizon=horizon, deterministic=True
        )

        batch = self.buffer.sample(n_rollouts, rng=self._rng)
        one_step_pred, _, _ = self.ensemble.predict_next_obs(
            batch["observations"], batch["actions"]
        )
        one_step_mse = float(np.mean((one_step_pred - batch["next_observations"]) ** 2))

        step_disagree = rollout["disagreements"].mean(axis=0)

        obs_traj = rollout["observations"]
        step_drift = np.zeros(horizon, dtype=np.float32)
        for t in range(horizon):
            diff = obs_traj[:, t + 1] - obs_traj[:, 0]
            step_drift[t] = float(np.mean(np.abs(diff)))

        return {
            "one_step_mse": one_step_mse,
            "step_drift": step_drift,
            "step_disagreement": step_disagree,
        }

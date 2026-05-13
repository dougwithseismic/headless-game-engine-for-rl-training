"""
Reusable SB3 training callbacks for GhostLobby RL pipelines.

All callbacks inherit from ``stable_baselines3.common.callbacks.BaseCallback``
and follow the standard SB3 callback protocol. They are designed to be
composed via ``CallbackList`` in any training script.

Callbacks
---------
PlateauStopCallback
    Early-stopping when eval reward plateaus.
SelfPlaySwapCallback
    Periodically freeze the current policy and distribute it as the opponent.
ThroughputCallback
    Log training throughput (steps/sec) at regular intervals.
LiveSnapshotCallback
    Save model snapshots for a live spectator viewer to pick up.
EntropyScheduleCallback
    Anneal entropy coefficient linearly over training.
KLAnchorCallback
    Penalize policy divergence from a BC reference model.
"""

import copy
import os
import random
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


# ---------------------------------------------------------------------------
# PlateauStopCallback
# ---------------------------------------------------------------------------

class PlateauStopCallback(BaseCallback):
    """Patience-based early stopping driven by evaluation reward.

    Reads the ``evaluations.npz`` file produced by SB3's ``EvalCallback``
    and computes a rolling average of the last *window* evaluations. If no
    new best average is observed for *patience* consecutive checks, training
    is stopped.

    A *min_steps* floor prevents premature stopping during the noisy early
    phase of training.

    Parameters
    ----------
    eval_log_dir : str
        Directory containing ``evaluations.npz`` (same path you pass as
        ``log_path`` to ``EvalCallback``).
    patience : int
        Number of checks without improvement before stopping.
    check_freq : int
        Minimum timesteps between plateau checks.
    min_steps : int
        Do not check for plateau until this many timesteps have elapsed.
    window : int
        Number of most-recent evaluations to average.
    verbose : int
        Verbosity level (0 = silent, 1 = print status).
    """

    def __init__(
        self,
        eval_log_dir: str,
        patience: int = 15,
        check_freq: int = 200_000,
        min_steps: int = 1_000_000,
        window: int = 3,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_log_dir = eval_log_dir
        self.patience = patience
        self.check_freq = check_freq
        self.min_steps = min_steps
        self.window = window

        self._last_check_step = 0
        self._best_reward = -float("inf")
        self._evals_since_best = 0

    def _on_step(self) -> bool:
        # Skip checks until we have enough data.
        if self.num_timesteps < self.min_steps:
            return True
        if self.num_timesteps - self._last_check_step < self.check_freq:
            return True
        self._last_check_step = self.num_timesteps

        eval_path = os.path.join(self.eval_log_dir, "evaluations.npz")
        if not os.path.exists(eval_path):
            return True

        try:
            data = np.load(eval_path)
            rewards = [float(r.mean()) for r in data["results"]]
        except Exception:
            return True

        if len(rewards) < self.window:
            return True

        recent_avg = float(np.mean(rewards[-self.window :]))

        if recent_avg > self._best_reward:
            self._best_reward = recent_avg
            self._evals_since_best = 0
            if self.verbose:
                print(
                    f"\n  [early-stop] New best: {recent_avg:+.1f} "
                    f"at step {self.num_timesteps:,}"
                )
        else:
            self._evals_since_best += 1
            if self.verbose:
                print(
                    f"\n  [early-stop] No improvement for "
                    f"{self._evals_since_best}/{self.patience} checks "
                    f"(best={self._best_reward:+.1f}, "
                    f"current={recent_avg:+.1f})"
                )

        if self._evals_since_best >= self.patience:
            print(
                f"\n  [early-stop] Converged at step {self.num_timesteps:,} "
                f"-- best reward {self._best_reward:+.1f}, no improvement in "
                f"{self.patience} checks. Stopping.\n"
            )
            return False

        return True


# ---------------------------------------------------------------------------
# SelfPlaySwapCallback  (+  _NormalizedOpponent helper)
# ---------------------------------------------------------------------------

class _NormalizedOpponent:
    """Wraps a frozen policy snapshot for use as an opponent.

    Handles observation normalization (if the training environment uses
    ``VecNormalize``) and LSTM state tracking for recurrent policies.

    This is an internal helper -- users interact with
    :class:`SelfPlaySwapCallback` instead.
    """

    def __init__(self, policy, env_wrapper):
        self.policy = policy
        self.obs_rms = (
            copy.deepcopy(env_wrapper.obs_rms)
            if hasattr(env_wrapper, "obs_rms")
            else None
        )
        self.clip_obs = getattr(env_wrapper, "clip_obs", 10.0)
        self.epsilon = getattr(env_wrapper, "epsilon", 1e-8)
        self._lstm_states = None

    def reset_states(self):
        """Reset LSTM hidden states (call at episode boundary)."""
        self._lstm_states = None

    def predict(self, obs, deterministic=False, state=None, episode_start=None):
        """Produce an action, optionally normalizing observations first."""
        if self.obs_rms is not None:
            obs = np.clip(
                (obs - self.obs_rms.mean)
                / np.sqrt(self.obs_rms.var + self.epsilon),
                -self.clip_obs,
                self.clip_obs,
            ).astype(np.float32)

        use_state = state if state is not None else self._lstm_states
        if episode_start is None:
            episode_start = np.array([use_state is None])

        action, new_states = self.policy.predict(
            obs,
            state=use_state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        self._lstm_states = new_states
        return action, new_states


class SelfPlaySwapCallback(BaseCallback):
    """Periodically freeze the current policy and swap it in as the opponent.

    After an initial *scripted_warmup* period (where the environment uses its
    built-in scripted AI), the callback deep-copies the learner's policy at
    regular *swap_interval* steps, adds it to a rolling opponent pool, and
    distributes a randomly-sampled opponent to every sub-environment via
    ``env.set_opponent()``.

    The opponent is wrapped in :class:`_NormalizedOpponent` so that
    observation normalization statistics travel with the snapshot.

    Parameters
    ----------
    train_env : VecEnv
        The vectorized training environment (may be wrapped in
        ``VecNormalize``).
    swap_interval : int
        Timesteps between opponent swaps.
    scripted_warmup : int
        Timesteps of scripted-AI warmup before self-play begins.
    max_history : int
        Maximum number of opponent snapshots to keep in the pool.
    verbose : int
        Verbosity level.
    """

    def __init__(
        self,
        train_env,
        eval_env=None,
        swap_interval: int = 500_000,
        scripted_warmup: int = 1_000_000,
        max_history: int = 20,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env
        self.swap_interval = swap_interval
        self.scripted_warmup = scripted_warmup
        self.max_history = max_history

        self._last_swap_step = 0
        self.swap_count = 0
        self.opponent_history: list = []

    def _on_step(self) -> bool:
        if self.num_timesteps < self.scripted_warmup:
            return True
        if self.num_timesteps - self._last_swap_step >= self.swap_interval:
            self._swap_opponent()
            self._last_swap_step = self.num_timesteps
        return True

    def _swap_opponent(self):
        self.swap_count += 1

        # Deep-copy the current policy and freeze it.
        snapshot = copy.deepcopy(self.model.policy)
        snapshot.set_training_mode(False)
        self.opponent_history.append(snapshot)
        if len(self.opponent_history) > self.max_history:
            self.opponent_history.pop(0)

        # Sample an opponent uniformly from the history pool.
        opponent = random.choice(self.opponent_history)

        # Unwrap VecNormalize if present to reach the actual SubprocVecEnv.
        vec_env = self.train_env
        inner_vec = vec_env.venv if hasattr(vec_env, "venv") else vec_env

        wrapped = _NormalizedOpponent(opponent, vec_env)

        for i in range(inner_vec.num_envs):
            inner_vec.env_method("set_opponent", wrapped, indices=[i])

        if self.eval_env is not None:
            eval_inner = self.eval_env.venv if hasattr(self.eval_env, "venv") else self.eval_env
            eval_wrapped = _NormalizedOpponent(opponent, self.eval_env)
            for i in range(eval_inner.num_envs):
                eval_inner.env_method("set_opponent", eval_wrapped, indices=[i])

        if self.verbose:
            print(
                f"\n  [self-play] Swap #{self.swap_count} at step "
                f"{self.num_timesteps:,} -- pool size "
                f"{len(self.opponent_history)}\n"
            )


# ---------------------------------------------------------------------------
# ThroughputCallback
# ---------------------------------------------------------------------------

class ThroughputCallback(BaseCallback):
    """Log training throughput (steps/sec) at regular intervals.

    Prints a status line every *log_interval* timesteps showing the
    cumulative steps-per-second since training started.

    Parameters
    ----------
    log_interval : int
        Timesteps between throughput prints.
    verbose : int
        Verbosity level.
    """

    def __init__(self, log_interval: int = 50_000, verbose: int = 1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._t0: float | None = None

    def _on_training_start(self) -> None:
        self._t0 = time.perf_counter()

    def _on_step(self) -> bool:
        if self._t0 is None:
            return True
        if self.num_timesteps % self.log_interval == 0:
            elapsed = time.perf_counter() - self._t0
            sps = self.num_timesteps / elapsed if elapsed > 0 else 0.0
            if self.verbose:
                print(
                    f"  [{self.num_timesteps:>10,} steps] "
                    f"{sps:,.0f} steps/sec"
                )
        return True


# ---------------------------------------------------------------------------
# LiveSnapshotCallback
# ---------------------------------------------------------------------------

class LiveSnapshotCallback(BaseCallback):
    """Save periodic model snapshots for a live spectator viewer.

    The companion ``live_view`` module runs a spectator environment that
    loads the latest snapshot and streams telemetry to a WebSocket viewer.
    This callback writes those snapshots at a configurable interval.

    Parameters
    ----------
    snapshot_path : str
        File path (without ``.zip``) where snapshots are saved. SB3
        automatically appends ``.zip``.
    interval : int
        Timesteps between snapshot saves.
    verbose : int
        Verbosity level.
    """

    def __init__(
        self,
        snapshot_path: str,
        interval: int = 50_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.snapshot_path = snapshot_path
        self.interval = interval
        self._last_save_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save_step >= self.interval:
            self.model.save(self.snapshot_path)
            self._last_save_step = self.num_timesteps
            if self.verbose:
                print(
                    f"  [live-snapshot] Saved at step "
                    f"{self.num_timesteps:,}"
                )
        return True


# ---------------------------------------------------------------------------
# EntropyScheduleCallback
# ---------------------------------------------------------------------------

class EntropyScheduleCallback(BaseCallback):
    """Anneal entropy coefficient linearly over training.

    High entropy early encourages exploration. Low entropy later forces
    the policy to commit to precise actions (critical for aiming tasks).

    The coefficient is interpolated linearly from *start_value* to
    *end_value* over *total_steps* timesteps, then clamped at *end_value*
    for the remainder of training. Progress and the current coefficient
    are logged every 10 000 steps under the ``entropy_schedule/`` prefix.

    Parameters
    ----------
    start_value : float
        Entropy coefficient at the beginning of training.
    end_value : float
        Entropy coefficient once the schedule is complete.
    total_steps : int
        Number of timesteps over which to anneal.
    verbose : int
        Verbosity level.
    """

    def __init__(
        self,
        start_value: float = 0.01,
        end_value: float = 0.001,
        total_steps: int = 5_000_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = total_steps

    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / self.total_steps, 1.0)
        new_ent = self.start_value + (self.end_value - self.start_value) * progress
        self.model.ent_coef = new_ent
        if self.num_timesteps % 10_000 == 0:
            self.logger.record("entropy_schedule/ent_coef", new_ent)
            self.logger.record("entropy_schedule/progress", progress)
        return True


# ---------------------------------------------------------------------------
# KLAnchorCallback
# ---------------------------------------------------------------------------


class KLAnchorCallback(BaseCallback):
    """Penalize policy divergence from a BC reference model.

    Modifies rollout buffer rewards to include a KL-like penalty,
    keeping the PPO policy anchored to the BC initialization while
    allowing gradual improvement.

    At each rollout end, the callback:

    1. Forward-passes current observations through the **PPO policy** to
       obtain its mean action distribution.
    2. Forward-passes the same observations through the **frozen BC
       reference** to obtain its predicted actions.
    3. Computes the per-observation MSE between the two action vectors as
       a proxy for KL divergence (for Gaussian policies with fixed
       variance, MSE between means is proportional to KL).
    4. Subtracts ``beta * mse`` from the rollout buffer rewards.
    5. Anneals ``beta`` linearly from *beta_start* to *beta_end* over
       *anneal_steps* timesteps.

    Metrics logged to TensorBoard every rollout:
      - ``kl_anchor/mean_divergence`` -- mean MSE across all observations
      - ``kl_anchor/beta`` -- current penalty coefficient
      - ``kl_anchor/penalty`` -- ``beta * mean_divergence``

    Parameters
    ----------
    bc_ref_path : str
        Path to BC reference ``.pt`` file (saved via
        ``BCTrainer.save_reference()``).
    beta_start : float
        Initial penalty coefficient.
    beta_end : float
        Final penalty coefficient (annealed to).
    anneal_steps : int
        Timesteps over which to anneal beta linearly.
    verbose : int
        Verbosity level (0 = silent, 1 = print status).
    """

    def __init__(
        self,
        bc_ref_path: str,
        beta_start: float = 0.5,
        beta_end: float = 0.0,
        anneal_steps: int = 2_000_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.bc_ref_path = bc_ref_path
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.anneal_steps = anneal_steps
        self.bc_model = None  # loaded in _on_training_start

    def _on_training_start(self) -> None:
        """Load and freeze the BC reference model."""
        from training.bc_pretrain import BCTrainer

        self.bc_model = BCTrainer.load_reference(self.bc_ref_path)
        self.bc_model.feature_extractor.eval()
        self.bc_model.action_head.eval()

        if self.verbose:
            print(
                f"  [kl-anchor] Loaded BC reference from {self.bc_ref_path} "
                f"(obs_dim={self.bc_model.obs_dim}, "
                f"act_dim={self.bc_model.act_dim})"
            )

    def _current_beta(self) -> float:
        """Compute the current beta via linear annealing."""
        progress = min(self.num_timesteps / self.anneal_steps, 1.0)
        return self.beta_start + (self.beta_end - self.beta_start) * progress

    def _on_rollout_end(self) -> bool:
        """Compute divergence and subtract penalty from rollout rewards."""
        import gymnasium
        import torch

        if isinstance(self.model.action_space, gymnasium.spaces.MultiDiscrete):
            if not hasattr(self, "_kl_warned"):
                print(
                    "  [kl-anchor] Skipping: KL anchor not supported for "
                    "MultiDiscrete action spaces (designed for continuous)."
                )
                self._kl_warned = True
            return True

        beta = self._current_beta()

        if beta < 1e-6:
            return True

        buf = self.model.rollout_buffer
        obs = buf.observations  # (n_steps, n_envs, obs_dim)
        orig_shape = obs.shape
        obs_flat = obs.reshape(-1, orig_shape[-1])  # (n_steps * n_envs, obs_dim)

        with torch.no_grad():
            obs_tensor = torch.tensor(obs_flat, dtype=torch.float32).to(
                self.model.policy.device
            )
            dist = self.model.policy.get_distribution(obs_tensor)
            current_actions = dist.distribution.mean.cpu().numpy()

        bc_actions = self.bc_model.predict(obs_flat)

        mse = np.mean((current_actions - bc_actions) ** 2, axis=-1)
        mean_mse = float(mse.mean())

        penalty = mse.reshape(orig_shape[:-1])

        buf.rewards -= beta * penalty

        self.logger.record("kl_anchor/mean_divergence", mean_mse)
        self.logger.record("kl_anchor/beta", float(beta))
        self.logger.record("kl_anchor/penalty", float(beta * mean_mse))

        if self.verbose:
            print(
                f"  [kl-anchor] beta={beta:.4f}  "
                f"divergence={mean_mse:.6f}  "
                f"penalty={beta * mean_mse:.6f}"
            )

        return True

    def _on_step(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# BehaviorEvalCallback
# ---------------------------------------------------------------------------

class BehaviorEvalCallback(BaseCallback):
    """Eval callback that tracks behavioral metrics alongside standard eval.

    Replaces EvalCallback with additional behavior tracking:
    accuracy, shoot rate, K/D ratio, damage, and reward component breakdown.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        best_model_save_path: str | None = None,
        log_path: str | None = None,
        replay_path: str | None = None,
        max_replay_episodes: int = 2,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.replay_path = replay_path
        self.max_replay_episodes = max_replay_episodes
        self.deterministic = deterministic
        self.best_mean_reward = -float("inf")
        self._last_eval_step = 0
        self.evaluations_timesteps = []
        self.evaluations_results = []
        self.evaluations_length = []

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps
        self._run_eval()
        return True

    def _run_eval(self):
        episode_rewards = []
        episode_lengths = []
        all_behavior = []
        all_breakdown = []

        for ep in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = [False]
            episode_reward = 0.0
            episode_length = 0

            # Start recording for first N episodes
            record_this = self.replay_path and ep < self.max_replay_episodes
            if record_this:
                try:
                    inner = self.eval_env.venv if hasattr(self.eval_env, "venv") else self.eval_env
                    inner.env_method("start_recording", indices=[0])
                except Exception:
                    record_this = False

            # Handle LSTM states
            lstm_states = None
            episode_starts = np.ones((self.eval_env.num_envs,), dtype=bool)

            while not done[0]:
                try:
                    action, lstm_states = self.model.predict(
                        obs, state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=self.deterministic,
                    )
                except TypeError:
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)

                obs, reward, done, infos = self.eval_env.step(action)
                episode_starts = done
                episode_reward += reward[0]
                episode_length += 1

                if done[0]:
                    info = infos[0]
                    if "behavior" in info:
                        all_behavior.append(info["behavior"])
                    if "reward_breakdown" in info:
                        all_breakdown.append(info["reward_breakdown"])

                    # Save replay
                    if record_this:
                        try:
                            rpath = os.path.join(
                                self.replay_path,
                                f"eval_{self.num_timesteps:08d}_ep{ep:02d}.jsonl"
                            )
                            inner = self.eval_env.venv if hasattr(self.eval_env, "venv") else self.eval_env
                            inner.env_method("save_replay", rpath, indices=[0])
                        except Exception:
                            pass

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)

        # Store for evaluations.npz compatibility
        self.evaluations_timesteps.append(self.num_timesteps)
        self.evaluations_results.append(episode_rewards)
        self.evaluations_length.append(episode_lengths)

        # Save evaluations.npz
        if self.log_path:
            os.makedirs(self.log_path, exist_ok=True)
            np.savez(
                os.path.join(self.log_path, "evaluations.npz"),
                timesteps=self.evaluations_timesteps,
                results=self.evaluations_results,
                ep_lengths=self.evaluations_length,
            )

        # Log standard eval metrics
        self.logger.record("eval/mean_reward", float(mean_reward))
        self.logger.record("eval/mean_ep_length", float(mean_length))

        if self.verbose:
            print(f"Eval num_timesteps={self.num_timesteps}, "
                  f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"Episode length: {mean_length:.2f} +/- {np.std(episode_lengths):.2f}")

        # Log behavioral metrics
        if all_behavior:
            totals = {}
            for b in all_behavior:
                for k, v in b.items():
                    totals[k] = totals.get(k, 0) + v
            n = len(all_behavior)

            shots_fired = totals.get("shots_fired", 0)
            shots_hit = totals.get("shots_hit", 0)
            kills = totals.get("kills", 0)
            deaths = totals.get("deaths", 0)

            accuracy = shots_hit / max(shots_fired, 1)
            shoot_rate = shots_fired / max(sum(episode_lengths), 1)
            kd = kills / max(deaths, 1)

            self.logger.record("behavior/accuracy", accuracy)
            self.logger.record("behavior/shoot_rate", shoot_rate)
            self.logger.record("behavior/kd_ratio", kd)
            self.logger.record("behavior/kills_per_ep", kills / n)
            self.logger.record("behavior/deaths_per_ep", deaths / n)
            self.logger.record("behavior/damage_dealt_per_ep", totals.get("damage_dealt", 0) / n)
            self.logger.record("behavior/damage_taken_per_ep", totals.get("damage_taken", 0) / n)
            self.logger.record("behavior/shots_fired_per_ep", shots_fired / n)

            rounds_won = totals.get("rounds_won", 0)
            rounds_lost = totals.get("rounds_lost", 0)
            total_rounds = rounds_won + rounds_lost
            win_rate = rounds_won / max(total_rounds, 1)
            self.logger.record("behavior/win_rate", win_rate)
            self.logger.record("behavior/rounds_won_per_ep", rounds_won / n)
            self.logger.record("behavior/rounds_lost_per_ep", rounds_lost / n)

            if self.verbose:
                print(f"  Accuracy: {accuracy:.1%}, K/D: {kd:.2f}, "
                      f"Kills/ep: {kills/n:.1f}, Win rate: {win_rate:.1%}, Shoot rate: {shoot_rate:.2f}")

        # Log reward breakdown
        if all_breakdown:
            combined = {}
            for bd in all_breakdown:
                for k, v in bd.items():
                    combined[k] = combined.get(k, 0) + v
            n = len(all_breakdown)
            for category, total in combined.items():
                self.logger.record(f"reward_components/{category}", total / n)

        # Save best model
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            if self.best_model_save_path:
                os.makedirs(self.best_model_save_path, exist_ok=True)
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            if self.verbose:
                print(f"New best mean reward!")

        self.logger.dump(self.num_timesteps)

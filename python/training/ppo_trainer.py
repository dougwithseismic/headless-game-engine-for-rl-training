"""Unified PPO trainer for all GhostLobby scenarios.

Consolidates train_arena3d.py, train_tactical.py, train_drone.py,
train_selfplay.py, and train.py into a single configurable trainer.

Usage::

    from training.ppo_trainer import PPOTrainer

    trainer = PPOTrainer(
        scenario="arena3d",
        config_path="configs/arena3d_aim_targets.json",
        timesteps=3_000_000,
        n_envs=32,
        phase=1,
    )
    final_model_path = trainer.train()

All scenarios in the GYM_REGISTRY are supported. Optional features
(KL anchoring, entropy annealing, self-play, early stopping) are
enabled by passing the corresponding constructor arguments.
"""

import importlib
import json
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

from training.utils import resolve_config, make_run_dir, save_experiment, make_vec_env
from training.callbacks import (
    BehaviorEvalCallback,
    PlateauStopCallback,
    SelfPlaySwapCallback,
    ThroughputCallback,
    EntropyScheduleCallback,
    KLAnchorCallback,
)


# ---------------------------------------------------------------------------
# Scenario -> Gym class mapping
# ---------------------------------------------------------------------------

GYM_REGISTRY: dict[str, str] = {
    "cs_lite":       "glgym.gym_cs_lite.CsLiteGym",
    "cs-lite":       "glgym.gym_cs_lite.CsLiteGym",
    "cs_lite_dummy": "glgym.gym_cs_lite.CsLiteGym",
    "tactical":      "glgym.gym_tactical.TacticalGym",
}


def _import_gym_class(scenario: str):
    """Dynamically import and return the gym class for a given scenario.

    Looks up *scenario* in :data:`GYM_REGISTRY`, splits the dotted path
    into a module and class name, imports the module, and returns the
    class object.

    Parameters
    ----------
    scenario : str
        Scenario key, e.g. ``"arena3d"``, ``"drone-hover"``.

    Returns
    -------
    type
        The Gymnasium ``Env`` subclass for this scenario.

    Raises
    ------
    ValueError
        If *scenario* is not found in :data:`GYM_REGISTRY`.
    """
    path = GYM_REGISTRY.get(scenario)
    if not path:
        raise ValueError(
            f"Unknown scenario: {scenario!r}. "
            f"Available: {list(GYM_REGISTRY.keys())}"
        )
    module_path, class_name = path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


# ---------------------------------------------------------------------------
# PPOTrainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """Unified PPO training for all GhostLobby scenarios.

    Wraps the full training lifecycle: environment creation, PPO model
    construction (or resumption), callback assembly, training loop, and
    final model persistence.

    Parameters
    ----------
    scenario : str
        Scenario key from :data:`GYM_REGISTRY` (e.g. ``"arena3d"``).
    config_path : str
        Path to a GhostLobby JSON config file. Resolved to an absolute
        path via :func:`training.utils.resolve_config`.
    name : str, optional
        Human-readable run name. Defaults to *scenario*.
    lr : float
        PPO learning rate.
    n_steps : int
        Number of steps per rollout buffer per environment.
    batch_size : int
        Minibatch size for PPO updates.
    n_epochs : int
        Number of PPO update epochs per rollout.
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE lambda for advantage estimation.
    clip_range : float
        PPO clip parameter.
    ent_coef : float
        Entropy bonus coefficient.
    n_envs : int
        Number of parallel training environments.
    frame_skip : int
        Engine ticks per gym step.
    max_steps : int
        Maximum gym steps before episode truncation.
    phase : int, optional
        Curriculum phase for phased action masking.
    timesteps : int
        Total training timesteps.
    eval_freq : int
        Timesteps between evaluation runs (scaled by n_envs internally).
    checkpoint_freq : int
        Timesteps between model checkpoints (scaled by n_envs internally).
    n_eval_episodes : int
        Number of episodes per evaluation round.
    resume : str, optional
        Path to a saved model to resume training from.
    kl_anchor : str, optional
        Path to a BC reference ``.pt`` file for KL divergence penalty.
    kl_beta_start : float
        Starting beta for KL penalty annealing.
    kl_beta_end : float
        Final beta for KL penalty annealing.
    kl_anneal_steps : int
        Timesteps over which to anneal KL beta.
    entropy_schedule : tuple of (float, float), optional
        ``(start_value, end_value)`` for linear entropy annealing.
    entropy_schedule_steps : int, optional
        Timesteps over which to anneal entropy. Defaults to *timesteps*.
    self_play : bool
        Enable self-play opponent swapping.
    swap_interval : int
        Timesteps between self-play opponent swaps.
    scripted_warmup : int
        Timesteps of scripted AI before self-play begins.
    auto_stop : bool
        Enable plateau-based early stopping.
    patience : int
        Number of eval checks without improvement before early stop.
    run_dir : str, optional
        Explicit run directory. If ``None``, one is created via
        :func:`training.utils.make_run_dir`.
    """

    def __init__(
        self,
        scenario: str,
        config_path: str,
        name: str | None = None,
        # PPO hyperparams
        lr: float = 3e-4,
        n_steps: int = 4096,
        batch_size: int = 256,
        n_epochs: int = 4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        # Env params
        n_envs: int = 32,
        frame_skip: int = 4,
        max_steps: int = 2048,
        phase: int | None = None,
        # Training params
        timesteps: int = 3_000_000,
        eval_freq: int = 100_000,
        checkpoint_freq: int = 1_000_000,
        n_eval_episodes: int = 5,
        # Optional features
        resume: str | None = None,
        kl_anchor: str | None = None,
        kl_beta_start: float = 0.5,
        kl_beta_end: float = 0.0,
        kl_anneal_steps: int = 2_000_000,
        entropy_schedule: tuple[float, float] | None = None,
        entropy_schedule_steps: int | None = None,
        self_play: bool = False,
        swap_interval: int = 500_000,
        scripted_warmup: int = 1_000_000,
        auto_stop: bool = False,
        patience: int = 15,
        run_dir: str | None = None,
        dummy_vec_env: bool = False,
        lstm: bool = False,
        lstm_hidden_size: int = 256,
        track_behavior: bool = True,
    ):
        self.scenario = scenario
        self.config_path = resolve_config(config_path)
        self.name = name or scenario

        # PPO hyperparams
        self.lr = lr
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef

        # Env params
        self.n_envs = n_envs
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.phase = phase

        # Training params
        self.timesteps = timesteps
        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq
        self.n_eval_episodes = n_eval_episodes

        # Optional features
        self.resume = resume
        self.kl_anchor = kl_anchor
        self.kl_beta_start = kl_beta_start
        self.kl_beta_end = kl_beta_end
        self.kl_anneal_steps = kl_anneal_steps
        self.entropy_schedule = entropy_schedule
        self.entropy_schedule_steps = entropy_schedule_steps or timesteps
        self.self_play = self_play
        self.swap_interval = swap_interval
        self.scripted_warmup = scripted_warmup
        self.auto_stop = auto_stop
        self.patience = patience
        self.dummy_vec_env = dummy_vec_env
        self.lstm = lstm
        self.lstm_hidden_size = lstm_hidden_size
        self.track_behavior = track_behavior

        # Resolve gym class from registry
        self.gym_class = _import_gym_class(scenario)

        # Create or use provided run directory
        self.run_dir = run_dir or make_run_dir(self.name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_env_kwargs(self) -> dict:
        """Build keyword arguments for the gym constructor.

        Returns a dict of kwargs forwarded to the gym class via
        :func:`make_vec_env`. Note that ``config_path`` is handled
        separately by ``make_vec_env`` and is NOT included here.
        """
        kwargs: dict = {
            "scenario": self.scenario,
            "frame_skip": self.frame_skip,
            "max_steps": self.max_steps,
        }
        if self.phase is not None:
            kwargs["phase"] = self.phase
        return kwargs

    def _build_game_config(self) -> dict:
        """Load and return the parsed game config JSON."""
        with open(self.config_path) as f:
            return json.load(f)

    def _experiment_dict(self) -> dict:
        """Build the training arguments dict for experiment metadata.

        This dict captures all hyperparameters and feature flags so
        that any run can be reproduced later.
        """
        return {
            "scenario": self.scenario,
            "config_path": self.config_path,
            "phase": self.phase,
            "timesteps": self.timesteps,
            "lr": self.lr,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "n_envs": self.n_envs,
            "frame_skip": self.frame_skip,
            "max_steps": self.max_steps,
            "resume": self.resume,
            "kl_anchor": self.kl_anchor,
            "entropy_schedule": list(self.entropy_schedule) if self.entropy_schedule else None,
            "self_play": self.self_play,
            "lstm": self.lstm,
            "lstm_hidden_size": self.lstm_hidden_size if self.lstm else None,
            "track_behavior": self.track_behavior,
        }

    def _build_callbacks(self, vec_env, eval_env) -> CallbackList:
        """Assemble the full callback list based on configured features.

        Always includes: checkpoint, eval, throughput.
        Conditionally includes: KL anchor, entropy schedule, self-play,
        early stopping.

        Parameters
        ----------
        vec_env : VecEnv
            The training vectorized environment.
        eval_env : VecEnv
            The evaluation vectorized environment.

        Returns
        -------
        CallbackList
            Composed callback list ready for ``model.learn()``.
        """
        callbacks = []

        # --- Always-on callbacks ---

        callbacks.append(CheckpointCallback(
            save_freq=max(self.checkpoint_freq // self.n_envs, 1),
            save_path=os.path.join(self.run_dir, "checkpoints"),
            name_prefix=self.scenario,
        ))

        callbacks.append(BehaviorEvalCallback(
            eval_env,
            eval_freq=self.eval_freq,
            n_eval_episodes=self.n_eval_episodes,
            best_model_save_path=os.path.join(self.run_dir, "best"),
            log_path=os.path.join(self.run_dir, "eval_logs"),
            replay_path=os.path.join(self.run_dir, "replays") if self.track_behavior else None,
            deterministic=True,
        ))

        callbacks.append(ThroughputCallback())

        # --- Optional callbacks ---

        if self.kl_anchor:
            callbacks.append(KLAnchorCallback(
                bc_ref_path=self.kl_anchor,
                beta_start=self.kl_beta_start,
                beta_end=self.kl_beta_end,
                anneal_steps=self.kl_anneal_steps,
            ))

        if self.entropy_schedule:
            start, end = self.entropy_schedule
            callbacks.append(EntropyScheduleCallback(
                start_value=start,
                end_value=end,
                total_steps=self.entropy_schedule_steps,
            ))

        if self.self_play:
            callbacks.append(SelfPlaySwapCallback(
                train_env=vec_env,
                swap_interval=self.swap_interval,
                scripted_warmup=self.scripted_warmup,
            ))

        if self.auto_stop:
            callbacks.append(PlateauStopCallback(
                eval_log_dir=os.path.join(self.run_dir, "eval_logs"),
                patience=self.patience,
            ))

        return CallbackList(callbacks)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> str:
        """Run the full training loop.

        Creates vectorized environments, builds or loads the PPO model,
        assembles callbacks, runs ``model.learn()``, saves the final
        model, and cleans up environments.

        Returns
        -------
        str
            Path to the saved final model zip file.
        """
        env_kwargs = self._make_env_kwargs()

        # Create training and evaluation environments
        vec_env = make_vec_env(
            self.gym_class,
            config_path=self.config_path,
            n_envs=self.n_envs,
            dummy=self.dummy_vec_env,
            **env_kwargs,
        )
        eval_kwargs = dict(env_kwargs)
        if self.track_behavior:
            eval_kwargs["track_behavior"] = True
        eval_env = make_vec_env(
            self.gym_class,
            config_path=self.config_path,
            n_envs=min(4, self.n_envs),
            dummy=self.dummy_vec_env,
            **eval_kwargs,
        )

        # Create or resume model
        if self.lstm:
            if RecurrentPPO is None:
                raise ImportError("sb3-contrib required for LSTM: pip install sb3-contrib")
            AlgoClass = RecurrentPPO
            policy_name = "MlpLstmPolicy"
            policy_kwargs = {"lstm_hidden_size": self.lstm_hidden_size}
        else:
            AlgoClass = PPO
            policy_name = "MlpPolicy"
            policy_kwargs = None

        if self.resume:
            print(f"Resuming from: {self.resume}")
            model = AlgoClass.load(self.resume, env=vec_env)
            model.learning_rate = self.lr
            model.tensorboard_log = os.path.join(self.run_dir, "tb")
        else:
            model = AlgoClass(
                policy_name,
                vec_env,
                learning_rate=self.lr,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                ent_coef=self.ent_coef,
                verbose=1,
                tensorboard_log=os.path.join(self.run_dir, "tb"),
                **({"policy_kwargs": policy_kwargs} if policy_kwargs else {}),
            )

        # Persist experiment metadata (game config + training args)
        game_config = self._build_game_config()
        training_args = self._experiment_dict()
        save_experiment(self.run_dir, game_config, training_args)

        # Build callbacks
        callbacks = self._build_callbacks(vec_env, eval_env)

        # Print training banner
        print(f"=== {self.name} Training ===")
        print(f"  Scenario: {self.scenario}")
        print(f"  Config: {self.config_path}")
        print(f"  Phase: {self.phase}")
        print(f"  Timesteps: {self.timesteps:,}")
        print(f"  Envs: {self.n_envs}")
        print(f"  KL anchor: {self.kl_anchor}")
        print(f"  Entropy schedule: {self.entropy_schedule}")
        print(f"  Self-play: {self.self_play}")
        print(f"  LSTM: {self.lstm} (hidden={self.lstm_hidden_size})" if self.lstm else f"  LSTM: False")
        print(f"  Run dir: {self.run_dir}")

        # Train
        t0 = time.time()
        model.learn(
            total_timesteps=self.timesteps,
            callback=callbacks,
            tb_log_name=self.name,
        )
        elapsed = time.time() - t0

        # Save final model
        final_path = os.path.join(self.run_dir, "final_model")
        model.save(final_path)
        print(f"\n=== Done in {elapsed:.0f}s. Model: {final_path}.zip ===")

        # Cleanup
        vec_env.close()
        eval_env.close()

        return f"{final_path}.zip"

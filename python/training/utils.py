"""
Shared training utilities for GhostLobby RL pipelines.

Provides config resolution, run directory management, experiment metadata
persistence, vectorized environment construction, and model loading. These
are the common building blocks every training script needs -- extracted here
so individual scripts stay focused on hyperparameters and scenario logic.

This module has NO dependency on the ``glgym`` package. It only imports
stable-baselines3 lazily inside the functions that need it.
"""

import json
import os
from datetime import datetime


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def resolve_config(path: str) -> str:
    """Resolve a config path to an absolute path.

    Tries CWD-relative first, then project root-relative.
    """
    if os.path.isabs(path):
        return path
    cwd_resolved = os.path.abspath(path)
    if os.path.exists(cwd_resolved):
        return cwd_resolved
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return os.path.join(project_root, path)


# ---------------------------------------------------------------------------
# Run directory management
# ---------------------------------------------------------------------------

def make_run_dir(name: str, base: str = "runs") -> str:
    """Create a timestamped run directory with standard sub-directories.

    The directory layout matches the convention used across all GhostLobby
    training scripts::

        {base}/{name}_{YYYYMMDD_HHMMSS}/
            checkpoints/
            eval_logs/
            tb/
            best/

    Parameters
    ----------
    name : str
        Short descriptive name for the run (e.g. ``"tactical"``).
    base : str
        Parent directory for all runs. Defaults to ``"runs"``.

    Returns
    -------
    str
        Absolute (or relative, matching *base*) path to the created run
        directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{name}_{timestamp}"
    run_dir = os.path.join(base, run_name)

    for subdir in ("checkpoints", "eval_logs", "tb", "best", "replays"):
        os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)

    return run_dir


# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------

def save_experiment(run_dir: str, config: dict, args: dict) -> None:
    """Persist experiment metadata as ``experiment.json`` inside *run_dir*.

    The saved JSON contains the game config, CLI arguments, and a UTC
    timestamp so that any run can be reproduced or audited later.

    Parameters
    ----------
    run_dir : str
        Path to the run directory (created by :func:`make_run_dir`).
    config : dict
        Parsed game configuration (the contents of the JSON config file).
    args : dict
        Training arguments -- typically ``vars(parsed_args)``.
    """
    experiment = {
        "name": os.path.basename(run_dir),
        "config": config,
        "args": args,
        "start_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    path = os.path.join(run_dir, "experiment.json")
    with open(path, "w") as f:
        json.dump(experiment, f, indent=2)


# ---------------------------------------------------------------------------
# Vectorized environment factory
# ---------------------------------------------------------------------------

def make_vec_env(gym_class, config_path: str, n_envs: int, dummy: bool = False, **kwargs):
    """Create a vectorized environment from a Gymnasium-compatible class.

    Uses ``SubprocVecEnv`` by default for true parallelism, or
    ``DummyVecEnv`` when *dummy* is ``True`` (useful when subprocess
    spawning is problematic, e.g. Python 3.14 + numpy compat issues,
    or for small env counts where the subprocess overhead isn't worth it).

    Parameters
    ----------
    gym_class : type
        A Gymnasium ``Env`` subclass (e.g. ``SelfPlayGym``, ``Arena3DGym``).
    config_path : str
        Path to the JSON config file, passed as the first positional arg.
    n_envs : int
        Number of parallel environments.
    dummy : bool
        If ``True``, use ``DummyVecEnv`` instead of ``SubprocVecEnv``.
        Default is ``False``.
    **kwargs
        Extra keyword arguments forwarded to the gym constructor
        (e.g. ``frame_skip=4``, ``max_steps=2048``).

    Returns
    -------
    VecEnv
        A vectorized environment ready for SB3 training.
    """
    def _make(rank: int):
        def _init():
            return gym_class(config_path=config_path, **kwargs)
        return _init

    env_fns = [_make(i) for i in range(n_envs)]

    if dummy:
        from stable_baselines3.common.vec_env import DummyVecEnv
        return DummyVecEnv(env_fns)
    else:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        return SubprocVecEnv(env_fns)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(path: str, env=None):
    """Load a saved SB3 model, auto-detecting PPO vs RecurrentPPO.

    Tries ``PPO.load`` first. If that fails (e.g. the zip contains LSTM
    weights), falls back to ``sb3_contrib.RecurrentPPO``.

    Parameters
    ----------
    path : str
        Path to the saved model zip (with or without ``.zip`` extension).
    env : optional
        Environment to bind to the loaded model.  ``None`` is fine for
        inference-only use.

    Returns
    -------
    tuple[BaseAlgorithm, bool]
        ``(model, is_recurrent)`` -- the loaded model and a flag indicating
        whether it is a RecurrentPPO (LSTM) model.
    """
    from stable_baselines3 import PPO

    try:
        return PPO.load(path, env=env), False
    except Exception:
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(path, env=env), True


def predict_with_state(model, obs, state=None, episode_start=True, deterministic=True):
    """Unified predict that works for both PPO and RecurrentPPO.

    Parameters
    ----------
    model : BaseAlgorithm
        A loaded SB3 model (PPO or RecurrentPPO).
    obs : np.ndarray
        Current observation.
    state : optional
        LSTM hidden state for recurrent models. ``None`` for feed-forward.
    episode_start : bool
        Whether this is the first step of an episode (resets LSTM state).
    deterministic : bool
        Whether to use greedy (deterministic) action selection.

    Returns
    -------
    tuple[np.ndarray, Any]
        ``(action, new_state)`` -- for PPO, *new_state* is always ``None``.
    """
    import numpy as np

    action, new_state = model.predict(
        obs,
        state=state,
        episode_start=np.array([episode_start]),
        deterministic=deterministic,
    )
    return action, new_state

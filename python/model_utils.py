"""Shared model loading and prediction utilities for PPO and RecurrentPPO."""

import numpy as np


def load_model(path):
    """Load a model, auto-detecting PPO vs RecurrentPPO. Returns (model, is_recurrent)."""
    from stable_baselines3 import PPO
    try:
        return PPO.load(path), False
    except Exception:
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(path), True


def predict_with_state(model, obs, state=None, episode_start=True, deterministic=True):
    """Unified predict that works for both PPO and RecurrentPPO.
    Returns (action, new_state). For PPO, state is always None."""
    action, new_state = model.predict(
        obs,
        state=state,
        episode_start=np.array([episode_start]),
        deterministic=deterministic,
    )
    return action, new_state

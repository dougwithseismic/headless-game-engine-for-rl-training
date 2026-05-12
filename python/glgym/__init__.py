"""
glgym -- Gymnasium wrappers for the GhostLobby headless ECS engine.

Provides a BaseGhostLobbyGym with scenario-specific subclasses:
  - CsLiteGym    (5v5 tactical shooter with bomb, A* nav, cover)
  - TacticalGym  (top-down tactical with 12-direction movement)

Usage:
    from glgym import CsLiteGym

    env = CsLiteGym("configs/cs_lite/5v5_elimination.json")
    obs, info = env.reset()
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
"""

__version__ = "0.2.0"

from glgym.gym_base import BaseGhostLobbyGym
from glgym.gym_cs_lite import CsLiteGym
from glgym.gym_tactical import TacticalGym

__all__ = [
    "BaseGhostLobbyGym",
    "CsLiteGym",
    "TacticalGym",
]

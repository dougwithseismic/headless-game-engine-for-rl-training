"""
Tactical Gymnasium wrapper for GhostLobby top-down tactical scenarios.

Action heads (MultiDiscrete, populated from engine action_space()):
  [0] move_target: Discrete(12)
  [1] aim_delta:   Discrete(N)
  [2] shoot:       Discrete(2)
  [3] weapon_select: Discrete(2)

Supports opponent for self-play and curriculum phases.
"""

import random

from glgym.gym_base import BaseGhostLobbyGym


class TacticalGym(BaseGhostLobbyGym):
    """Top-down tactical scenario with MultiDiscrete action space.

    Action heads are populated from the engine's action_space() definition.
    Integer indices are passed directly to the engine.

    Curriculum phases:
      Phase 1: Scripted movement (random compass direction held 8-20 steps).
               Agent learns aim + shoot. Weapon locked to rifle.
      Phase 2: Agent controls all except weapon_select (locked to rifle).
      Phase 3+: Full action space.

    Args:
        config_path: Path to a tactical GhostLobby JSON config.
        scenario: Scenario name, defaults to "fps" (tactical deathmatch).
        frame_skip: Engine ticks per gym step.
        max_steps: Max gym steps before truncation.
        phase: Curriculum phase. None means no masking.
    """

    def __init__(
        self,
        config_path: str,
        scenario: str = "fps",
        frame_skip: int = 4,
        max_steps: int = 2048,
        phase: int | None = None,
        track_behavior: bool = False,
    ):
        super().__init__(
            config_path=config_path,
            scenario=scenario,
            frame_skip=frame_skip,
            max_steps=max_steps,
            phase=phase,
            track_behavior=track_behavior,
        )

    def _init_agent_ids(self) -> None:
        """Randomise sides for self-play on each reset."""
        self.agent_id = random.randint(0, 1)
        self.opp_id = 1 - self.agent_id

    def _apply_phase_mask(self, action_list: list[float]) -> list[float]:
        """Lock action heads based on curriculum phase.

        Phase 1: Scripted movement (random compass direction, held 8-20 steps).
                 Agent learns aim + shoot only. Weapon locked to rifle.
        Phase 2: Agent controls movement + aim + shoot. Weapon locked.
        Phase 3+: Full action space.
        """
        if self.phase == 1 and len(action_list) >= 4:
            if self._scripted_move_hold <= 0:
                self._scripted_move_target = random.randint(0, 7)
                self._scripted_move_hold = random.randint(8, 20)
            self._scripted_move_hold -= 1
            action_list[0] = float(self._scripted_move_target)
            action_list[3] = 0.0   # weapon locked to rifle
        elif self.phase == 2 and len(action_list) >= 4:
            action_list[3] = 0.0   # weapon locked
        return action_list

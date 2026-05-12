"""
Tactical Gymnasium wrapper for GhostLobby top-down tactical scenarios.

Action heads:
  [0] move_target: discrete 0-11 (12 cardinal + diagonal directions)
  [1] aim_delta:   continuous [-1,1] (pass-through)
  [2] shoot:       binary (threshold at 0)
  [3] weapon_select: binary (threshold at 0)

Supports opponent for self-play and curriculum phases.
"""

import random

from glgym.gym_base import BaseGhostLobbyGym


class TacticalGym(BaseGhostLobbyGym):
    """Top-down tactical scenario with 12-direction discrete movement.

    Uses the same action remapping as SelfPlayGym: move_target is
    discretised from continuous [-1,1] to [0,11], aim passes through,
    shoot and weapon_select are thresholded to binary.

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

    def _remap_actions(self, action_list: list[float]) -> list[float]:
        """Remap continuous [-1,1] to tactical engine format.

          [0] move_target:    [-1,1] -> [0,11] (12 discrete directions)
          [1] aim_delta:      [-1,1] -> [-1,1] (pass-through)
          [2] shoot:          [-1,1] -> 0 or 1
          [3] weapon_select:  [-1,1] -> 0 or 1
        """
        if len(action_list) >= 4:
            action_list[0] = (action_list[0] + 1.0) * 5.5   # [-1,1] -> [0,11]
            action_list[2] = 1.0 if action_list[2] > 0.0 else 0.0
            action_list[3] = 1.0 if action_list[3] > 0.0 else 0.0
        return action_list

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

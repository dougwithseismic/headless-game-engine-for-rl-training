"""
CS-Lite Gymnasium wrapper for GhostLobby 5v5 tactical shooter.

Action heads (MultiDiscrete):
  [0] move_target:   Discrete(12)
  [1] shoot:         Discrete(2)
  [2] reload:        Discrete(2)
  [3] use_action:    Discrete(3)

Supports shared-policy multi-agent (one network, 5 instances per team),
self-play (T vs CT), and curriculum phases.
"""

import random

from glgym.gym_base import BaseGhostLobbyGym


class CsLiteGym(BaseGhostLobbyGym):
    """5v5 tactical shooter with bomb plant/defuse, A* navigation, and cover.

    Each gym step controls a single agent (agent_id) on one team. The
    scripted AI controls the opponent team and optionally teammates.
    For self-play, set an opponent policy to replace the scripted AI.

    Curriculum phases:
      Phase 1: Scripted movement (random candidate, held 8-20 steps).
               Agent learns shoot only.
      Phase 2+: No masking (move + shoot).

    Args:
        config_path: Path to a CS-Lite JSON config.
        scenario: Scenario name, defaults to "cs_lite".
        frame_skip: Engine ticks per gym step.
        max_steps: Max gym steps before truncation.
        phase: Curriculum phase. None means no masking.
        control_team: Which team the RL agent controls (0=T, 1=CT, None=random).
    """

    def __init__(
        self,
        config_path: str,
        scenario: str = "cs_lite",
        frame_skip: int = 4,
        max_steps: int = 2048,
        phase: int | None = None,
        control_team: int | None = None,
        track_behavior: bool = False,
    ):
        self.control_team = control_team
        super().__init__(
            config_path=config_path,
            scenario=scenario,
            frame_skip=frame_skip,
            max_steps=max_steps,
            phase=phase,
            track_behavior=track_behavior,
        )

    def _init_agent_ids(self) -> None:
        """Pick which agent to control each episode.

        Discovers team size from the env's agent count rather than
        hardcoding 5. Works for 1v1, 2v2, 5v5, etc.
        """
        if self.control_team is not None:
            team = self.control_team
        else:
            team = random.randint(0, 1)

        ppt = self.env.num_agents() // 2
        self.agent_id = team * ppt
        self.opp_id = (1 - team) * ppt

    def _apply_phase_mask(self, action_list: list[float]) -> list[float]:
        """Lock action heads based on curriculum phase.

        Phase 1: Scripted movement (random candidate, held 8-20 steps).
               Agent learns shoot/reload/use only.
        Phase 2+: No masking.
        """
        if self.phase is None or len(action_list) < 4:
            return action_list

        if self.phase == 1:
            if self._scripted_move_hold <= 0:
                self._scripted_move_target = random.randint(0, 11)
                self._scripted_move_hold = random.randint(8, 20)
            self._scripted_move_hold -= 1
            action_list[0] = float(self._scripted_move_target)

        return action_list

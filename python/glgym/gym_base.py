"""
Base Gymnasium wrapper for GhostLobby environments.

Extracts the shared logic from Arena3DGym, SelfPlayGym, SingleAgentGym,
and GhostLobbyGym into a single base class. Subclasses override:
  - _remap_actions()    to transform policy outputs before sending to engine
  - _apply_phase_mask() to lock action heads for curriculum learning
  - _init_agent_ids()   to set agent_id/opp_id on each reset
"""

import gymnasium as gym
import numpy as np
import ghostlobby as gl


class BaseGhostLobbyGym(gym.Env):
    """Base Gymnasium environment wrapping a GhostLobbyEnv.

    Handles environment lifecycle, observation flattening, frame skipping,
    telemetry draining, and episode bookkeeping. Subclasses customise
    action remapping, phase masking, and opponent management.

    Args:
        config_path: Path to a GhostLobby JSON config file.
        scenario: Scenario name string (e.g. "arena3d", "fps", "drone-hover").
        frame_skip: Number of engine ticks per gym step. Higher values give
            the agent more "think time" but coarser control.
        max_steps: Maximum gym steps before the episode is truncated.
        phase: Optional curriculum phase integer. Passed to _apply_phase_mask().
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path: str,
        scenario: str = "fps",
        frame_skip: int = 4,
        max_steps: int = 2048,
        phase: int | None = None,
        track_behavior: bool = False,
    ):
        super().__init__()
        self.config_path = config_path
        self.scenario = scenario
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.phase = phase
        self.track_behavior = track_behavior

        # Episode state
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_ticks = 0

        # Behavior tracking
        self._behavior = {
            "shots_fired": 0, "shots_hit": 0,
            "kills": 0, "deaths": 0,
            "damage_dealt": 0.0, "damage_taken": 0.0,
            "rounds_won": 0, "rounds_lost": 0,
        }
        self._my_entity_id: int | None = None

        # Agent identifiers -- subclasses may override in _init_agent_ids()
        self.agent_id: int = 0
        self.opp_id: int | None = None

        # Opponent policy (for self-play subclasses)
        self.opponent_policy = None
        self.opponent_obs: dict | None = None
        self.opponent_lstm_states = None

        # Telemetry sink -- append JSON strings here each tick
        self.telemetry_sink = None

        # Scripted movement state for phase masking
        self._scripted_move_target = 0
        self._scripted_move_hold = 0

        # Replay buffer -- preserved across resets so frames survive env recreation
        self._pending_replay: list[dict] = []
        self._recording = False

        # Create environment and discover spaces
        self.env = gl.GhostLobbyEnv(config_path, scenario=scenario)
        space_info = self.env.action_space()

        nvec = []
        for h in space_info["heads"]:
            if "Discrete" in h:
                nvec.append(h["Discrete"]["n"])
            elif "n" in h:
                nvec.append(h["n"])
        self.action_space = gym.spaces.MultiDiscrete(nvec)
        self.action_size: int = len(nvec)

        obs, _ = self.env.reset()
        sample_obs = self._flatten_obs(obs[self.agent_id])
        self.obs_size: int = len(sample_obs)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Extension points -- override in subclasses
    # ------------------------------------------------------------------

    def _init_agent_ids(self) -> None:
        """Set agent_id and opp_id for the upcoming episode.

        Called at the start of each reset(). Default: agent_id=0, no opponent.
        Subclasses with self-play should randomise sides here.
        """
        self.agent_id = 0
        self.opp_id = None

    def _remap_actions(self, action_list: list[float]) -> list[float]:
        """Transform policy outputs before sending to the engine.

        Default is identity (pass-through). With MultiDiscrete actions,
        integer indices are passed directly.
        """
        return action_list

    def _apply_phase_mask(self, action_list: list[float]) -> list[float]:
        """Apply curriculum phase masking to lock/override action heads.

        Default is identity (no masking). Override for scenarios with
        phased curriculum training.
        """
        return action_list

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _flatten_obs(self, agent_obs: dict) -> np.ndarray:
        """Flatten a dict of named observation arrays into a single vector.

        Keys are sorted alphabetically so the concatenation order is
        deterministic across Python versions and dict insertion order.
        """
        parts = []
        for key in sorted(agent_obs.keys()):
            parts.extend(agent_obs[key])
        return np.array(parts, dtype=np.float32)

    # ------------------------------------------------------------------
    # Opponent management
    # ------------------------------------------------------------------

    def set_opponent(self, policy) -> None:
        """Set or replace the frozen opponent policy for self-play.

        Args:
            policy: An SB3-compatible policy with a .predict() method.
        """
        self.opponent_policy = policy
        self.opponent_lstm_states = None

    def _get_opponent_action(self, obs_dict: dict) -> list[float] | None:
        """Query the opponent policy for an action.

        Returns None if no opponent policy is set, if there is no opponent
        agent, or if the opponent's observations are not in the dict.
        """
        if self.opponent_policy is None or self.opp_id is None:
            return None
        if self.opp_id not in obs_dict:
            return None

        opp_obs = self._flatten_obs(obs_dict[self.opp_id])
        episode_start = np.array([self.opponent_lstm_states is None])
        action, self.opponent_lstm_states = self.opponent_policy.predict(
            opp_obs,
            state=self.opponent_lstm_states,
            episode_start=episode_start,
            deterministic=False,
        )
        raw = action.tolist() if hasattr(action, "tolist") else list(action)
        return [float(a) for a in raw]

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode.

        Recreates the underlying GhostLobbyEnv (ensuring clean Bevy world),
        resets all episode counters, and returns the initial observation.
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_ticks = 0

        self._init_agent_ids()

        # Drain recorded frames before destroying the old env
        if self._recording and hasattr(self, "env"):
            try:
                self._pending_replay = self.env.replay_frames()
            except Exception:
                pass

        self.env = gl.GhostLobbyEnv(self.config_path, scenario=self.scenario)
        obs, info = self.env.reset()

        self.opponent_obs = obs
        self.opponent_lstm_states = None
        self._scripted_move_hold = 0

        if self.track_behavior:
            self._my_entity_id = self.env.agent_entity_id(self.agent_id)
            self._behavior = {k: 0 if isinstance(v, int) else 0.0 for k, v in self._behavior.items()}

        return self._flatten_obs(obs[self.agent_id]), {}

    def step(self, action):
        """Execute one gym step (frame_skip engine ticks).

        Applies action remapping and phase masking, queries the opponent
        policy if present, accumulates reward over frame_skip ticks, and
        returns the standard Gymnasium 5-tuple.
        """
        action_list = [float(a) for a in (action.tolist() if hasattr(action, "tolist") else list(action))]
        action_list = self._remap_actions(action_list)
        action_list = self._apply_phase_mask(action_list)

        total_reward = 0.0
        terminated = False
        truncated = False
        accumulated_breakdown: dict[str, float] = {}

        opp_action = self._get_opponent_action(self.opponent_obs)

        for _ in range(self.frame_skip):
            actions = {self.agent_id: action_list}
            if opp_action is not None and self.opp_id is not None:
                actions[self.opp_id] = opp_action

            obs, rewards, term, trunc, infos = self.env.step(actions)

            # Drain telemetry
            telemetry = self.env.drain_telemetry()
            if self.telemetry_sink is not None:
                for event_json in telemetry:
                    self.telemetry_sink.append(event_json)

            if self.track_behavior and self._my_entity_id is not None:
                for event_json in telemetry:
                    self._parse_behavior_event(event_json)

            if self.track_behavior:
                try:
                    bd = self.env.reward_breakdown()
                    if self.agent_id in bd:
                        for k, v in bd[self.agent_id].items():
                            accumulated_breakdown[k] = accumulated_breakdown.get(k, 0.0) + v
                except Exception:
                    pass

            total_reward += rewards.get(self.agent_id, 0.0)
            self.episode_ticks += 1
            terminated = term.get(self.agent_id, False)
            if terminated:
                break

        self.opponent_obs = obs
        self.current_step += 1
        self.episode_reward += total_reward

        if self.current_step >= self.max_steps:
            truncated = True

        flat_obs = self._flatten_obs(obs[self.agent_id])

        info = {}
        if terminated or truncated:
            info["episode_reward"] = self.episode_reward
            info["episode_ticks"] = self.episode_ticks
            info["episode_steps"] = self.current_step
            if self.track_behavior:
                info["behavior"] = dict(self._behavior)
            if self.track_behavior and accumulated_breakdown:
                info["reward_breakdown"] = accumulated_breakdown

        return flat_obs, total_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Behavior tracking
    # ------------------------------------------------------------------

    def _parse_behavior_event(self, event_json: str) -> None:
        """Parse a telemetry event JSON string and update behavior counters."""
        import json
        try:
            evt = json.loads(event_json)
        except (json.JSONDecodeError, TypeError):
            return

        eid = self._my_entity_id
        etype = evt.get("type", "")

        if etype == "ShotFired":
            if evt.get("shooter") == eid:
                self._behavior["shots_fired"] += 1
                if evt.get("hit_target") is not None:
                    self._behavior["shots_hit"] += 1
        elif etype == "Kill":
            if evt.get("killer") == eid:
                self._behavior["kills"] += 1
            if evt.get("victim") == eid:
                self._behavior["deaths"] += 1
        elif etype == "Damage":
            if evt.get("source") == eid:
                self._behavior["damage_dealt"] += evt.get("amount", 0.0)
            if evt.get("target") == eid:
                self._behavior["damage_taken"] += evt.get("amount", 0.0)
        elif etype == "CsLiteRoundState":
            if evt.get("phase") == "round_end":
                my_team = self.agent_id // max(1, self.env.num_agents() // 2)
                t_score = evt.get("t_score", 0)
                ct_score = evt.get("ct_score", 0)
                my_score = t_score if my_team == 0 else ct_score
                opp_score = ct_score if my_team == 0 else t_score
                if not hasattr(self, "_last_my_score"):
                    self._last_my_score = 0
                    self._last_opp_score = 0
                if my_score > self._last_my_score:
                    self._behavior["rounds_won"] += 1
                if opp_score > self._last_opp_score:
                    self._behavior["rounds_lost"] += 1
                self._last_my_score = my_score
                self._last_opp_score = opp_score

    # ------------------------------------------------------------------
    # Recording wrappers
    # ------------------------------------------------------------------

    def start_recording(self):
        self._recording = True
        self._pending_replay = []
        self.env.start_recording()

    def stop_recording(self):
        self._recording = False
        try:
            self.env.stop_recording()
        except Exception:
            pass

    def save_replay(self, path: str):
        import json, os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        frames = self._pending_replay
        if not frames:
            try:
                frames = self.env.replay_frames()
            except Exception:
                frames = []
        with open(path, "w") as f:
            for frame in frames:
                f.write(json.dumps(frame) + "\n")
        self._pending_replay = []
        self._recording = False

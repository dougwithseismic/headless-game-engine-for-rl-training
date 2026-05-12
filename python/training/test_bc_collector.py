"""Tests for the BC (Behavioral Cloning) demo collector.

Covers the collect_demonstrations() and collect_demonstrations_native()
functions. Tests are pure-Python (no engine needed) for the gym-based
collector, and require the built Rust module for native integration tests.
"""

import os

import numpy as np
import pytest

from training.bc_collector import collect_demonstrations


# ---------------------------------------------------------------------------
# Mock environment for testing collect_demonstrations
# ---------------------------------------------------------------------------

OBS_SIZE = 64  # arbitrary size for mock environment


def _make_obs() -> np.ndarray:
    """Build a dummy obs vector."""
    return np.random.randn(OBS_SIZE).astype(np.float32)


class _MockGym:
    """Minimal mock gym for testing without the Rust engine."""

    def __init__(self, **kwargs):
        self._step_count = 0
        self._max_steps = kwargs.get("max_steps", 10)

    def reset(self, seed=None, options=None):
        self._step_count = 0
        return _make_obs(), {}

    def step(self, action):
        self._step_count += 1
        obs = _make_obs()
        reward = 0.01
        terminated = False
        truncated = self._step_count >= self._max_steps
        return obs, reward, terminated, truncated, {}

    def close(self):
        pass


def _make_mock_env(**kwargs):
    """Factory for mock environments."""
    return _MockGym(**kwargs)


class _ConstantExpert:
    """Expert that always returns 4D actions of 0.42."""

    def predict(self, obs):
        return np.full(4, 0.42, dtype=np.float32)


# ---------------------------------------------------------------------------
# Tests: collect_demonstrations (gym-based)
# ---------------------------------------------------------------------------

class TestCollectDemonstrations:
    """Tests for the collect_demonstrations function.

    These tests mock the gym environment to avoid requiring the Rust engine.
    """

    def test_output_file_created(self, tmp_path):
        """collect_demonstrations should create the output .npz file."""
        output = str(tmp_path / "demos.npz")
        collect_demonstrations(
            config_path="dummy",
            scenario="tactical",
            num_episodes=2,
            max_steps_per_episode=5,
            output_path=output,
            _env_factory=_make_mock_env,
            expert_policy=_ConstantExpert(),
        )
        assert (tmp_path / "demos.npz").exists()

    def test_output_shapes(self, tmp_path):
        """Observations and actions arrays should have matching lengths."""
        output = str(tmp_path / "demos.npz")
        collect_demonstrations(
            config_path="dummy",
            scenario="tactical",
            num_episodes=3,
            max_steps_per_episode=10,
            output_path=output,
            _env_factory=_make_mock_env,
            expert_policy=_ConstantExpert(),
        )
        data = np.load(output)
        assert "observations" in data
        assert "actions" in data
        assert "episode_starts" in data
        assert "rewards" in data
        assert data["observations"].shape[0] == data["actions"].shape[0]
        assert data["observations"].shape[0] == data["episode_starts"].shape[0]
        assert data["observations"].shape[0] == data["rewards"].shape[0]

    def test_episode_starts_correctness(self, tmp_path):
        """Episode_starts should be True only at the first step of each episode."""
        output = str(tmp_path / "demos.npz")
        num_episodes = 3
        max_steps = 4
        collect_demonstrations(
            config_path="dummy",
            scenario="tactical",
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps,
            output_path=output,
            _env_factory=_make_mock_env,
            expert_policy=_ConstantExpert(),
        )
        data = np.load(output)
        starts = data["episode_starts"]
        assert starts.sum() == num_episodes

    def test_observations_dtype(self, tmp_path):
        """Observations and actions should be float32."""
        output = str(tmp_path / "demos.npz")
        collect_demonstrations(
            config_path="dummy",
            scenario="tactical",
            num_episodes=2,
            max_steps_per_episode=5,
            output_path=output,
            _env_factory=_make_mock_env,
            expert_policy=_ConstantExpert(),
        )
        data = np.load(output)
        assert data["observations"].dtype == np.float32
        assert data["actions"].dtype == np.float32
        assert data["episode_starts"].dtype == bool

    def test_expert_policy_override(self, tmp_path):
        """A custom expert policy should be usable via the expert_policy param."""
        output = str(tmp_path / "demos.npz")
        collect_demonstrations(
            config_path="dummy",
            scenario="tactical",
            num_episodes=2,
            max_steps_per_episode=5,
            output_path=output,
            _env_factory=_make_mock_env,
            expert_policy=_ConstantExpert(),
        )
        data = np.load(output)
        np.testing.assert_allclose(data["actions"], 0.42, atol=1e-6)

    def test_output_dir_created(self, tmp_path):
        """Should create intermediate directories for the output path."""
        output = str(tmp_path / "subdir" / "nested" / "demos.npz")
        collect_demonstrations(
            config_path="dummy",
            scenario="tactical",
            num_episodes=1,
            max_steps_per_episode=3,
            output_path=output,
            _env_factory=_make_mock_env,
            expert_policy=_ConstantExpert(),
        )
        assert (tmp_path / "subdir" / "nested" / "demos.npz").exists()


# ---------------------------------------------------------------------------
# Native demo collector tests (requires the compiled Rust module)
# ---------------------------------------------------------------------------

try:
    import ghostlobby
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

from training.bc_collector import collect_demonstrations_native


@pytest.mark.skipif(not HAS_ENGINE, reason="Rust ghostlobby module not built")
class TestCollectDemonstrationsNative:
    """Integration tests for collect_demonstrations_native().

    These require the compiled Rust ghostlobby module and a valid cs_lite
    config with scripted_ai enabled.
    """

    _CS_LITE_CONFIG = os.path.join(_PROJECT_ROOT, "configs", "cs_lite", "cs_lite.json")

    def test_output_file_created(self, tmp_path):
        """The .npz output file should be created on disk."""
        output = str(tmp_path / "demos.npz")
        collect_demonstrations_native(
            config_path=self._CS_LITE_CONFIG,
            scenario="cs_lite",
            num_episodes=2,
            output_path=output,
            max_ticks_per_episode=100,
        )
        assert (tmp_path / "demos.npz").exists(), "output .npz file was not created"

    def test_output_shapes(self, tmp_path):
        """All arrays should have matching first dimension and correct keys."""
        output = str(tmp_path / "demos.npz")
        collect_demonstrations_native(
            config_path=self._CS_LITE_CONFIG,
            scenario="cs_lite",
            num_episodes=2,
            output_path=output,
            max_ticks_per_episode=100,
        )
        data = np.load(output)

        assert "observations" in data, "missing 'observations' key"
        assert "actions" in data, "missing 'actions' key"
        assert "episode_starts" in data, "missing 'episode_starts' key"
        assert "rewards" in data, "missing 'rewards' key"

        n = data["observations"].shape[0]
        assert n > 0, "should have at least 1 transition"
        assert data["actions"].shape[0] == n, "actions length mismatch"
        assert data["episode_starts"].shape[0] == n, "episode_starts length mismatch"
        assert data["rewards"].shape[0] == n, "rewards length mismatch"

        assert data["observations"].dtype == np.float32
        assert data["actions"].dtype == np.float32
        assert data["episode_starts"].dtype == bool
        assert data["rewards"].dtype == np.float32

    def test_actions_finite(self, tmp_path):
        """All recorded actions should be finite (no NaN or Inf)."""
        output = str(tmp_path / "demos.npz")
        collect_demonstrations_native(
            config_path=self._CS_LITE_CONFIG,
            scenario="cs_lite",
            num_episodes=2,
            output_path=output,
            max_ticks_per_episode=100,
        )
        data = np.load(output)
        assert np.all(np.isfinite(data["actions"])), "actions contain NaN or Inf"

    def test_episode_starts_count(self, tmp_path):
        """episode_starts should have exactly num_episodes True values."""
        num_episodes = 3
        output = str(tmp_path / "demos.npz")
        collect_demonstrations_native(
            config_path=self._CS_LITE_CONFIG,
            scenario="cs_lite",
            num_episodes=num_episodes,
            output_path=output,
            max_ticks_per_episode=100,
        )
        data = np.load(output)
        assert data["episode_starts"].sum() == num_episodes, (
            f"expected {num_episodes} episode starts, got {data['episode_starts'].sum()}"
        )

    def test_returns_stats_dict(self, tmp_path):
        """The return value should be a dict with expected summary keys."""
        num_episodes = 2
        output = str(tmp_path / "demos.npz")
        stats = collect_demonstrations_native(
            config_path=self._CS_LITE_CONFIG,
            scenario="cs_lite",
            num_episodes=num_episodes,
            output_path=output,
            max_ticks_per_episode=100,
        )

        assert isinstance(stats, dict), f"expected dict, got {type(stats)}"
        assert "num_transitions" in stats, "missing 'num_transitions' key"
        assert "num_episodes" in stats, "missing 'num_episodes' key"
        assert "mean_reward" in stats, "missing 'mean_reward' key"

        assert stats["num_transitions"] > 0, "should have at least 1 transition"
        assert stats["num_episodes"] == num_episodes, (
            f"expected {num_episodes} episodes, got {stats['num_episodes']}"
        )

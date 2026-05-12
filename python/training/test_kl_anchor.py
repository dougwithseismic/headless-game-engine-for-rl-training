"""Tests for the KLAnchorCallback.

Tests the KL anchor callback that penalizes PPO policy divergence from a
frozen BC reference model. Uses mocks and synthetic data so tests run
without the Rust engine or real SB3 training loops.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from training.bc_pretrain import BCTrainer

# Import after bc_pretrain to ensure the module is available
from training.callbacks import KLAnchorCallback


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OBS_DIM = 16
ACT_DIM = 4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bc_ref_path(tmp_path):
    """Create a saved BC reference model and return its path."""
    trainer = BCTrainer(obs_dim=OBS_DIM, act_dim=ACT_DIM)
    path = str(tmp_path / "bc_ref.pt")
    trainer.save_reference(path)
    return path


def _make_mock_rollout_buffer(n_steps: int, n_envs: int, obs_dim: int):
    """Create a mock rollout buffer with realistic shapes.

    SB3's RolloutBuffer stores:
      - observations: (n_steps, n_envs, obs_dim)  -- for vectorized envs
      - rewards: (n_steps, n_envs)
    """
    buf = MagicMock()
    buf.observations = np.random.randn(n_steps, n_envs, obs_dim).astype(
        np.float32
    )
    buf.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
    return buf


def _make_mock_distribution(actions_tensor):
    """Create a mock SB3 distribution wrapping a Normal with given mean."""
    inner_dist = MagicMock()
    inner_dist.mean = actions_tensor

    outer_dist = MagicMock()
    outer_dist.distribution = inner_dist
    return outer_dist


def _make_mock_model(n_steps: int, n_envs: int, obs_dim: int, act_dim: int):
    """Create a mock SB3 PPO model with a rollout buffer, policy, and logger.

    The mock policy's ``get_distribution()`` returns a mock Normal distribution
    whose ``.distribution.mean`` yields random actions of the correct shape.
    This mirrors SB3's ``DiagGaussianDistribution`` for continuous action spaces.

    Includes a ``_MockLogger`` on ``model.logger`` so that the callback's
    ``self.logger`` property (which delegates to ``self.model.logger``) works.
    """
    model = MagicMock()
    model.rollout_buffer = _make_mock_rollout_buffer(n_steps, n_envs, obs_dim)
    model.logger = _MockLogger()

    def fake_get_distribution(obs_tensor):
        """Return a mock distribution with random mean actions."""
        batch_size = obs_tensor.shape[0]
        actions = torch.randn(batch_size, act_dim)
        return _make_mock_distribution(actions)

    model.policy = MagicMock()
    model.policy.get_distribution = fake_get_distribution
    model.policy.device = torch.device("cpu")

    return model


class _MockLogger:
    """Mock logger that records calls to record().

    SB3's BaseCallback.logger is a property that returns self.model.logger,
    so this must be set on the mock model, not the callback directly.
    """

    def __init__(self):
        self.recorded: dict = {}

    def record(self, key: str, value) -> None:
        self.recorded[key] = value


# ---------------------------------------------------------------------------
# TestKLAnchorCallbackInit
# ---------------------------------------------------------------------------


class TestKLAnchorCallbackInit:
    """Verify constructor stores parameters correctly."""

    def test_stores_bc_ref_path(self):
        cb = KLAnchorCallback(bc_ref_path="/tmp/ref.pt")
        assert cb.bc_ref_path == "/tmp/ref.pt"

    def test_default_beta_start(self):
        cb = KLAnchorCallback(bc_ref_path="/tmp/ref.pt")
        assert cb.beta_start == 0.5

    def test_default_beta_end(self):
        cb = KLAnchorCallback(bc_ref_path="/tmp/ref.pt")
        assert cb.beta_end == 0.0

    def test_default_anneal_steps(self):
        cb = KLAnchorCallback(bc_ref_path="/tmp/ref.pt")
        assert cb.anneal_steps == 2_000_000

    def test_custom_params(self):
        cb = KLAnchorCallback(
            bc_ref_path="/tmp/ref.pt",
            beta_start=1.0,
            beta_end=0.1,
            anneal_steps=500_000,
        )
        assert cb.beta_start == 1.0
        assert cb.beta_end == 0.1
        assert cb.anneal_steps == 500_000

    def test_bc_model_not_loaded_at_init(self):
        """BC model should not be loaded until _on_training_start."""
        cb = KLAnchorCallback(bc_ref_path="/tmp/nonexistent.pt")
        assert cb.bc_model is None

    def test_inherits_base_callback(self):
        from stable_baselines3.common.callbacks import BaseCallback

        cb = KLAnchorCallback(bc_ref_path="/tmp/ref.pt")
        assert isinstance(cb, BaseCallback)


# ---------------------------------------------------------------------------
# TestKLAnchorBetaAnnealing
# ---------------------------------------------------------------------------


class TestKLAnchorBetaAnnealing:
    """Test linear beta annealing schedule."""

    def test_beta_at_start(self):
        cb = KLAnchorCallback(
            bc_ref_path="/tmp/ref.pt",
            beta_start=0.5,
            beta_end=0.0,
            anneal_steps=1_000_000,
        )
        cb.num_timesteps = 0
        assert cb._current_beta() == pytest.approx(0.5)

    def test_beta_at_midpoint(self):
        cb = KLAnchorCallback(
            bc_ref_path="/tmp/ref.pt",
            beta_start=0.5,
            beta_end=0.0,
            anneal_steps=1_000_000,
        )
        cb.num_timesteps = 500_000
        assert cb._current_beta() == pytest.approx(0.25)

    def test_beta_at_end(self):
        cb = KLAnchorCallback(
            bc_ref_path="/tmp/ref.pt",
            beta_start=0.5,
            beta_end=0.0,
            anneal_steps=1_000_000,
        )
        cb.num_timesteps = 1_000_000
        assert cb._current_beta() == pytest.approx(0.0)

    def test_beta_clamps_beyond_anneal_steps(self):
        """Beta should not go below beta_end after anneal_steps."""
        cb = KLAnchorCallback(
            bc_ref_path="/tmp/ref.pt",
            beta_start=0.5,
            beta_end=0.0,
            anneal_steps=1_000_000,
        )
        cb.num_timesteps = 2_000_000  # well past anneal_steps
        assert cb._current_beta() == pytest.approx(0.0)

    def test_beta_quarter_point(self):
        cb = KLAnchorCallback(
            bc_ref_path="/tmp/ref.pt",
            beta_start=1.0,
            beta_end=0.0,
            anneal_steps=100,
        )
        cb.num_timesteps = 25
        assert cb._current_beta() == pytest.approx(0.75)

    def test_beta_non_zero_end(self):
        """Annealing to a non-zero beta_end."""
        cb = KLAnchorCallback(
            bc_ref_path="/tmp/ref.pt",
            beta_start=1.0,
            beta_end=0.2,
            anneal_steps=1000,
        )
        cb.num_timesteps = 1000
        assert cb._current_beta() == pytest.approx(0.2)

    def test_beta_non_zero_end_midpoint(self):
        cb = KLAnchorCallback(
            bc_ref_path="/tmp/ref.pt",
            beta_start=1.0,
            beta_end=0.2,
            anneal_steps=1000,
        )
        cb.num_timesteps = 500
        # midpoint: 1.0 + (0.2 - 1.0) * 0.5 = 1.0 - 0.4 = 0.6
        assert cb._current_beta() == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# TestKLAnchorTrainingStart
# ---------------------------------------------------------------------------


class TestKLAnchorTrainingStart:
    """Test that _on_training_start loads and freezes the BC model."""

    def test_loads_bc_model(self, bc_ref_path):
        cb = KLAnchorCallback(bc_ref_path=bc_ref_path)
        cb.model = MagicMock()  # SB3 sets this before _on_training_start
        cb._on_training_start()
        assert cb.bc_model is not None
        assert isinstance(cb.bc_model, BCTrainer)

    def test_bc_model_in_eval_mode(self, bc_ref_path):
        """BC model's networks should be in eval mode (frozen)."""
        cb = KLAnchorCallback(bc_ref_path=bc_ref_path)
        cb.model = MagicMock()
        cb._on_training_start()
        assert not cb.bc_model.feature_extractor.training
        assert not cb.bc_model.action_head.training

    def test_bc_model_preserves_dims(self, bc_ref_path):
        cb = KLAnchorCallback(bc_ref_path=bc_ref_path)
        cb.model = MagicMock()
        cb._on_training_start()
        assert cb.bc_model.obs_dim == OBS_DIM
        assert cb.bc_model.act_dim == ACT_DIM


# ---------------------------------------------------------------------------
# TestKLAnchorRolloutEnd
# ---------------------------------------------------------------------------


class TestKLAnchorRolloutEnd:
    """Test _on_rollout_end reward modification and logging."""

    @pytest.fixture
    def setup_callback(self, bc_ref_path):
        """Create a fully-initialized KLAnchorCallback with mocked model."""
        n_steps, n_envs = 32, 2
        cb = KLAnchorCallback(
            bc_ref_path=bc_ref_path,
            beta_start=0.5,
            beta_end=0.0,
            anneal_steps=1_000_000,
        )
        cb.model = _make_mock_model(n_steps, n_envs, OBS_DIM, ACT_DIM)

        cb.num_timesteps = 100_000  # some midpoint
        cb._on_training_start()
        return cb, n_steps, n_envs

    def test_returns_true(self, setup_callback):
        cb, _, _ = setup_callback
        result = cb._on_rollout_end()
        assert result is True

    def test_modifies_rewards(self, setup_callback):
        """Rewards should be reduced by the penalty."""
        cb, n_steps, n_envs = setup_callback
        original_rewards = cb.model.rollout_buffer.rewards.copy()
        cb._on_rollout_end()
        modified_rewards = cb.model.rollout_buffer.rewards

        # Rewards should have been decreased (penalty subtracted)
        # Since original rewards are zero, modified should be negative
        assert np.all(modified_rewards <= 0.0), (
            "Penalties should make zero rewards negative"
        )

    def test_penalty_has_correct_shape(self, setup_callback):
        """After _on_rollout_end, rewards should still have (n_steps, n_envs) shape."""
        cb, n_steps, n_envs = setup_callback
        cb._on_rollout_end()
        assert cb.model.rollout_buffer.rewards.shape == (n_steps, n_envs)

    def test_logs_mean_divergence(self, setup_callback):
        cb, _, _ = setup_callback
        cb._on_rollout_end()
        assert "kl_anchor/mean_divergence" in cb.logger.recorded
        assert isinstance(cb.logger.recorded["kl_anchor/mean_divergence"], float)

    def test_logs_beta(self, setup_callback):
        cb, _, _ = setup_callback
        cb._on_rollout_end()
        assert "kl_anchor/beta" in cb.logger.recorded
        assert isinstance(cb.logger.recorded["kl_anchor/beta"], float)

    def test_logs_penalty(self, setup_callback):
        cb, _, _ = setup_callback
        cb._on_rollout_end()
        assert "kl_anchor/penalty" in cb.logger.recorded
        assert isinstance(cb.logger.recorded["kl_anchor/penalty"], float)

    def test_logged_beta_matches_schedule(self, setup_callback):
        """Logged beta should match the annealing schedule at current timestep."""
        cb, _, _ = setup_callback
        expected_beta = cb._current_beta()
        cb._on_rollout_end()
        assert cb.logger.recorded["kl_anchor/beta"] == pytest.approx(
            expected_beta
        )

    def test_logged_penalty_equals_beta_times_divergence(self, setup_callback):
        cb, _, _ = setup_callback
        cb._on_rollout_end()
        beta = cb.logger.recorded["kl_anchor/beta"]
        div = cb.logger.recorded["kl_anchor/mean_divergence"]
        penalty = cb.logger.recorded["kl_anchor/penalty"]
        assert penalty == pytest.approx(beta * div, abs=1e-6)

    def test_divergence_is_non_negative(self, setup_callback):
        """MSE divergence should always be >= 0."""
        cb, _, _ = setup_callback
        cb._on_rollout_end()
        assert cb.logger.recorded["kl_anchor/mean_divergence"] >= 0.0


# ---------------------------------------------------------------------------
# TestKLAnchorBetaZeroShortCircuit
# ---------------------------------------------------------------------------


class TestKLAnchorBetaZeroShortCircuit:
    """When beta is effectively zero, rewards should not be modified."""

    def test_skips_when_beta_near_zero(self, bc_ref_path):
        cb = KLAnchorCallback(
            bc_ref_path=bc_ref_path,
            beta_start=0.5,
            beta_end=0.0,
            anneal_steps=100,
        )
        n_steps, n_envs = 16, 1
        cb.model = _make_mock_model(n_steps, n_envs, OBS_DIM, ACT_DIM)

        cb.num_timesteps = 1_000_000  # way past anneal
        cb._on_training_start()

        original_rewards = cb.model.rollout_buffer.rewards.copy()
        result = cb._on_rollout_end()

        assert result is True
        np.testing.assert_array_equal(
            cb.model.rollout_buffer.rewards,
            original_rewards,
            err_msg="Rewards should not change when beta is ~0",
        )


# ---------------------------------------------------------------------------
# TestKLAnchorSingleEnv
# ---------------------------------------------------------------------------


class TestKLAnchorSingleEnv:
    """Test with a single environment (n_envs=1)."""

    def test_works_with_single_env(self, bc_ref_path):
        n_steps, n_envs = 64, 1
        cb = KLAnchorCallback(
            bc_ref_path=bc_ref_path,
            beta_start=0.5,
            beta_end=0.0,
            anneal_steps=1_000_000,
        )
        cb.model = _make_mock_model(n_steps, n_envs, OBS_DIM, ACT_DIM)

        cb.num_timesteps = 50_000
        cb._on_training_start()

        result = cb._on_rollout_end()
        assert result is True
        assert cb.model.rollout_buffer.rewards.shape == (n_steps, n_envs)


# ---------------------------------------------------------------------------
# TestKLAnchorOnStep
# ---------------------------------------------------------------------------


class TestKLAnchorOnStep:
    """_on_step should always return True (no-op)."""

    def test_on_step_returns_true(self):
        cb = KLAnchorCallback(bc_ref_path="/tmp/ref.pt")
        assert cb._on_step() is True


# ---------------------------------------------------------------------------
# TestKLAnchorPenaltyMagnitude
# ---------------------------------------------------------------------------


class TestKLAnchorPenaltyMagnitude:
    """Test that the penalty magnitude scales correctly with beta."""

    def test_higher_beta_means_larger_penalty(self, bc_ref_path):
        """Same divergence should produce larger penalty with higher beta."""
        n_steps, n_envs = 32, 2

        # Run with beta_start=1.0 (at timestep 0)
        cb_high = KLAnchorCallback(
            bc_ref_path=bc_ref_path,
            beta_start=1.0,
            beta_end=1.0,
            anneal_steps=1,
        )
        cb_high.model = _make_mock_model(n_steps, n_envs, OBS_DIM, ACT_DIM)
        cb_high.num_timesteps = 0
        cb_high._on_training_start()

        # Run with beta_start=0.1 (at timestep 0)
        cb_low = KLAnchorCallback(
            bc_ref_path=bc_ref_path,
            beta_start=0.1,
            beta_end=0.1,
            anneal_steps=1,
        )
        cb_low.model = _make_mock_model(n_steps, n_envs, OBS_DIM, ACT_DIM)
        cb_low.num_timesteps = 0
        cb_low._on_training_start()

        # Use identical observations for both
        shared_obs = np.random.randn(n_steps, n_envs, OBS_DIM).astype(
            np.float32
        )
        cb_high.model.rollout_buffer.observations = shared_obs.copy()
        cb_high.model.rollout_buffer.rewards = np.zeros(
            (n_steps, n_envs), dtype=np.float32
        )
        cb_low.model.rollout_buffer.observations = shared_obs.copy()
        cb_low.model.rollout_buffer.rewards = np.zeros(
            (n_steps, n_envs), dtype=np.float32
        )

        cb_high._on_rollout_end()
        cb_low._on_rollout_end()

        penalty_high = cb_high.logger.recorded["kl_anchor/penalty"]
        penalty_low = cb_low.logger.recorded["kl_anchor/penalty"]

        # Higher beta should yield higher penalty (same divergence assumed)
        # Note: divergence may differ slightly due to random policy mock,
        # so we compare the logged beta * divergence values
        assert penalty_high > penalty_low or (
            penalty_high == pytest.approx(0.0) and
            penalty_low == pytest.approx(0.0)
        )


# ---------------------------------------------------------------------------
# TestKLAnchorZeroDivergence
# ---------------------------------------------------------------------------


class TestKLAnchorZeroDivergence:
    """When policy exactly matches BC, divergence should be zero."""

    def test_identical_policy_zero_divergence(self, bc_ref_path):
        """If the PPO policy returns the same actions as BC, penalty is zero."""
        n_steps, n_envs = 16, 1
        cb = KLAnchorCallback(
            bc_ref_path=bc_ref_path,
            beta_start=0.5,
            beta_end=0.0,
            anneal_steps=1_000_000,
        )
        cb.model = _make_mock_model(n_steps, n_envs, OBS_DIM, ACT_DIM)

        cb.num_timesteps = 0
        cb._on_training_start()

        # Make the mock policy return the exact same actions as BC
        obs = cb.model.rollout_buffer.observations
        obs_flat = obs.reshape(-1, OBS_DIM)
        bc_actions_np = cb.bc_model.predict(obs_flat)
        bc_actions_t = torch.tensor(bc_actions_np, dtype=torch.float32)

        def fake_get_distribution_matching(obs_tensor):
            return _make_mock_distribution(bc_actions_t)

        cb.model.policy.get_distribution = fake_get_distribution_matching
        cb.model.rollout_buffer.rewards = np.zeros(
            (n_steps, n_envs), dtype=np.float32
        )

        cb._on_rollout_end()

        assert cb.logger.recorded["kl_anchor/mean_divergence"] == pytest.approx(
            0.0, abs=1e-6
        )
        np.testing.assert_allclose(
            cb.model.rollout_buffer.rewards,
            np.zeros((n_steps, n_envs)),
            atol=1e-6,
            err_msg="Zero divergence should produce zero penalty",
        )

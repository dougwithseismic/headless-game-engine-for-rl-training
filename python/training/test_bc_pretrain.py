"""Tests for the BC (Behavioral Cloning) pre-trainer.

Tests the BCTrainer class end-to-end: network construction, training on
synthetic data, prediction, reference model save/load round-trip, and
SB3-compatible model export.

All tests use synthetic data so they run without the Rust engine.
"""

import math
import os
import tempfile

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Synthetic demo helpers
# ---------------------------------------------------------------------------

OBS_DIM = 116
ACT_DIM = 5


def _make_synthetic_demos(
    n: int = 1000,
    obs_dim: int = OBS_DIM,
    act_dim: int = ACT_DIM,
    seed: int = 42,
) -> str:
    """Create a temporary .npz file with a learnable obs->action mapping.

    The mapping is: action = tanh(obs[:act_dim] * 0.1), so a network
    should be able to learn it and drive loss down.
    """
    rng = np.random.RandomState(seed)
    obs = rng.randn(n, obs_dim).astype(np.float32)
    acts = np.tanh(obs[:, :act_dim] * 0.1).astype(np.float32)

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "demos.npz")
    np.savez(
        path,
        observations=obs,
        actions=acts,
        episode_starts=np.zeros(n, dtype=bool),
        rewards=np.zeros(n, dtype=np.float32),
    )
    return path


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from training.bc_pretrain import BCTrainer


# ---------------------------------------------------------------------------
# TestBCTrainerInit
# ---------------------------------------------------------------------------

class TestBCTrainerInit:
    """Verify the BCTrainer initialises with correct architecture."""

    def test_default_hidden_sizes(self):
        trainer = BCTrainer(obs_dim=OBS_DIM, act_dim=ACT_DIM)
        # feature_extractor should have 4 children: Linear, Tanh, Linear, Tanh
        children = list(trainer.feature_extractor.children())
        assert len(children) == 4
        assert isinstance(children[0], torch.nn.Linear)
        assert isinstance(children[1], torch.nn.Tanh)
        assert isinstance(children[2], torch.nn.Linear)
        assert isinstance(children[3], torch.nn.Tanh)

    def test_default_layer_sizes(self):
        trainer = BCTrainer(obs_dim=OBS_DIM, act_dim=ACT_DIM)
        children = list(trainer.feature_extractor.children())
        assert children[0].in_features == OBS_DIM
        assert children[0].out_features == 64
        assert children[2].in_features == 64
        assert children[2].out_features == 64

    def test_action_head_shape(self):
        trainer = BCTrainer(obs_dim=OBS_DIM, act_dim=ACT_DIM)
        assert trainer.action_head.in_features == 64
        assert trainer.action_head.out_features == ACT_DIM

    def test_custom_hidden_sizes(self):
        trainer = BCTrainer(obs_dim=10, act_dim=3, hidden_sizes=(32, 16))
        children = list(trainer.feature_extractor.children())
        assert len(children) == 4
        assert children[0].in_features == 10
        assert children[0].out_features == 32
        assert children[2].in_features == 32
        assert children[2].out_features == 16
        assert trainer.action_head.in_features == 16
        assert trainer.action_head.out_features == 3

    def test_three_hidden_layers(self):
        trainer = BCTrainer(obs_dim=20, act_dim=4, hidden_sizes=(128, 64, 32))
        children = list(trainer.feature_extractor.children())
        # 3 layers: Linear, Tanh, Linear, Tanh, Linear, Tanh
        assert len(children) == 6
        assert children[0].in_features == 20
        assert children[0].out_features == 128
        assert children[2].in_features == 128
        assert children[2].out_features == 64
        assert children[4].in_features == 64
        assert children[4].out_features == 32
        assert trainer.action_head.in_features == 32

    def test_stores_metadata(self):
        trainer = BCTrainer(obs_dim=OBS_DIM, act_dim=ACT_DIM, lr=3e-4)
        assert trainer.obs_dim == OBS_DIM
        assert trainer.act_dim == ACT_DIM
        assert trainer.lr == 3e-4

    def test_device_cpu_by_default(self):
        trainer = BCTrainer(obs_dim=OBS_DIM, act_dim=ACT_DIM)
        assert trainer.device == "cpu"


# ---------------------------------------------------------------------------
# TestBCTrainerTrain
# ---------------------------------------------------------------------------

class TestBCTrainerTrain:
    """Test the training loop with synthetic demonstrations."""

    @pytest.fixture
    def demo_path(self):
        return _make_synthetic_demos()

    def test_loss_decreases(self, demo_path):
        """Training loss should decrease over epochs on learnable data."""
        trainer = BCTrainer(obs_dim=OBS_DIM, act_dim=ACT_DIM, lr=1e-3)
        stats = trainer.train(demo_path, epochs=30, batch_size=64)
        # Loss should decrease meaningfully
        assert stats["train_losses"][-1] < stats["train_losses"][0], (
            f"Train loss did not decrease: "
            f"{stats['train_losses'][0]:.6f} -> {stats['train_losses'][-1]:.6f}"
        )

    def test_val_loss_decreases(self, demo_path):
        """Validation loss should also decrease on learnable data."""
        trainer = BCTrainer(obs_dim=OBS_DIM, act_dim=ACT_DIM, lr=1e-3)
        stats = trainer.train(demo_path, epochs=30, batch_size=64)
        assert stats["val_losses"][-1] < stats["val_losses"][0]

    def test_stats_structure(self, demo_path):
        """Returned stats dict should have the expected keys and lengths."""
        epochs = 10
        trainer = BCTrainer(obs_dim=OBS_DIM, act_dim=ACT_DIM)
        stats = trainer.train(demo_path, epochs=epochs, batch_size=64)
        assert "train_losses" in stats
        assert "val_losses" in stats
        assert "best_val_loss" in stats
        assert len(stats["train_losses"]) == epochs
        assert len(stats["val_losses"]) == epochs

    def test_best_val_loss_is_minimum(self, demo_path):
        """best_val_loss should be the minimum of val_losses."""
        trainer = BCTrainer(obs_dim=OBS_DIM, act_dim=ACT_DIM)
        stats = trainer.train(demo_path, epochs=15, batch_size=64)
        assert stats["best_val_loss"] == pytest.approx(
            min(stats["val_losses"]), abs=1e-8
        )

    def test_val_split_zero(self, demo_path):
        """With val_split=0, val_losses should still be tracked (empty val set)."""
        trainer = BCTrainer(obs_dim=OBS_DIM, act_dim=ACT_DIM)
        stats = trainer.train(demo_path, epochs=5, batch_size=64, val_split=0.0)
        assert len(stats["val_losses"]) == 5
        # val_loss with 0 batches should be 0.0 (epoch_loss / max(0, 1))
        # just check it doesn't crash

    def test_single_epoch(self, demo_path):
        """Training for 1 epoch should work."""
        trainer = BCTrainer(obs_dim=OBS_DIM, act_dim=ACT_DIM)
        stats = trainer.train(demo_path, epochs=1, batch_size=64)
        assert len(stats["train_losses"]) == 1
        assert len(stats["val_losses"]) == 1

    def test_large_batch_size(self, demo_path):
        """Batch size larger than dataset should still work (single batch)."""
        trainer = BCTrainer(obs_dim=OBS_DIM, act_dim=ACT_DIM)
        stats = trainer.train(demo_path, epochs=3, batch_size=10000)
        assert len(stats["train_losses"]) == 3

    def test_losses_are_finite(self, demo_path):
        """No NaN or Inf in loss values."""
        trainer = BCTrainer(obs_dim=OBS_DIM, act_dim=ACT_DIM)
        stats = trainer.train(demo_path, epochs=10, batch_size=64)
        for loss in stats["train_losses"]:
            assert math.isfinite(loss), f"Non-finite train loss: {loss}"
        for loss in stats["val_losses"]:
            assert math.isfinite(loss), f"Non-finite val loss: {loss}"


# ---------------------------------------------------------------------------
# TestBCTrainerPredict
# ---------------------------------------------------------------------------

class TestBCTrainerPredict:
    """Test the predict() method."""

    @pytest.fixture
    def trained_trainer(self):
        path = _make_synthetic_demos(n=500)
        trainer = BCTrainer(obs_dim=OBS_DIM, act_dim=ACT_DIM, lr=1e-3)
        trainer.train(path, epochs=10, batch_size=64)
        return trainer

    def test_single_obs_shape(self, trained_trainer):
        """Predict on a single observation should return (act_dim,) array."""
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action = trained_trainer.predict(obs)
        assert action.shape == (ACT_DIM,)

    def test_batch_obs_shape(self, trained_trainer):
        """Predict on a batch should return (batch, act_dim) array."""
        obs = np.random.randn(10, OBS_DIM).astype(np.float32)
        action = trained_trainer.predict(obs)
        assert action.shape == (10, ACT_DIM)

    def test_output_dtype(self, trained_trainer):
        """Predicted actions should be numpy float32."""
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action = trained_trainer.predict(obs)
        assert action.dtype == np.float32

    def test_deterministic(self, trained_trainer):
        """Same input should produce same output (deterministic)."""
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        a1 = trained_trainer.predict(obs)
        a2 = trained_trainer.predict(obs)
        np.testing.assert_array_equal(a1, a2)

    def test_predict_is_finite(self, trained_trainer):
        """Predictions should not contain NaN or Inf."""
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action = trained_trainer.predict(obs)
        assert np.all(np.isfinite(action)), f"Non-finite action: {action}"


# ---------------------------------------------------------------------------
# TestBCTrainerSaveLoadReference
# ---------------------------------------------------------------------------

class TestBCTrainerSaveLoadReference:
    """Test .pt reference model save/load round-trip."""

    @pytest.fixture
    def trained_trainer(self):
        path = _make_synthetic_demos(n=300)
        trainer = BCTrainer(obs_dim=OBS_DIM, act_dim=ACT_DIM)
        trainer.train(path, epochs=5, batch_size=64)
        return trainer

    def test_save_creates_file(self, trained_trainer, tmp_path):
        ref_path = str(tmp_path / "ref.pt")
        trained_trainer.save_reference(ref_path)
        assert os.path.exists(ref_path)

    def test_load_reference_matches_predictions(self, trained_trainer, tmp_path):
        """Loaded reference model should produce identical predictions."""
        ref_path = str(tmp_path / "ref.pt")
        trained_trainer.save_reference(ref_path)

        loaded = BCTrainer.load_reference(ref_path)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        original_action = trained_trainer.predict(obs)
        loaded_action = loaded.predict(obs)
        np.testing.assert_allclose(original_action, loaded_action, atol=1e-5)

    def test_load_reference_preserves_dims(self, trained_trainer, tmp_path):
        ref_path = str(tmp_path / "ref.pt")
        trained_trainer.save_reference(ref_path)
        loaded = BCTrainer.load_reference(ref_path)
        assert loaded.obs_dim == OBS_DIM
        assert loaded.act_dim == ACT_DIM

    def test_load_reference_custom_dims(self, tmp_path):
        """Reference save/load works with non-default dimensions."""
        trainer = BCTrainer(obs_dim=50, act_dim=3, hidden_sizes=(32, 16))
        ref_path = str(tmp_path / "ref.pt")
        trainer.save_reference(ref_path)
        loaded = BCTrainer.load_reference(ref_path)
        assert loaded.obs_dim == 50
        assert loaded.act_dim == 3
        # Predict should work
        obs = np.random.randn(50).astype(np.float32)
        action = loaded.predict(obs)
        assert action.shape == (3,)

    def test_save_reference_checkpoint_contents(self, trained_trainer, tmp_path):
        """Checkpoint should contain the expected keys."""
        ref_path = str(tmp_path / "ref.pt")
        trained_trainer.save_reference(ref_path)
        checkpoint = torch.load(ref_path, map_location="cpu", weights_only=True)
        assert "feature_extractor" in checkpoint
        assert "action_head" in checkpoint
        assert "obs_dim" in checkpoint
        assert "act_dim" in checkpoint


# ---------------------------------------------------------------------------
# TestBCTrainerSaveAsSB3
# ---------------------------------------------------------------------------

class TestBCTrainerSaveAsSB3:
    """Test SB3-compatible .zip export and PPO.load() round-trip.

    Uses a minimal gymnasium Box environment as a stand-in so that we don't
    need the Rust engine for these tests.
    """

    @pytest.fixture
    def mock_env(self):
        """Create a simple Box env with matching obs/act dims."""
        import gymnasium as gym
        return gym.make(
            "Pendulum-v1",
        )

    @pytest.fixture
    def bc_trainer_for_pendulum(self):
        """BCTrainer sized for Pendulum-v1 (obs=3, act=1)."""
        path = _make_synthetic_demos(n=500, obs_dim=3, act_dim=1, seed=99)
        trainer = BCTrainer(obs_dim=3, act_dim=1, lr=1e-3)
        trainer.train(path, epochs=10, batch_size=64)
        return trainer

    def test_save_creates_zip(self, bc_trainer_for_pendulum, mock_env, tmp_path):
        """save_as_sb3 should create a .zip file."""
        out = str(tmp_path / "bc_model")
        bc_trainer_for_pendulum.save_as_sb3(out, mock_env)
        assert os.path.exists(out + ".zip")

    def test_ppo_load_succeeds(self, bc_trainer_for_pendulum, mock_env, tmp_path):
        """PPO.load() should succeed on the saved .zip."""
        from stable_baselines3 import PPO

        out = str(tmp_path / "bc_model")
        bc_trainer_for_pendulum.save_as_sb3(out, mock_env)
        loaded = PPO.load(out, env=mock_env)
        assert loaded is not None

    def test_loaded_model_predicts(self, bc_trainer_for_pendulum, mock_env, tmp_path):
        """Loaded PPO model should produce actions with correct shape."""
        from stable_baselines3 import PPO

        out = str(tmp_path / "bc_model")
        bc_trainer_for_pendulum.save_as_sb3(out, mock_env)
        loaded = PPO.load(out, env=mock_env)

        obs, _ = mock_env.reset()
        action, _ = loaded.predict(obs, deterministic=True)
        assert action.shape == mock_env.action_space.shape

    def test_weights_transferred_to_policy_net(
        self, bc_trainer_for_pendulum, mock_env, tmp_path
    ):
        """BC feature_extractor weights should appear in SB3 policy_net."""
        from stable_baselines3 import PPO

        out = str(tmp_path / "bc_model")
        bc_trainer_for_pendulum.save_as_sb3(out, mock_env)
        loaded = PPO.load(out, env=mock_env)

        # Compare first Linear layer weights
        bc_w = bc_trainer_for_pendulum.feature_extractor[0].weight.data.cpu()
        sb3_w = loaded.policy.mlp_extractor.policy_net[0].weight.data.cpu()
        np.testing.assert_allclose(
            bc_w.numpy(), sb3_w.numpy(), atol=1e-6,
            err_msg="Policy net weights don't match BC feature_extractor",
        )

    def test_weights_transferred_to_action_net(
        self, bc_trainer_for_pendulum, mock_env, tmp_path
    ):
        """BC action_head weights should appear in SB3 action_net."""
        from stable_baselines3 import PPO

        out = str(tmp_path / "bc_model")
        bc_trainer_for_pendulum.save_as_sb3(out, mock_env)
        loaded = PPO.load(out, env=mock_env)

        bc_w = bc_trainer_for_pendulum.action_head.weight.data.cpu()
        sb3_w = loaded.policy.action_net.weight.data.cpu()
        np.testing.assert_allclose(
            bc_w.numpy(), sb3_w.numpy(), atol=1e-6,
            err_msg="Action net weights don't match BC action_head",
        )

    def test_weights_transferred_to_value_net(
        self, bc_trainer_for_pendulum, mock_env, tmp_path
    ):
        """BC feature_extractor weights should also be copied to value_net."""
        from stable_baselines3 import PPO

        out = str(tmp_path / "bc_model")
        bc_trainer_for_pendulum.save_as_sb3(out, mock_env)
        loaded = PPO.load(out, env=mock_env)

        bc_w = bc_trainer_for_pendulum.feature_extractor[0].weight.data.cpu()
        sb3_w = loaded.policy.mlp_extractor.value_net[0].weight.data.cpu()
        np.testing.assert_allclose(
            bc_w.numpy(), sb3_w.numpy(), atol=1e-6,
            err_msg="Value net weights don't match BC feature_extractor",
        )

    def test_loaded_model_can_resume_training(
        self, bc_trainer_for_pendulum, mock_env, tmp_path
    ):
        """PPO.load() model should be able to call .learn() (resume training)."""
        from stable_baselines3 import PPO

        out = str(tmp_path / "bc_model")
        bc_trainer_for_pendulum.save_as_sb3(out, mock_env)
        loaded = PPO.load(out, env=mock_env)
        # A short .learn() call to verify nothing crashes
        loaded.learn(total_timesteps=64)

    def test_save_returns_output_path(
        self, bc_trainer_for_pendulum, mock_env, tmp_path
    ):
        """save_as_sb3 should return the output_path argument."""
        out = str(tmp_path / "bc_model")
        result = bc_trainer_for_pendulum.save_as_sb3(out, mock_env)
        assert result == out


# ---------------------------------------------------------------------------
# TestBCTrainerEdgeCases
# ---------------------------------------------------------------------------

class TestBCTrainerEdgeCases:
    """Edge cases and error handling."""

    def test_tiny_dataset(self):
        """Training on a very small dataset should not crash."""
        path = _make_synthetic_demos(n=10, seed=7)
        trainer = BCTrainer(obs_dim=OBS_DIM, act_dim=ACT_DIM)
        stats = trainer.train(path, epochs=3, batch_size=8, val_split=0.2)
        assert len(stats["train_losses"]) == 3

    def test_single_sample(self):
        """Training on a single sample should not crash."""
        path = _make_synthetic_demos(n=1, seed=8)
        trainer = BCTrainer(obs_dim=OBS_DIM, act_dim=ACT_DIM)
        stats = trainer.train(path, epochs=2, batch_size=1, val_split=0.0)
        assert len(stats["train_losses"]) == 2

    def test_predict_untrained_model(self):
        """Predict on an untrained model should produce finite output."""
        trainer = BCTrainer(obs_dim=OBS_DIM, act_dim=ACT_DIM)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action = trainer.predict(obs)
        assert action.shape == (ACT_DIM,)
        assert np.all(np.isfinite(action))

    def test_hidden_sizes_single_layer(self):
        """Single hidden layer should work."""
        trainer = BCTrainer(obs_dim=10, act_dim=2, hidden_sizes=(32,))
        children = list(trainer.feature_extractor.children())
        assert len(children) == 2  # Linear, Tanh
        assert trainer.action_head.in_features == 32

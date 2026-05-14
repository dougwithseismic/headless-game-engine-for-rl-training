"""Smoke tests for the Dyna world model components.

All tests use synthetic data and run without the Rust engine.
Tests verify: ReplayBuffer, DynamicsEnsemble, DynamicsTrainer,
DynaCallback, and evaluation utilities.
"""

import os
import tempfile

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Constants matching CsLite 1v1
# ---------------------------------------------------------------------------

OBS_DIM = 250
ACT_DIM = 4
ACT_NVEC = [12, 2, 2, 3]
ACT_OH_DIM = sum(ACT_NVEC)  # 19


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_transition_batch(n=500, seed=42):
    """Generate synthetic transitions with learnable dynamics."""
    rng = np.random.RandomState(seed)
    obs = rng.randn(n, OBS_DIM).astype(np.float32) * 0.1
    actions = np.column_stack([
        rng.randint(0, nv, size=n) for nv in ACT_NVEC
    ]).astype(np.float32)
    delta = obs[:, :ACT_DIM] * 0.01 + actions * 0.005
    next_obs = obs.copy()
    next_obs[:, :ACT_DIM] += delta
    rewards = np.sum(delta, axis=1).astype(np.float32)
    dones = np.zeros(n, dtype=np.float32)
    return obs, actions, rewards, next_obs, dones


class _MockPolicy:
    """Minimal policy for rollout testing."""
    def predict(self, obs, deterministic=True):
        n = obs.shape[0] if obs.ndim == 2 else 1
        actions = np.column_stack([
            np.random.randint(0, nv, size=n) for nv in ACT_NVEC
        ]).astype(np.float32)
        return actions, None


# ===========================================================================
# ReplayBuffer Tests
# ===========================================================================

from training.replay_buffer import ReplayBuffer


class TestReplayBuffer:
    def test_add_single(self):
        buf = ReplayBuffer(capacity=100, obs_dim=OBS_DIM, act_dim=ACT_DIM)
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        act = np.zeros(ACT_DIM, dtype=np.float32)
        buf.add(obs, act, 1.0, obs, False)
        assert len(buf) == 1

    def test_add_batch(self):
        buf = ReplayBuffer(capacity=1000, obs_dim=OBS_DIM, act_dim=ACT_DIM)
        obs, actions, rewards, next_obs, dones = _make_transition_batch(200)
        buf.add_batch(obs, actions, rewards, next_obs, dones)
        assert len(buf) == 200

    def test_fifo_eviction(self):
        buf = ReplayBuffer(capacity=50, obs_dim=OBS_DIM, act_dim=ACT_DIM)
        obs, actions, rewards, next_obs, dones = _make_transition_batch(100)
        buf.add_batch(obs, actions, rewards, next_obs, dones)
        assert len(buf) == 50
        assert buf.full

    def test_sample_shape(self):
        buf = ReplayBuffer(capacity=1000, obs_dim=OBS_DIM, act_dim=ACT_DIM)
        obs, actions, rewards, next_obs, dones = _make_transition_batch(200)
        buf.add_batch(obs, actions, rewards, next_obs, dones)
        batch = buf.sample(32)
        assert batch["observations"].shape == (32, OBS_DIM)
        assert batch["actions"].shape == (32, ACT_DIM)
        assert batch["rewards"].shape == (32,)
        assert batch["next_observations"].shape == (32, OBS_DIM)
        assert batch["dones"].shape == (32,)

    def test_sample_from_partial_buffer(self):
        buf = ReplayBuffer(capacity=1000, obs_dim=OBS_DIM, act_dim=ACT_DIM)
        obs, actions, rewards, next_obs, dones = _make_transition_batch(10)
        buf.add_batch(obs, actions, rewards, next_obs, dones)
        batch = buf.sample(100)
        assert batch["observations"].shape == (10, OBS_DIM)

    def test_sample_states(self):
        buf = ReplayBuffer(capacity=1000, obs_dim=OBS_DIM, act_dim=ACT_DIM)
        obs, actions, rewards, next_obs, dones = _make_transition_batch(100)
        buf.add_batch(obs, actions, rewards, next_obs, dones)
        states = buf.sample_states(20)
        assert states.shape == (20, OBS_DIM)

    def test_save_load_roundtrip(self):
        buf = ReplayBuffer(capacity=500, obs_dim=OBS_DIM, act_dim=ACT_DIM)
        obs, actions, rewards, next_obs, dones = _make_transition_batch(200)
        buf.add_batch(obs, actions, rewards, next_obs, dones)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "buf.npz")
            buf.save(path)
            loaded = ReplayBuffer.load(path)

            assert len(loaded) == len(buf)
            assert loaded.obs_dim == OBS_DIM
            assert loaded.act_dim == ACT_DIM
            np.testing.assert_array_almost_equal(
                loaded.observations[:200], buf.observations[:200]
            )

    def test_empty_buffer_raises(self):
        buf = ReplayBuffer(capacity=100, obs_dim=OBS_DIM, act_dim=ACT_DIM)
        with pytest.raises(ValueError, match="empty"):
            buf.sample(10)

    def test_wraparound_correctness(self):
        buf = ReplayBuffer(capacity=10, obs_dim=OBS_DIM, act_dim=ACT_DIM)
        for i in range(25):
            obs = np.full(OBS_DIM, float(i), dtype=np.float32)
            buf.add(obs, np.zeros(ACT_DIM), 0.0, obs, False)
        assert len(buf) == 10
        last_obs = buf.observations[buf._pos - 1]
        assert last_obs[0] == 24.0


# ===========================================================================
# DynamicsEnsemble Tests
# ===========================================================================

from training.dynamics_model import DynamicsEnsemble, one_hot_actions


class TestOneHotActions:
    def test_shape(self):
        actions = np.array([[3, 1, 0, 2]], dtype=np.float32)
        oh = one_hot_actions(actions, ACT_NVEC)
        assert oh.shape == (1, ACT_OH_DIM)

    def test_correct_encoding(self):
        actions = np.array([[0, 0, 0, 0]], dtype=np.float32)
        oh = one_hot_actions(actions, ACT_NVEC)
        assert oh[0, 0].item() == 1.0  # first element of first head
        assert oh[0, 12].item() == 1.0  # first element of second head
        assert oh[0, 14].item() == 1.0  # first element of third head
        assert oh[0, 16].item() == 1.0  # first element of fourth head
        assert oh.sum().item() == 4.0  # exactly 4 ones

    def test_batch(self):
        actions = np.array([
            [11, 1, 1, 2],
            [0, 0, 0, 0],
        ], dtype=np.float32)
        oh = one_hot_actions(actions, ACT_NVEC)
        assert oh.shape == (2, ACT_OH_DIM)
        assert oh[0].sum().item() == 4.0
        assert oh[1].sum().item() == 4.0

    def test_1d_input(self):
        actions = np.array([5, 1, 0, 1], dtype=np.float32)
        oh = one_hot_actions(actions, ACT_NVEC)
        assert oh.shape == (1, ACT_OH_DIM)


class TestDynamicsEnsemble:
    def _make_ensemble(self, n_models=3, hidden=64, n_layers=2):
        return DynamicsEnsemble(
            obs_dim=OBS_DIM,
            act_dim=ACT_DIM,
            act_nvec=ACT_NVEC,
            n_models=n_models,
            hidden=hidden,
            n_layers=n_layers,
        )

    def test_predict_shapes(self):
        model = self._make_ensemble()
        obs, actions, _, _, _ = _make_transition_batch(16)
        model.fit_normalizer(obs)
        delta, rew, disagree = model.predict(obs, actions)
        assert delta.shape == (16, OBS_DIM)
        assert rew.shape == (16,)
        assert disagree.shape == (16,)

    def test_predict_single_member(self):
        model = self._make_ensemble()
        obs, actions, _, _, _ = _make_transition_batch(16)
        model.fit_normalizer(obs)
        delta, rew, disagree = model.predict(obs, actions, member_idx=0)
        assert delta.shape == (16, OBS_DIM)
        assert np.all(disagree == 0)  # single member = no disagreement

    def test_predict_next_obs(self):
        model = self._make_ensemble()
        obs, actions, _, _, _ = _make_transition_batch(16)
        model.fit_normalizer(obs)
        next_obs, rew, _ = model.predict_next_obs(obs, actions)
        assert next_obs.shape == (16, OBS_DIM)

    def test_train_reduces_loss(self):
        model = self._make_ensemble(n_models=2, hidden=32, n_layers=2)
        obs, actions, rewards, next_obs, _ = _make_transition_batch(500, seed=0)
        model.fit_normalizer(obs)

        losses_start = model.train_on_batch(obs, actions, next_obs, rewards)

        for _ in range(30):
            model.train_on_batch(obs, actions, next_obs, rewards)

        losses_end = model.train_on_batch(obs, actions, next_obs, rewards)

        assert np.mean(losses_end) < np.mean(losses_start), (
            f"Training did not reduce loss: {np.mean(losses_start):.4f} -> {np.mean(losses_end):.4f}"
        )

    def test_rollout_shapes(self):
        model = self._make_ensemble()
        obs, _, _, _, _ = _make_transition_batch(8)
        model.fit_normalizer(obs)
        starts = obs[:8]
        policy = _MockPolicy()
        result = model.rollout(starts, policy, horizon=5)
        assert result["observations"].shape == (8, 6, OBS_DIM)
        assert result["actions"].shape == (8, 5, ACT_DIM)
        assert result["rewards"].shape == (8, 5)
        assert result["dones"].shape == (8, 5)
        assert result["disagreements"].shape == (8, 5)

    def test_rollout_disagreement_truncation(self):
        model = self._make_ensemble()
        obs, _, _, _, _ = _make_transition_batch(8)
        model.fit_normalizer(obs)
        starts = obs[:8]
        policy = _MockPolicy()
        result = model.rollout(
            starts, policy, horizon=5, disagree_threshold=0.0
        )
        assert result["dones"].sum() > 0

    def test_save_load_roundtrip(self):
        model = self._make_ensemble(n_models=2, hidden=32)
        obs, actions, _, _, _ = _make_transition_batch(16)
        model.fit_normalizer(obs)
        delta_before, _, _ = model.predict(obs, actions)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ensemble.pt")
            model.save(path)
            loaded = DynamicsEnsemble.load(path)
            delta_after, _, _ = loaded.predict(obs, actions)
            np.testing.assert_array_almost_equal(delta_before, delta_after, decimal=5)


# ===========================================================================
# DynamicsTrainer Tests
# ===========================================================================

from training.dynamics_trainer import DynamicsTrainer


class TestDynamicsTrainer:
    def _setup(self):
        buf = ReplayBuffer(capacity=5000, obs_dim=OBS_DIM, act_dim=ACT_DIM)
        obs, actions, rewards, next_obs, dones = _make_transition_batch(2000)
        buf.add_batch(obs, actions, rewards, next_obs, dones)

        ensemble = DynamicsEnsemble(
            obs_dim=OBS_DIM, act_dim=ACT_DIM, act_nvec=ACT_NVEC,
            n_models=2, hidden=64, n_layers=2,
        )
        trainer = DynamicsTrainer(ensemble, buf, batch_size=128)
        return trainer, ensemble, buf

    def test_train_returns_metrics(self):
        trainer, _, _ = self._setup()
        metrics = trainer.train(n_steps=3)
        assert "mean_loss" in metrics
        assert "val_loss" in metrics
        assert "per_member_loss" in metrics
        assert not np.isnan(metrics["mean_loss"])

    def test_training_improves_loss(self):
        trainer, _, _ = self._setup()
        m1 = trainer.train(n_steps=5)
        for _ in range(10):
            trainer.train(n_steps=5)
        m2 = trainer.train(n_steps=5)
        assert m2["val_loss"] < m1["val_loss"] * 1.5, (
            "Validation loss should not explode after training"
        )

    def test_evaluate_accuracy(self):
        trainer, _, _ = self._setup()
        trainer.train(n_steps=5)
        metrics = trainer.evaluate_accuracy(n_samples=200)
        assert "state_mse" in metrics
        assert "reward_mse" in metrics
        assert "r_squared" in metrics
        assert metrics["r_squared"].shape == (OBS_DIM,)

    def test_evaluate_nstep(self):
        trainer, _, _ = self._setup()
        trainer.train(n_steps=5)
        policy = _MockPolicy()
        metrics = trainer.evaluate_nstep(policy, horizon=5, n_rollouts=10)
        assert "one_step_mse" in metrics
        assert "step_drift" in metrics
        assert metrics["step_drift"].shape == (5,)

    def test_small_buffer_returns_nan(self):
        buf = ReplayBuffer(capacity=100, obs_dim=OBS_DIM, act_dim=ACT_DIM)
        ensemble = DynamicsEnsemble(
            obs_dim=OBS_DIM, act_dim=ACT_DIM, act_nvec=ACT_NVEC,
            n_models=2, hidden=32, n_layers=2,
        )
        trainer = DynamicsTrainer(ensemble, buf, batch_size=256)
        metrics = trainer.train(n_steps=3)
        assert np.isnan(metrics["mean_loss"])


# ===========================================================================
# DynaCallback Tests (mock integration)
# ===========================================================================

from training.dyna_callback import DynaCallback


class TestDynaCallback:
    def test_initialization(self):
        cb = DynaCallback(
            obs_dim=OBS_DIM, act_dim=ACT_DIM, act_nvec=ACT_NVEC,
            buffer_capacity=1000, warmup_steps=100,
        )
        assert cb.obs_dim == OBS_DIM
        assert len(cb.replay_buffer) == 0
        assert cb.ensemble.n_models == 5

    def test_manual_buffer_fill_and_train(self):
        cb = DynaCallback(
            obs_dim=OBS_DIM, act_dim=ACT_DIM, act_nvec=ACT_NVEC,
            buffer_capacity=5000, warmup_steps=100,
            model_train_steps=5, n_models=2, hidden=64, n_layers=2,
        )
        obs, actions, rewards, next_obs, dones = _make_transition_batch(500)
        cb.replay_buffer.add_batch(obs, actions, rewards, next_obs, dones)
        metrics = cb.trainer.train(n_steps=5)
        assert not np.isnan(metrics["mean_loss"])

    def test_ensemble_accessible(self):
        cb = DynaCallback(
            obs_dim=OBS_DIM, act_dim=ACT_DIM, act_nvec=ACT_NVEC,
        )
        assert cb.get_ensemble() is cb.ensemble
        assert cb.get_replay_buffer() is cb.replay_buffer

    def test_checkpoint_roundtrip(self):
        cb = DynaCallback(
            obs_dim=OBS_DIM, act_dim=ACT_DIM, act_nvec=ACT_NVEC,
            buffer_capacity=1000, n_models=2, hidden=32,
        )
        obs, actions, rewards, next_obs, dones = _make_transition_batch(100)
        cb.replay_buffer.add_batch(obs, actions, rewards, next_obs, dones)
        cb.ensemble.fit_normalizer(obs)

        with tempfile.TemporaryDirectory() as tmpdir:
            cb.save_checkpoint(tmpdir)
            cb2 = DynaCallback(
                obs_dim=OBS_DIM, act_dim=ACT_DIM, act_nvec=ACT_NVEC,
                buffer_capacity=1000, n_models=2, hidden=32,
            )
            cb2.load_checkpoint(tmpdir)
            assert len(cb2.replay_buffer) == 100


# ===========================================================================
# Evaluation Utils Tests
# ===========================================================================

from training.dynamics_eval import (
    evaluate_1step,
    evaluate_nstep,
    evaluate_world_model,
    print_report,
)


class TestDynamicsEval:
    def _setup_trained(self):
        buf = ReplayBuffer(capacity=5000, obs_dim=OBS_DIM, act_dim=ACT_DIM)
        obs, actions, rewards, next_obs, dones = _make_transition_batch(2000)
        buf.add_batch(obs, actions, rewards, next_obs, dones)

        ensemble = DynamicsEnsemble(
            obs_dim=OBS_DIM, act_dim=ACT_DIM, act_nvec=ACT_NVEC,
            n_models=2, hidden=64, n_layers=2,
        )
        ensemble.fit_normalizer(obs)
        for _ in range(20):
            ensemble.train_on_batch(obs, actions, next_obs, rewards)
        return ensemble, buf

    def test_evaluate_1step(self):
        ensemble, buf = self._setup_trained()
        results = evaluate_1step(ensemble, buf, n_samples=200)
        assert "state_mse" in results
        assert "r_squared" in results
        assert results["r_squared"].shape == (OBS_DIM,)

    def test_evaluate_nstep(self):
        ensemble, buf = self._setup_trained()
        policy = _MockPolicy()
        results = evaluate_nstep(ensemble, buf, policy, horizons=[1, 3, 5])
        assert "real_1step_mse" in results
        assert 1 in results["horizons"]
        assert 5 in results["horizons"]

    def test_full_evaluation(self):
        ensemble, buf = self._setup_trained()
        policy = _MockPolicy()
        results = evaluate_world_model(
            ensemble, buf, policy,
            n_1step=200, n_rollouts=20, horizons=[1, 3],
        )
        assert "one_step" in results
        assert "nstep" in results

    def test_print_report_runs(self, capsys):
        ensemble, buf = self._setup_trained()
        policy = _MockPolicy()
        results = evaluate_world_model(
            ensemble, buf, policy,
            n_1step=100, n_rollouts=10, horizons=[1, 3],
        )
        print_report(results)
        captured = capsys.readouterr()
        assert "DYNAMICS MODEL EVALUATION REPORT" in captured.out
        assert "R-squared" in captured.out

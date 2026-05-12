"""
Tests for the unified PPOTrainer.

Tests cover:
  - Construction and parameter storage
  - GYM_REGISTRY lookups (valid and invalid scenarios)
  - _import_gym_class dynamic import
  - _make_env_kwargs for each scenario type
  - _build_game_config loading
  - _experiment_dict serialization
  - _build_callbacks produces correct callback types and counts
  - save_experiment is called with correct 3-arg signature
  - train() orchestration (mocked end-to-end)
"""

import json
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch, call

import pytest
from stable_baselines3.common.vec_env import VecEnv


def _make_mock_vec_env():
    """Create a MagicMock that passes isinstance(obj, VecEnv) checks.

    SB3's EvalCallback and other utilities use isinstance checks to
    distinguish VecEnv from raw Gym envs, so a plain MagicMock fails.
    """
    return MagicMock(spec=VecEnv)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_run_dir(tmp_path):
    """Create a temp run directory with expected sub-dirs."""
    run_dir = str(tmp_path / "test_run")
    for subdir in ("checkpoints", "eval_logs", "tb", "best"):
        os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)
    return run_dir


@pytest.fixture
def sample_config(tmp_path):
    """Create a minimal config JSON file."""
    cfg = {
        "arena": {"width": 100, "height": 100},
        "movement": {"speed": 10},
        "combat": {"damage": 5},
    }
    path = str(tmp_path / "test_config.json")
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


# ---------------------------------------------------------------------------
# GYM_REGISTRY tests
# ---------------------------------------------------------------------------

class TestGymRegistry:
    def test_all_expected_scenarios_present(self):
        from training.ppo_trainer import GYM_REGISTRY

        expected = ["cs_lite", "cs-lite", "cs_lite_dummy", "tactical"]
        for scenario in expected:
            assert scenario in GYM_REGISTRY, f"Missing scenario: {scenario}"

    def test_cs_lite_maps_to_correct_class(self):
        from training.ppo_trainer import GYM_REGISTRY

        assert GYM_REGISTRY["cs_lite"] == "glgym.gym_cs_lite.CsLiteGym"
        assert GYM_REGISTRY["cs-lite"] == "glgym.gym_cs_lite.CsLiteGym"
        assert GYM_REGISTRY["cs_lite_dummy"] == "glgym.gym_cs_lite.CsLiteGym"

    def test_tactical_maps_to_correct_class(self):
        from training.ppo_trainer import GYM_REGISTRY

        assert GYM_REGISTRY["tactical"] == "glgym.gym_tactical.TacticalGym"


# ---------------------------------------------------------------------------
# _import_gym_class tests
# ---------------------------------------------------------------------------

class TestImportGymClass:
    def test_unknown_scenario_raises_valueerror(self):
        from training.ppo_trainer import _import_gym_class

        with pytest.raises(ValueError, match="Unknown scenario"):
            _import_gym_class("nonexistent_scenario")

    def test_error_message_lists_available_scenarios(self):
        from training.ppo_trainer import _import_gym_class

        with pytest.raises(ValueError, match="cs_lite"):
            _import_gym_class("nonexistent_scenario")

    @patch("importlib.import_module")
    def test_imports_correct_module_for_cs_lite(self, mock_import):
        from training.ppo_trainer import _import_gym_class

        mock_mod = MagicMock()
        mock_mod.CsLiteGym = "FakeCsLiteGym"
        mock_import.return_value = mock_mod

        result = _import_gym_class("cs_lite")
        mock_import.assert_called_once_with("glgym.gym_cs_lite")
        assert result == "FakeCsLiteGym"

    @patch("importlib.import_module")
    def test_imports_correct_module_for_tactical(self, mock_import):
        from training.ppo_trainer import _import_gym_class

        mock_mod = MagicMock()
        mock_mod.TacticalGym = "FakeTacticalGym"
        mock_import.return_value = mock_mod

        result = _import_gym_class("tactical")
        mock_import.assert_called_once_with("glgym.gym_tactical")
        assert result == "FakeTacticalGym"


# ---------------------------------------------------------------------------
# PPOTrainer construction tests
# ---------------------------------------------------------------------------

class TestPPOTrainerConstruction:
    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_run_dir", return_value="/tmp/fake_run")
    def test_default_construction(self, mock_run_dir, mock_import, sample_config):
        from training.ppo_trainer import PPOTrainer

        trainer = PPOTrainer(
            scenario="cs_lite",
            config_path=sample_config,
        )

        assert trainer.scenario == "cs_lite"
        assert trainer.name == "cs_lite"
        assert trainer.lr == 3e-4
        assert trainer.n_steps == 4096
        assert trainer.batch_size == 256
        assert trainer.n_epochs == 4
        assert trainer.gamma == 0.99
        assert trainer.gae_lambda == 0.95
        assert trainer.clip_range == 0.2
        assert trainer.ent_coef == 0.01
        assert trainer.n_envs == 32
        assert trainer.frame_skip == 4
        assert trainer.max_steps == 2048
        assert trainer.phase is None
        assert trainer.timesteps == 3_000_000
        assert trainer.resume is None
        assert trainer.kl_anchor is None
        assert trainer.self_play is False
        assert trainer.auto_stop is False

    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_run_dir", return_value="/tmp/fake_run")
    def test_custom_name(self, mock_run_dir, mock_import, sample_config):
        from training.ppo_trainer import PPOTrainer

        trainer = PPOTrainer(
            scenario="cs_lite",
            config_path=sample_config,
            name="my_custom_run",
        )
        assert trainer.name == "my_custom_run"

    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_run_dir", return_value="/tmp/fake_run")
    def test_custom_hyperparams(self, mock_run_dir, mock_import, sample_config):
        from training.ppo_trainer import PPOTrainer

        trainer = PPOTrainer(
            scenario="tactical",
            config_path=sample_config,
            lr=1e-3,
            n_steps=2048,
            batch_size=128,
            gamma=0.98,
            ent_coef=0.005,
            n_envs=16,
        )
        assert trainer.lr == 1e-3
        assert trainer.n_steps == 2048
        assert trainer.batch_size == 128
        assert trainer.gamma == 0.98
        assert trainer.ent_coef == 0.005
        assert trainer.n_envs == 16

    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    def test_custom_run_dir_skips_make_run_dir(self, mock_import, sample_config, tmp_run_dir):
        from training.ppo_trainer import PPOTrainer

        trainer = PPOTrainer(
            scenario="cs_lite",
            config_path=sample_config,
            run_dir=tmp_run_dir,
        )
        assert trainer.run_dir == tmp_run_dir

    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_run_dir", return_value="/tmp/fake_run")
    def test_entropy_schedule_steps_defaults_to_timesteps(self, mock_run_dir, mock_import, sample_config):
        from training.ppo_trainer import PPOTrainer

        trainer = PPOTrainer(
            scenario="cs_lite",
            config_path=sample_config,
            timesteps=5_000_000,
            entropy_schedule=(0.01, 0.001),
        )
        assert trainer.entropy_schedule_steps == 5_000_000


# ---------------------------------------------------------------------------
# _make_env_kwargs tests
# ---------------------------------------------------------------------------

class TestMakeEnvKwargs:
    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_run_dir", return_value="/tmp/fake_run")
    def test_basic_env_kwargs(self, mock_run_dir, mock_import, sample_config):
        from training.ppo_trainer import PPOTrainer

        trainer = PPOTrainer(
            scenario="cs_lite",
            config_path=sample_config,
            frame_skip=2,
            max_steps=1024,
        )
        kwargs = trainer._make_env_kwargs()

        assert kwargs["scenario"] == "cs_lite"
        assert kwargs["frame_skip"] == 2
        assert kwargs["max_steps"] == 1024
        assert "phase" not in kwargs
        # config_path should NOT be in kwargs (passed separately to make_vec_env)
        assert "config_path" not in kwargs

    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_run_dir", return_value="/tmp/fake_run")
    def test_env_kwargs_with_phase(self, mock_run_dir, mock_import, sample_config):
        from training.ppo_trainer import PPOTrainer

        trainer = PPOTrainer(
            scenario="cs_lite",
            config_path=sample_config,
            phase=2,
        )
        kwargs = trainer._make_env_kwargs()
        assert kwargs["phase"] == 2

    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_run_dir", return_value="/tmp/fake_run")
    def test_non_hyphenated_scenario_passed_as_is(self, mock_run_dir, mock_import, sample_config):
        from training.ppo_trainer import PPOTrainer

        trainer = PPOTrainer(
            scenario="tactical",
            config_path=sample_config,
        )
        kwargs = trainer._make_env_kwargs()
        assert kwargs["scenario"] == "tactical"


# ---------------------------------------------------------------------------
# _build_game_config tests
# ---------------------------------------------------------------------------

class TestBuildGameConfig:
    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_run_dir", return_value="/tmp/fake_run")
    def test_loads_config_from_file(self, mock_run_dir, mock_import, sample_config):
        from training.ppo_trainer import PPOTrainer

        trainer = PPOTrainer(
            scenario="cs_lite",
            config_path=sample_config,
        )
        cfg = trainer._build_game_config()
        assert "arena" in cfg
        assert cfg["arena"]["width"] == 100


# ---------------------------------------------------------------------------
# _experiment_dict tests
# ---------------------------------------------------------------------------

class TestExperimentDict:
    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_run_dir", return_value="/tmp/fake_run")
    def test_contains_all_essential_keys(self, mock_run_dir, mock_import, sample_config):
        from training.ppo_trainer import PPOTrainer

        trainer = PPOTrainer(
            scenario="cs_lite",
            config_path=sample_config,
            phase=2,
            timesteps=500_000,
        )
        exp = trainer._experiment_dict()

        required_keys = [
            "scenario", "config_path", "phase", "timesteps",
            "lr", "n_steps", "batch_size", "n_epochs", "gamma",
            "ent_coef", "n_envs", "frame_skip", "max_steps",
            "resume", "kl_anchor", "entropy_schedule", "self_play",
        ]
        for key in required_keys:
            assert key in exp, f"Missing key in experiment dict: {key}"

    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_run_dir", return_value="/tmp/fake_run")
    def test_entropy_schedule_converted_to_list(self, mock_run_dir, mock_import, sample_config):
        from training.ppo_trainer import PPOTrainer

        trainer = PPOTrainer(
            scenario="cs_lite",
            config_path=sample_config,
            entropy_schedule=(0.01, 0.001),
        )
        exp = trainer._experiment_dict()
        assert exp["entropy_schedule"] == [0.01, 0.001]

    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_run_dir", return_value="/tmp/fake_run")
    def test_no_entropy_schedule_is_none(self, mock_run_dir, mock_import, sample_config):
        from training.ppo_trainer import PPOTrainer

        trainer = PPOTrainer(
            scenario="cs_lite",
            config_path=sample_config,
        )
        exp = trainer._experiment_dict()
        assert exp["entropy_schedule"] is None

    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_run_dir", return_value="/tmp/fake_run")
    def test_experiment_dict_is_json_serializable(self, mock_run_dir, mock_import, sample_config):
        from training.ppo_trainer import PPOTrainer

        trainer = PPOTrainer(
            scenario="cs_lite",
            config_path=sample_config,
            entropy_schedule=(0.01, 0.001),
            phase=1,
        )
        exp = trainer._experiment_dict()
        # Should not raise
        serialized = json.dumps(exp)
        assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# _build_callbacks tests
# ---------------------------------------------------------------------------

class TestBuildCallbacks:
    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_run_dir", return_value="/tmp/fake_run")
    def test_minimal_callbacks(self, mock_run_dir, mock_import, sample_config):
        """With no optional features, should get checkpoint + eval + throughput."""
        from training.ppo_trainer import PPOTrainer
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
        from training.callbacks import ThroughputCallback

        trainer = PPOTrainer(
            scenario="cs_lite",
            config_path=sample_config,
            run_dir="/tmp/fake_run",
        )

        mock_vec_env = _make_mock_vec_env()
        mock_eval_env = _make_mock_vec_env()
        cb_list = trainer._build_callbacks(mock_vec_env, mock_eval_env)

        assert isinstance(cb_list, CallbackList)
        callbacks = cb_list.callbacks
        assert len(callbacks) == 3

        types = [type(cb).__name__ for cb in callbacks]
        assert "CheckpointCallback" in types
        assert "BehaviorEvalCallback" in types
        assert "ThroughputCallback" in types

    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_run_dir", return_value="/tmp/fake_run")
    def test_kl_anchor_callback_added(self, mock_run_dir, mock_import, sample_config):
        from training.ppo_trainer import PPOTrainer

        trainer = PPOTrainer(
            scenario="cs_lite",
            config_path=sample_config,
            kl_anchor="/some/bc_ref.pt",
            run_dir="/tmp/fake_run",
        )

        mock_vec_env = _make_mock_vec_env()
        mock_eval_env = _make_mock_vec_env()
        cb_list = trainer._build_callbacks(mock_vec_env, mock_eval_env)

        types = [type(cb).__name__ for cb in cb_list.callbacks]
        assert "KLAnchorCallback" in types

    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_run_dir", return_value="/tmp/fake_run")
    def test_entropy_schedule_callback_added(self, mock_run_dir, mock_import, sample_config):
        from training.ppo_trainer import PPOTrainer

        trainer = PPOTrainer(
            scenario="cs_lite",
            config_path=sample_config,
            entropy_schedule=(0.01, 0.001),
            entropy_schedule_steps=2_000_000,
            run_dir="/tmp/fake_run",
        )

        mock_vec_env = _make_mock_vec_env()
        mock_eval_env = _make_mock_vec_env()
        cb_list = trainer._build_callbacks(mock_vec_env, mock_eval_env)

        types = [type(cb).__name__ for cb in cb_list.callbacks]
        assert "EntropyScheduleCallback" in types

    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_run_dir", return_value="/tmp/fake_run")
    def test_self_play_callback_added(self, mock_run_dir, mock_import, sample_config):
        from training.ppo_trainer import PPOTrainer

        trainer = PPOTrainer(
            scenario="tactical",
            config_path=sample_config,
            self_play=True,
            run_dir="/tmp/fake_run",
        )

        mock_vec_env = _make_mock_vec_env()
        mock_eval_env = _make_mock_vec_env()
        cb_list = trainer._build_callbacks(mock_vec_env, mock_eval_env)

        types = [type(cb).__name__ for cb in cb_list.callbacks]
        assert "SelfPlaySwapCallback" in types

    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_run_dir", return_value="/tmp/fake_run")
    def test_auto_stop_callback_added(self, mock_run_dir, mock_import, sample_config):
        from training.ppo_trainer import PPOTrainer

        trainer = PPOTrainer(
            scenario="cs_lite",
            config_path=sample_config,
            auto_stop=True,
            patience=20,
            run_dir="/tmp/fake_run",
        )

        mock_vec_env = _make_mock_vec_env()
        mock_eval_env = _make_mock_vec_env()
        cb_list = trainer._build_callbacks(mock_vec_env, mock_eval_env)

        types = [type(cb).__name__ for cb in cb_list.callbacks]
        assert "PlateauStopCallback" in types

    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_run_dir", return_value="/tmp/fake_run")
    def test_all_optional_callbacks_together(self, mock_run_dir, mock_import, sample_config):
        """All optional features enabled should produce 7 callbacks total."""
        from training.ppo_trainer import PPOTrainer

        trainer = PPOTrainer(
            scenario="tactical",
            config_path=sample_config,
            kl_anchor="/some/bc_ref.pt",
            entropy_schedule=(0.01, 0.001),
            self_play=True,
            auto_stop=True,
            run_dir="/tmp/fake_run",
        )

        mock_vec_env = _make_mock_vec_env()
        mock_eval_env = _make_mock_vec_env()
        cb_list = trainer._build_callbacks(mock_vec_env, mock_eval_env)

        # checkpoint + eval + throughput + kl_anchor + entropy + selfplay + plateau
        assert len(cb_list.callbacks) == 7

    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_run_dir", return_value="/tmp/fake_run")
    def test_checkpoint_freq_scaled_by_n_envs(self, mock_run_dir, mock_import, sample_config):
        from training.ppo_trainer import PPOTrainer
        from stable_baselines3.common.callbacks import CheckpointCallback

        trainer = PPOTrainer(
            scenario="cs_lite",
            config_path=sample_config,
            n_envs=16,
            checkpoint_freq=1_000_000,
            run_dir="/tmp/fake_run",
        )

        mock_vec_env = _make_mock_vec_env()
        mock_eval_env = _make_mock_vec_env()
        cb_list = trainer._build_callbacks(mock_vec_env, mock_eval_env)

        checkpoint_cb = [cb for cb in cb_list.callbacks if isinstance(cb, CheckpointCallback)][0]
        assert checkpoint_cb.save_freq == 1_000_000 // 16

    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_run_dir", return_value="/tmp/fake_run")
    def test_eval_freq_scaled_by_n_envs(self, mock_run_dir, mock_import, sample_config):
        from training.ppo_trainer import PPOTrainer
        from training.callbacks import BehaviorEvalCallback

        trainer = PPOTrainer(
            scenario="cs_lite",
            config_path=sample_config,
            n_envs=8,
            eval_freq=100_000,
            run_dir="/tmp/fake_run",
        )

        mock_vec_env = _make_mock_vec_env()
        mock_eval_env = _make_mock_vec_env()
        cb_list = trainer._build_callbacks(mock_vec_env, mock_eval_env)

        eval_cb = [cb for cb in cb_list.callbacks if isinstance(cb, BehaviorEvalCallback)][0]
        assert eval_cb.eval_freq == 100_000

    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_run_dir", return_value="/tmp/fake_run")
    def test_self_play_callback_receives_train_env(self, mock_run_dir, mock_import, sample_config):
        """SelfPlaySwapCallback should receive the vec_env as train_env."""
        from training.ppo_trainer import PPOTrainer
        from training.callbacks import SelfPlaySwapCallback

        trainer = PPOTrainer(
            scenario="tactical",
            config_path=sample_config,
            self_play=True,
            swap_interval=200_000,
            scripted_warmup=500_000,
            run_dir="/tmp/fake_run",
        )

        mock_vec_env = _make_mock_vec_env()
        mock_eval_env = _make_mock_vec_env()
        cb_list = trainer._build_callbacks(mock_vec_env, mock_eval_env)

        sp_cb = [cb for cb in cb_list.callbacks if isinstance(cb, SelfPlaySwapCallback)][0]
        assert sp_cb.train_env is mock_vec_env
        assert sp_cb.swap_interval == 200_000
        assert sp_cb.scripted_warmup == 500_000


# ---------------------------------------------------------------------------
# train() orchestration tests (fully mocked)
# ---------------------------------------------------------------------------

class TestTrainOrchestration:
    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_vec_env")
    @patch("training.ppo_trainer.save_experiment")
    @patch("training.ppo_trainer.PPO")
    def test_train_creates_envs_with_correct_args(
        self, MockPPO, mock_save_exp, mock_make_vec, mock_import, sample_config, tmp_run_dir
    ):
        from training.ppo_trainer import PPOTrainer

        mock_model = MagicMock()
        MockPPO.return_value = mock_model

        mock_vec = _make_mock_vec_env()
        mock_make_vec.return_value = mock_vec

        trainer = PPOTrainer(
            scenario="cs_lite",
            config_path=sample_config,
            n_envs=8,
            frame_skip=2,
            max_steps=512,
            timesteps=256,
            run_dir=tmp_run_dir,
        )

        trainer.train()

        # make_vec_env should be called twice (train + eval)
        assert mock_make_vec.call_count == 2

        # First call: training env with n_envs=8
        train_call = mock_make_vec.call_args_list[0]
        assert train_call[1]["n_envs"] == 8
        assert train_call[1]["config_path"] == trainer.config_path

        # Second call: eval env with min(4, n_envs)
        eval_call = mock_make_vec.call_args_list[1]
        assert eval_call[1]["n_envs"] == 4

    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_vec_env")
    @patch("training.ppo_trainer.save_experiment")
    @patch("training.ppo_trainer.PPO")
    def test_train_creates_ppo_model(
        self, MockPPO, mock_save_exp, mock_make_vec, mock_import, sample_config, tmp_run_dir
    ):
        from training.ppo_trainer import PPOTrainer

        mock_model = MagicMock()
        MockPPO.return_value = mock_model
        mock_make_vec.return_value = _make_mock_vec_env()

        trainer = PPOTrainer(
            scenario="cs_lite",
            config_path=sample_config,
            lr=1e-3,
            n_steps=2048,
            batch_size=128,
            n_epochs=8,
            gamma=0.98,
            gae_lambda=0.9,
            clip_range=0.1,
            ent_coef=0.005,
            timesteps=256,
            run_dir=tmp_run_dir,
        )

        trainer.train()

        MockPPO.assert_called_once()
        call_kwargs = MockPPO.call_args[1]
        assert call_kwargs["learning_rate"] == 1e-3
        assert call_kwargs["n_steps"] == 2048
        assert call_kwargs["batch_size"] == 128
        assert call_kwargs["n_epochs"] == 8
        assert call_kwargs["gamma"] == 0.98
        assert call_kwargs["gae_lambda"] == 0.9
        assert call_kwargs["clip_range"] == 0.1
        assert call_kwargs["ent_coef"] == 0.005

    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_vec_env")
    @patch("training.ppo_trainer.save_experiment")
    @patch("training.ppo_trainer.PPO")
    def test_train_saves_experiment_with_three_args(
        self, MockPPO, mock_save_exp, mock_make_vec, mock_import, sample_config, tmp_run_dir
    ):
        """save_experiment() requires (run_dir, config_dict, args_dict)."""
        from training.ppo_trainer import PPOTrainer

        mock_model = MagicMock()
        MockPPO.return_value = mock_model
        mock_make_vec.return_value = _make_mock_vec_env()

        trainer = PPOTrainer(
            scenario="cs_lite",
            config_path=sample_config,
            timesteps=256,
            run_dir=tmp_run_dir,
        )

        trainer.train()

        mock_save_exp.assert_called_once()
        args = mock_save_exp.call_args[0]
        assert len(args) == 3, "save_experiment must be called with (run_dir, config, args)"
        assert args[0] == tmp_run_dir
        assert isinstance(args[1], dict)  # game config
        assert isinstance(args[2], dict)  # training args

    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_vec_env")
    @patch("training.ppo_trainer.save_experiment")
    @patch("training.ppo_trainer.PPO")
    def test_train_calls_model_learn(
        self, MockPPO, mock_save_exp, mock_make_vec, mock_import, sample_config, tmp_run_dir
    ):
        from training.ppo_trainer import PPOTrainer

        mock_model = MagicMock()
        MockPPO.return_value = mock_model
        mock_make_vec.return_value = _make_mock_vec_env()

        trainer = PPOTrainer(
            scenario="cs_lite",
            config_path=sample_config,
            timesteps=10_000,
            run_dir=tmp_run_dir,
        )

        trainer.train()

        mock_model.learn.assert_called_once()
        learn_kwargs = mock_model.learn.call_args[1]
        assert learn_kwargs["total_timesteps"] == 10_000
        assert learn_kwargs["tb_log_name"] == "cs_lite"

    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_vec_env")
    @patch("training.ppo_trainer.save_experiment")
    @patch("training.ppo_trainer.PPO")
    def test_train_saves_final_model(
        self, MockPPO, mock_save_exp, mock_make_vec, mock_import, sample_config, tmp_run_dir
    ):
        from training.ppo_trainer import PPOTrainer

        mock_model = MagicMock()
        MockPPO.return_value = mock_model
        mock_make_vec.return_value = _make_mock_vec_env()

        trainer = PPOTrainer(
            scenario="cs_lite",
            config_path=sample_config,
            timesteps=256,
            run_dir=tmp_run_dir,
        )

        result = trainer.train()

        expected_path = os.path.join(tmp_run_dir, "final_model")
        mock_model.save.assert_called_once_with(expected_path)
        assert result == f"{expected_path}.zip"

    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_vec_env")
    @patch("training.ppo_trainer.save_experiment")
    @patch("training.ppo_trainer.PPO")
    def test_train_closes_envs(
        self, MockPPO, mock_save_exp, mock_make_vec, mock_import, sample_config, tmp_run_dir
    ):
        from training.ppo_trainer import PPOTrainer

        mock_model = MagicMock()
        MockPPO.return_value = mock_model
        mock_vec = _make_mock_vec_env()
        mock_eval = _make_mock_vec_env()
        mock_make_vec.side_effect = [mock_vec, mock_eval]

        trainer = PPOTrainer(
            scenario="cs_lite",
            config_path=sample_config,
            timesteps=256,
            run_dir=tmp_run_dir,
        )

        trainer.train()

        mock_vec.close.assert_called_once()
        mock_eval.close.assert_called_once()

    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_vec_env")
    @patch("training.ppo_trainer.save_experiment")
    @patch("training.ppo_trainer.PPO")
    def test_resume_loads_existing_model(
        self, MockPPO, mock_save_exp, mock_make_vec, mock_import, sample_config, tmp_run_dir
    ):
        from training.ppo_trainer import PPOTrainer

        mock_loaded_model = MagicMock()
        MockPPO.load.return_value = mock_loaded_model
        mock_make_vec.return_value = _make_mock_vec_env()

        trainer = PPOTrainer(
            scenario="cs_lite",
            config_path=sample_config,
            resume="/path/to/model.zip",
            lr=5e-4,
            timesteps=256,
            run_dir=tmp_run_dir,
        )

        trainer.train()

        # PPO.load should be called instead of PPO()
        MockPPO.load.assert_called_once()
        MockPPO.assert_not_called()  # PPO() constructor should NOT be called

        # Learning rate should be updated
        assert mock_loaded_model.learning_rate == 5e-4

    @patch("training.ppo_trainer._import_gym_class", return_value=MagicMock)
    @patch("training.ppo_trainer.make_vec_env")
    @patch("training.ppo_trainer.save_experiment")
    @patch("training.ppo_trainer.PPO")
    def test_eval_envs_capped_at_n_envs(
        self, MockPPO, mock_save_exp, mock_make_vec, mock_import, sample_config, tmp_run_dir
    ):
        """When n_envs < 4, eval should use n_envs, not 4."""
        from training.ppo_trainer import PPOTrainer

        mock_model = MagicMock()
        MockPPO.return_value = mock_model
        mock_make_vec.return_value = _make_mock_vec_env()

        trainer = PPOTrainer(
            scenario="cs_lite",
            config_path=sample_config,
            n_envs=2,
            timesteps=256,
            run_dir=tmp_run_dir,
        )

        trainer.train()

        eval_call = mock_make_vec.call_args_list[1]
        assert eval_call[1]["n_envs"] == 2  # min(4, 2) = 2

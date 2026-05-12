"""
Tests for CLI entry point scripts.

Tests cover:
  - Argument parsing for all three scripts (train, collect_demos, evaluate)
  - --help exits cleanly with code 0
  - Default values are set correctly
  - Mode dispatch routes to the right function
  - Required argument validation
  - Import connectivity (all training modules are importable)
  - Entropy schedule parsing
  - Edge cases: missing --config for PPO, missing --demos for BC, etc.
"""

import subprocess
import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from io import StringIO


SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(SCRIPTS_DIR)


# ---------------------------------------------------------------------------
# Helper: run a script as a subprocess and capture output
# ---------------------------------------------------------------------------

def _run_script(script_name: str, args: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run a script via subprocess and return the result."""
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    return subprocess.run(
        [sys.executable, script_path] + args,
        capture_output=True,
        text=True,
        cwd=cwd or PYTHON_DIR,
        timeout=30,
    )


# ===========================================================================
# train.py tests
# ===========================================================================

class TestTrainHelp:
    """Verify --help exits cleanly for train.py."""

    def test_help_exits_zero(self):
        result = _run_script("train.py", ["--help"])
        assert result.returncode == 0
        assert "GhostLobby Training" in result.stdout

    def test_help_shows_modes(self):
        result = _run_script("train.py", ["--help"])
        assert "ppo" in result.stdout
        assert "bc" in result.stdout
        assert "curriculum" in result.stdout

    def test_help_shows_scenarios(self):
        result = _run_script("train.py", ["--help"])
        assert "arena3d" in result.stdout
        assert "tactical" in result.stdout
        assert "drone" in result.stdout


class TestTrainParseArgs:
    """Test argument parsing logic in train.py."""

    def test_missing_scenario_exits_nonzero(self):
        result = _run_script("train.py", ["--mode", "ppo"])
        assert result.returncode != 0

    def test_ppo_requires_config(self):
        result = _run_script("train.py", ["--scenario", "arena3d", "--mode", "ppo"])
        assert result.returncode != 0
        assert "config" in result.stderr.lower() or "config" in result.stdout.lower()

    def test_bc_requires_demos(self):
        result = _run_script("train.py", ["--scenario", "arena3d", "--mode", "bc"])
        assert result.returncode != 0
        assert "demos" in result.stderr.lower() or "demos" in result.stdout.lower()

    def test_curriculum_requires_curriculum_flag(self):
        result = _run_script("train.py", ["--scenario", "arena3d", "--mode", "curriculum"])
        assert result.returncode != 0
        assert "curriculum" in result.stderr.lower() or "curriculum" in result.stdout.lower()


class TestTrainArgDefaults:
    """Verify default argument values by importing parse_args."""

    def test_defaults(self):
        # Import the parse_args function directly
        sys.path.insert(0, SCRIPTS_DIR)
        try:
            from train import parse_args
            with patch("sys.argv", ["train.py", "--scenario", "arena3d"]):
                args = parse_args()
            assert args.scenario == "arena3d"
            assert args.mode == "ppo"
            assert args.timesteps == 3_000_000
            assert args.lr == 3e-4
            assert args.batch_size == 256
            assert args.n_steps == 4096
            assert args.n_epochs == 4
            assert args.gamma == 0.99
            assert args.ent_coef == 0.01
            assert args.n_envs == 32
            assert args.frame_skip == 4
            assert args.max_steps == 2048
            assert args.eval_freq == 100_000
            assert args.checkpoint_freq == 1_000_000
            assert args.self_play is False
            assert args.auto_stop is False
            assert args.patience == 15
            assert args.config is None
            assert args.resume is None
            assert args.kl_anchor is None
            assert args.entropy_schedule is None
            assert args.name is None
        finally:
            sys.path.remove(SCRIPTS_DIR)

    def test_entropy_schedule_parsing(self):
        sys.path.insert(0, SCRIPTS_DIR)
        try:
            from train import parse_args
            with patch("sys.argv", [
                "train.py", "--scenario", "arena3d",
                "--entropy-schedule", "0.01:0.001",
            ]):
                args = parse_args()
            assert args.entropy_schedule == "0.01:0.001"
        finally:
            sys.path.remove(SCRIPTS_DIR)

    def test_scenario_choices_reject_invalid(self):
        result = _run_script("train.py", ["--scenario", "nonexistent"])
        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower()


class TestTrainRunPPO:
    """Test the run_ppo dispatch function with mocked PPOTrainer."""

    def test_run_ppo_constructs_trainer(self):
        sys.path.insert(0, SCRIPTS_DIR)
        try:
            from train import run_ppo, parse_args

            mock_trainer = MagicMock()
            mock_trainer.train.return_value = "runs/test/final_model.zip"

            with patch("sys.argv", [
                "train.py", "--scenario", "arena3d",
                "--config", "configs/test.json",
                "--timesteps", "1000",
                "--entropy-schedule", "0.01:0.001",
            ]):
                args = parse_args()

            with patch("training.ppo_trainer.PPOTrainer", return_value=mock_trainer) as MockPPO:
                result = run_ppo(args)

            MockPPO.assert_called_once()
            call_kwargs = MockPPO.call_args[1]
            assert call_kwargs["scenario"] == "arena3d"
            assert call_kwargs["config_path"] == "configs/test.json"
            assert call_kwargs["timesteps"] == 1000
            assert call_kwargs["entropy_schedule"] == (0.01, 0.001)
            mock_trainer.train.assert_called_once()
        finally:
            sys.path.remove(SCRIPTS_DIR)

    def test_run_ppo_no_entropy_schedule(self):
        sys.path.insert(0, SCRIPTS_DIR)
        try:
            from train import run_ppo, parse_args

            mock_trainer = MagicMock()
            mock_trainer.train.return_value = "runs/test/final_model.zip"

            with patch("sys.argv", [
                "train.py", "--scenario", "tactical",
                "--config", "configs/test.json",
            ]):
                args = parse_args()

            with patch("training.ppo_trainer.PPOTrainer", return_value=mock_trainer) as MockPPO:
                run_ppo(args)

            call_kwargs = MockPPO.call_args[1]
            assert call_kwargs["entropy_schedule"] is None
        finally:
            sys.path.remove(SCRIPTS_DIR)


class TestTrainRunBC:
    """Test the run_bc dispatch function with mocks."""

    def test_run_bc_missing_demos_exits(self):
        sys.path.insert(0, SCRIPTS_DIR)
        try:
            from train import run_bc, parse_args

            with patch("sys.argv", [
                "train.py", "--scenario", "arena3d", "--mode", "bc",
            ]):
                args = parse_args()

            with pytest.raises(SystemExit) as exc:
                run_bc(args)
            assert exc.value.code == 1
        finally:
            sys.path.remove(SCRIPTS_DIR)


class TestTrainRunCurriculum:
    """Test the run_curriculum dispatch function with mocks."""

    def test_run_curriculum_missing_yaml_exits(self):
        sys.path.insert(0, SCRIPTS_DIR)
        try:
            from train import run_curriculum, parse_args

            with patch("sys.argv", [
                "train.py", "--scenario", "arena3d", "--mode", "curriculum",
            ]):
                args = parse_args()

            with pytest.raises(SystemExit) as exc:
                run_curriculum(args)
            assert exc.value.code == 1
        finally:
            sys.path.remove(SCRIPTS_DIR)


# ===========================================================================
# collect_demos.py tests
# ===========================================================================

class TestCollectDemosHelp:
    """Verify --help exits cleanly for collect_demos.py."""

    def test_help_exits_zero(self):
        result = _run_script("collect_demos.py", ["--help"])
        assert result.returncode == 0
        assert "Collect BC demonstrations" in result.stdout

    def test_help_shows_args(self):
        result = _run_script("collect_demos.py", ["--help"])
        assert "--scenario" in result.stdout
        assert "--config" in result.stdout
        assert "--episodes" in result.stdout
        assert "--output" in result.stdout


class TestCollectDemosArgs:
    """Test argument parsing for collect_demos.py."""

    def test_missing_required_args(self):
        result = _run_script("collect_demos.py", [])
        assert result.returncode != 0

    def test_missing_config(self):
        result = _run_script("collect_demos.py", ["--scenario", "arena3d"])
        assert result.returncode != 0


# ===========================================================================
# evaluate.py tests
# ===========================================================================

class TestEvaluateHelp:
    """Verify --help exits cleanly for evaluate.py."""

    def test_help_exits_zero(self):
        result = _run_script("evaluate.py", ["--help"])
        assert result.returncode == 0
        assert "Evaluate trained model" in result.stdout

    def test_help_shows_args(self):
        result = _run_script("evaluate.py", ["--help"])
        assert "--model" in result.stdout
        assert "--config" in result.stdout
        assert "--episodes" in result.stdout
        assert "--deterministic" in result.stdout


class TestEvaluateArgs:
    """Test argument parsing for evaluate.py."""

    def test_missing_model(self):
        result = _run_script("evaluate.py", ["--config", "test.json"])
        assert result.returncode != 0

    def test_missing_config(self):
        result = _run_script("evaluate.py", ["--model", "test.zip"])
        assert result.returncode != 0


# ===========================================================================
# Import connectivity tests
# ===========================================================================

class TestImportConnectivity:
    """Verify that all training modules can be imported from the python/ dir."""

    def test_import_ppo_trainer(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, '.'); "
             "from training.ppo_trainer import PPOTrainer, _import_gym_class; "
             "print('OK')"],
            capture_output=True,
            text=True,
            cwd=PYTHON_DIR,
            timeout=30,
        )
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_import_curriculum(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, '.'); "
             "from training.curriculum import CurriculumRunner; "
             "print('OK')"],
            capture_output=True,
            text=True,
            cwd=PYTHON_DIR,
            timeout=30,
        )
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_import_bc_collector(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, '.'); "
             "from training.bc_collector import collect_demonstrations, Arena3DExpertPolicy; "
             "print('OK')"],
            capture_output=True,
            text=True,
            cwd=PYTHON_DIR,
            timeout=30,
        )
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_import_bc_pretrain(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, '.'); "
             "from training.bc_pretrain import BCTrainer; "
             "print('OK')"],
            capture_output=True,
            text=True,
            cwd=PYTHON_DIR,
            timeout=30,
        )
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_import_utils(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, '.'); "
             "from training.utils import load_model, resolve_config, make_run_dir; "
             "print('OK')"],
            capture_output=True,
            text=True,
            cwd=PYTHON_DIR,
            timeout=30,
        )
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_gym_registry_lookup(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, '.'); "
             "from training.ppo_trainer import _import_gym_class; "
             "cls = _import_gym_class('arena3d'); print(cls.__name__); "
             "cls = _import_gym_class('tactical'); print(cls.__name__); "
             "cls = _import_gym_class('drone'); print(cls.__name__)"],
            capture_output=True,
            text=True,
            cwd=PYTHON_DIR,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Arena3DGym" in result.stdout
        assert "TacticalGym" in result.stdout
        assert "DroneGym" in result.stdout

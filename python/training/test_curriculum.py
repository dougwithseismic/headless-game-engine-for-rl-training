"""Tests for the curriculum auto-progression system.

Covers YAML config parsing, phase sequencing, weight transfer logic,
progress persistence, and BC warmstart orchestration. All tests are
pure-Python and mock external dependencies (PPOTrainer, file I/O).
"""

import json
import os

import numpy as np
import pytest
import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def arena3d_yaml(tmp_path):
    """Write a minimal arena3d-style curriculum YAML and return its path."""
    config = {
        "name": "arena3d_test",
        "scenario": "arena3d",
        "phases": [
            {
                "name": "aim_targets",
                "config": "configs/arena3d/phase1_aim_targets.json",
                "max_timesteps": 100,
                "advance_threshold": 15.0,
                "advance_patience": 3,
                "bc_warmstart": True,
                "bc_demos": "data/demos/arena3d_aim.npz",
                "kl_anchor": True,
                "kl_beta_start": 0.5,
                "kl_beta_end": 0.0,
                "kl_anneal_steps": 50,
                "entropy_start": 0.01,
                "entropy_end": 0.001,
                "n_envs": 2,
                "lr": 0.0003,
                "phase_mask": 1,
            },
            {
                "name": "tracking",
                "config": "configs/arena3d/phase2_track.json",
                "max_timesteps": 200,
                "advance_threshold": 20.0,
                "advance_patience": 3,
                "resume_from": "previous",
                "entropy_start": 0.008,
                "entropy_end": 0.002,
                "n_envs": 2,
                "lr": 0.0002,
                "phase_mask": 2,
            },
            {
                "name": "combat",
                "config": "configs/arena3d/phase4_combat.json",
                "max_timesteps": 300,
                "advance_threshold": 30.0,
                "advance_patience": 5,
                "resume_from": "previous",
                "self_play": True,
                "swap_interval": 100,
                "scripted_warmup": 50,
                "entropy_start": 0.005,
                "entropy_end": 0.001,
                "n_envs": 2,
                "lr": 0.0001,
                "phase_mask": 4,
            },
        ],
    }
    path = tmp_path / "curriculum.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return str(path)


@pytest.fixture
def single_phase_yaml(tmp_path):
    """Curriculum with one phase (no weight transfer)."""
    config = {
        "name": "single_phase",
        "scenario": "tactical",
        "phases": [
            {
                "name": "basic",
                "config": "configs/tactical/phase1_open.json",
                "max_timesteps": 100,
                "advance_threshold": 10.0,
                "advance_patience": 3,
                "n_envs": 2,
                "lr": 0.0003,
                "phase_mask": 1,
            },
        ],
    }
    path = tmp_path / "single.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return str(path)


@pytest.fixture
def no_threshold_yaml(tmp_path):
    """Curriculum with phases that have no advance_threshold (run to completion)."""
    config = {
        "name": "no_threshold",
        "scenario": "tactical",
        "phases": [
            {
                "name": "phase_a",
                "config": "configs/tactical/phase1_open.json",
                "max_timesteps": 100,
                "n_envs": 2,
                "lr": 0.0003,
            },
            {
                "name": "phase_b",
                "config": "configs/tactical/phase2_obstacles.json",
                "max_timesteps": 200,
                "resume_from": "previous",
                "n_envs": 2,
                "lr": 0.0002,
            },
        ],
    }
    path = tmp_path / "no_threshold.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return str(path)


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from training.curriculum import CurriculumRunner, PhaseResult


# ---------------------------------------------------------------------------
# Tests: YAML parsing and initialization
# ---------------------------------------------------------------------------

class TestCurriculumRunnerInit:
    """Test that CurriculumRunner correctly parses YAML configs."""

    def test_loads_name(self, arena3d_yaml):
        runner = CurriculumRunner(arena3d_yaml)
        assert runner.name == "arena3d_test"

    def test_loads_scenario(self, arena3d_yaml):
        runner = CurriculumRunner(arena3d_yaml)
        assert runner.scenario == "arena3d"

    def test_loads_phases(self, arena3d_yaml):
        runner = CurriculumRunner(arena3d_yaml)
        assert len(runner.phases) == 3

    def test_phase_names(self, arena3d_yaml):
        runner = CurriculumRunner(arena3d_yaml)
        names = [p["name"] for p in runner.phases]
        assert names == ["aim_targets", "tracking", "combat"]

    def test_phase_configs(self, arena3d_yaml):
        runner = CurriculumRunner(arena3d_yaml)
        assert runner.phases[0]["config"] == "configs/arena3d/phase1_aim_targets.json"
        assert runner.phases[1]["config"] == "configs/arena3d/phase2_track.json"

    def test_phase_timesteps(self, arena3d_yaml):
        runner = CurriculumRunner(arena3d_yaml)
        assert runner.phases[0]["max_timesteps"] == 100
        assert runner.phases[1]["max_timesteps"] == 200
        assert runner.phases[2]["max_timesteps"] == 300

    def test_phase_advance_threshold(self, arena3d_yaml):
        runner = CurriculumRunner(arena3d_yaml)
        assert runner.phases[0]["advance_threshold"] == 15.0
        assert runner.phases[1]["advance_threshold"] == 20.0

    def test_phase_resume_from(self, arena3d_yaml):
        runner = CurriculumRunner(arena3d_yaml)
        assert "resume_from" not in runner.phases[0]
        assert runner.phases[1]["resume_from"] == "previous"
        assert runner.phases[2]["resume_from"] == "previous"

    def test_phase_bc_warmstart(self, arena3d_yaml):
        runner = CurriculumRunner(arena3d_yaml)
        assert runner.phases[0].get("bc_warmstart") is True
        assert runner.phases[1].get("bc_warmstart") is None

    def test_phase_self_play(self, arena3d_yaml):
        runner = CurriculumRunner(arena3d_yaml)
        assert runner.phases[0].get("self_play") is None
        assert runner.phases[2].get("self_play") is True

    def test_phase_entropy_schedule(self, arena3d_yaml):
        runner = CurriculumRunner(arena3d_yaml)
        assert runner.phases[0]["entropy_start"] == 0.01
        assert runner.phases[0]["entropy_end"] == 0.001

    def test_custom_run_dir(self, arena3d_yaml, tmp_path):
        run_dir = str(tmp_path / "custom_runs")
        runner = CurriculumRunner(arena3d_yaml, run_dir=run_dir)
        assert runner.run_dir == run_dir
        assert os.path.isdir(run_dir)

    def test_auto_run_dir(self, arena3d_yaml):
        runner = CurriculumRunner(arena3d_yaml)
        assert runner.run_dir.startswith("runs/arena3d_test_")
        # Clean up auto-created dir
        if os.path.isdir(runner.run_dir):
            os.rmdir(runner.run_dir)

    def test_results_initially_empty(self, arena3d_yaml):
        runner = CurriculumRunner(arena3d_yaml)
        assert runner.results == []

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CurriculumRunner(str(tmp_path / "nonexistent.yaml"))

    def test_invalid_yaml_raises(self, tmp_path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("{{{{not valid yaml")
        with pytest.raises(yaml.YAMLError):
            CurriculumRunner(str(bad_yaml))

    def test_missing_name_raises(self, tmp_path):
        config = {"scenario": "test", "phases": []}
        path = tmp_path / "no_name.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)
        with pytest.raises(KeyError):
            CurriculumRunner(str(path))

    def test_missing_phases_raises(self, tmp_path):
        config = {"name": "test", "scenario": "test"}
        path = tmp_path / "no_phases.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)
        with pytest.raises(KeyError):
            CurriculumRunner(str(path))


# ---------------------------------------------------------------------------
# Tests: PhaseResult
# ---------------------------------------------------------------------------

class TestPhaseResult:
    """Test the PhaseResult data class."""

    def test_attributes(self):
        r = PhaseResult(
            phase_name="aim",
            model_path="/models/aim.zip",
            advanced=True,
            peak_reward=25.3,
            timesteps_used=100_000,
        )
        assert r.phase_name == "aim"
        assert r.model_path == "/models/aim.zip"
        assert r.advanced is True
        assert r.peak_reward == 25.3
        assert r.timesteps_used == 100_000

    def test_not_advanced(self):
        r = PhaseResult("combat", "/m.zip", False, 5.0, 50_000)
        assert r.advanced is False


# ---------------------------------------------------------------------------
# Tests: Resume logic
# ---------------------------------------------------------------------------

class TestResumeLogic:
    """Test _resolve_resume_model logic without running actual training."""

    def test_first_phase_no_resume(self, arena3d_yaml, tmp_path):
        runner = CurriculumRunner(arena3d_yaml, run_dir=str(tmp_path / "runs"))
        # First phase has no resume_from key
        model = runner._resolve_resume_model(runner.phases[0], current_model=None)
        assert model is None

    def test_resume_from_previous(self, arena3d_yaml, tmp_path):
        runner = CurriculumRunner(arena3d_yaml, run_dir=str(tmp_path / "runs"))
        model = runner._resolve_resume_model(
            runner.phases[1], current_model="/path/to/model.zip"
        )
        assert model == "/path/to/model.zip"

    def test_resume_from_previous_no_current(self, arena3d_yaml, tmp_path):
        runner = CurriculumRunner(arena3d_yaml, run_dir=str(tmp_path / "runs"))
        model = runner._resolve_resume_model(
            runner.phases[1], current_model=None
        )
        assert model is None

    def test_resume_from_explicit_path(self, tmp_path):
        config = {
            "name": "test",
            "scenario": "test",
            "phases": [
                {
                    "name": "p1",
                    "config": "c.json",
                    "max_timesteps": 100,
                    "resume_from": "/explicit/model.zip",
                },
            ],
        }
        path = tmp_path / "explicit.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)
        runner = CurriculumRunner(str(path), run_dir=str(tmp_path / "runs"))
        model = runner._resolve_resume_model(
            runner.phases[0], current_model="/other.zip"
        )
        assert model == "/explicit/model.zip"


# ---------------------------------------------------------------------------
# Tests: Progress persistence
# ---------------------------------------------------------------------------

class TestProgressPersistence:
    """Test _save_progress writes correct JSON."""

    def test_saves_progress_file(self, single_phase_yaml, tmp_path):
        run_dir = str(tmp_path / "runs")
        runner = CurriculumRunner(single_phase_yaml, run_dir=run_dir)
        runner.results.append(PhaseResult("basic", "/m.zip", True, 12.5, 50))
        runner._save_progress()

        progress_path = os.path.join(run_dir, "curriculum_progress.json")
        assert os.path.exists(progress_path)

    def test_progress_structure(self, single_phase_yaml, tmp_path):
        run_dir = str(tmp_path / "runs")
        runner = CurriculumRunner(single_phase_yaml, run_dir=run_dir)
        runner.results.append(PhaseResult("basic", "/m.zip", True, 12.5, 50))
        runner._save_progress()

        progress_path = os.path.join(run_dir, "curriculum_progress.json")
        with open(progress_path) as f:
            data = json.load(f)

        assert data["name"] == "single_phase"
        assert data["phases_completed"] == 1
        assert len(data["results"]) == 1

    def test_progress_result_fields(self, single_phase_yaml, tmp_path):
        run_dir = str(tmp_path / "runs")
        runner = CurriculumRunner(single_phase_yaml, run_dir=run_dir)
        runner.results.append(PhaseResult("basic", "/m.zip", True, 12.5, 50_000))
        runner._save_progress()

        progress_path = os.path.join(run_dir, "curriculum_progress.json")
        with open(progress_path) as f:
            data = json.load(f)

        result = data["results"][0]
        assert result["name"] == "basic"
        assert result["model"] == "/m.zip"
        assert result["advanced"] is True
        assert result["peak_reward"] == 12.5
        assert result["timesteps"] == 50_000

    def test_multiple_results(self, arena3d_yaml, tmp_path):
        run_dir = str(tmp_path / "runs")
        runner = CurriculumRunner(arena3d_yaml, run_dir=run_dir)
        runner.results.append(PhaseResult("aim", "/a.zip", True, 20.0, 100))
        runner.results.append(PhaseResult("track", "/b.zip", False, 15.0, 200))
        runner._save_progress()

        progress_path = os.path.join(run_dir, "curriculum_progress.json")
        with open(progress_path) as f:
            data = json.load(f)

        assert data["phases_completed"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["name"] == "aim"
        assert data["results"][1]["name"] == "track"

    def test_overwrites_on_subsequent_save(self, single_phase_yaml, tmp_path):
        run_dir = str(tmp_path / "runs")
        runner = CurriculumRunner(single_phase_yaml, run_dir=run_dir)

        runner.results.append(PhaseResult("basic", "/m1.zip", False, 5.0, 10))
        runner._save_progress()

        runner.results.append(PhaseResult("extra", "/m2.zip", True, 20.0, 50))
        runner._save_progress()

        progress_path = os.path.join(run_dir, "curriculum_progress.json")
        with open(progress_path) as f:
            data = json.load(f)

        assert data["phases_completed"] == 2


# ---------------------------------------------------------------------------
# Tests: Peak reward evaluation
# ---------------------------------------------------------------------------

class TestEvalRewardCheck:
    """Test _check_advancement reads evaluations.npz correctly."""

    def test_advanced_when_above_threshold(self, tmp_path):
        from training.curriculum import CurriculumRunner

        # Create a fake evaluations.npz
        eval_dir = tmp_path / "eval_logs"
        eval_dir.mkdir()
        # results shape: (n_evaluations, n_eval_episodes)
        results = np.array([[18.0, 20.0], [22.0, 25.0]])  # means: 19.0, 23.5
        np.savez(eval_dir / "evaluations.npz", results=results)

        advanced, peak = CurriculumRunner._check_advancement(
            str(eval_dir), threshold=20.0
        )
        assert advanced is True
        assert peak == pytest.approx(23.5, abs=0.1)

    def test_not_advanced_when_below_threshold(self, tmp_path):
        eval_dir = tmp_path / "eval_logs"
        eval_dir.mkdir()
        results = np.array([[5.0, 8.0], [6.0, 7.0]])  # means: 6.5, 6.5
        np.savez(eval_dir / "evaluations.npz", results=results)

        advanced, peak = CurriculumRunner._check_advancement(
            str(eval_dir), threshold=20.0
        )
        assert advanced is False
        assert peak == pytest.approx(6.5, abs=0.1)

    def test_no_eval_file(self, tmp_path):
        eval_dir = tmp_path / "eval_logs"
        eval_dir.mkdir()
        # No evaluations.npz file

        advanced, peak = CurriculumRunner._check_advancement(
            str(eval_dir), threshold=20.0
        )
        assert advanced is False
        assert peak == 0.0

    def test_no_threshold_never_advances(self, tmp_path):
        eval_dir = tmp_path / "eval_logs"
        eval_dir.mkdir()
        results = np.array([[100.0]])
        np.savez(eval_dir / "evaluations.npz", results=results)

        advanced, peak = CurriculumRunner._check_advancement(
            str(eval_dir), threshold=None
        )
        assert advanced is False
        assert peak == pytest.approx(100.0, abs=0.1)

    def test_empty_results(self, tmp_path):
        eval_dir = tmp_path / "eval_logs"
        eval_dir.mkdir()
        results = np.array([]).reshape(0, 0)
        np.savez(eval_dir / "evaluations.npz", results=results)

        advanced, peak = CurriculumRunner._check_advancement(
            str(eval_dir), threshold=10.0
        )
        assert advanced is False
        assert peak == 0.0

    def test_single_evaluation(self, tmp_path):
        eval_dir = tmp_path / "eval_logs"
        eval_dir.mkdir()
        results = np.array([[25.0, 30.0]])  # mean = 27.5
        np.savez(eval_dir / "evaluations.npz", results=results)

        advanced, peak = CurriculumRunner._check_advancement(
            str(eval_dir), threshold=25.0
        )
        assert advanced is True
        assert peak == pytest.approx(27.5, abs=0.1)


# ---------------------------------------------------------------------------
# Tests: Best model selection
# ---------------------------------------------------------------------------

class TestBestModelSelection:
    """Test _select_best_model prefers best over final."""

    def test_prefers_best_model(self, tmp_path):
        phase_dir = tmp_path / "phase0_aim"
        best_dir = phase_dir / "best"
        best_dir.mkdir(parents=True)
        (best_dir / "best_model.zip").write_text("fake")
        (phase_dir / "final_model.zip").write_text("fake")

        path = CurriculumRunner._select_best_model(
            str(phase_dir), "/fallback.zip"
        )
        assert path == str(best_dir / "best_model.zip")

    def test_falls_back_to_final(self, tmp_path):
        phase_dir = tmp_path / "phase0_aim"
        phase_dir.mkdir(parents=True)
        # No best/ directory

        path = CurriculumRunner._select_best_model(
            str(phase_dir), "/fallback.zip"
        )
        assert path == "/fallback.zip"


# ---------------------------------------------------------------------------
# Tests: Full run orchestration (with mocked PPOTrainer)
# ---------------------------------------------------------------------------

class _MockPPOTrainer:
    """Mock PPOTrainer that records calls and produces fake model files."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.timesteps = kwargs.get("timesteps", 100)
        self.run_dir = kwargs.get("run_dir", "/tmp/mock_run")
        _MockPPOTrainer.instances.append(self)

    def train(self):
        # Create run_dir structure with eval logs
        os.makedirs(os.path.join(self.run_dir, "eval_logs"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "best"), exist_ok=True)

        # Write fake eval results (high reward to trigger advancement)
        results = np.array([[50.0, 60.0], [70.0, 80.0]])
        np.savez(
            os.path.join(self.run_dir, "eval_logs", "evaluations.npz"),
            results=results,
        )

        # Write fake best model
        best_path = os.path.join(self.run_dir, "best", "best_model.zip")
        with open(best_path, "w") as f:
            f.write("fake model")

        final_path = os.path.join(self.run_dir, "final_model.zip")
        with open(final_path, "w") as f:
            f.write("fake model")

        return final_path


class _MockPPOTrainerLowReward:
    """Mock PPOTrainer that produces low rewards (no advancement)."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.timesteps = kwargs.get("timesteps", 100)
        self.run_dir = kwargs.get("run_dir", "/tmp/mock_run")
        _MockPPOTrainerLowReward.instances.append(self)

    def train(self):
        os.makedirs(os.path.join(self.run_dir, "eval_logs"), exist_ok=True)

        results = np.array([[2.0, 3.0]])
        np.savez(
            os.path.join(self.run_dir, "eval_logs", "evaluations.npz"),
            results=results,
        )

        final_path = os.path.join(self.run_dir, "final_model.zip")
        with open(final_path, "w") as f:
            f.write("fake model")

        return final_path


class TestFullRun:
    """Integration tests for the full run() method with mocked trainer."""

    def setup_method(self):
        _MockPPOTrainer.instances = []
        _MockPPOTrainerLowReward.instances = []

    def test_runs_all_phases(self, no_threshold_yaml, tmp_path, monkeypatch):
        run_dir = str(tmp_path / "runs")
        runner = CurriculumRunner(no_threshold_yaml, run_dir=run_dir)

        # Monkey-patch the PPOTrainer import
        monkeypatch.setattr(
            "training.curriculum._import_ppo_trainer",
            lambda: _MockPPOTrainer,
        )

        final_model = runner.run()

        assert final_model is not None
        assert len(runner.results) == 2
        assert len(_MockPPOTrainer.instances) == 2

    def test_passes_correct_params_to_trainer(self, single_phase_yaml, tmp_path, monkeypatch):
        run_dir = str(tmp_path / "runs")
        runner = CurriculumRunner(single_phase_yaml, run_dir=run_dir)

        monkeypatch.setattr(
            "training.curriculum._import_ppo_trainer",
            lambda: _MockPPOTrainer,
        )

        runner.run()

        trainer = _MockPPOTrainer.instances[0]
        assert trainer.kwargs["scenario"] == "tactical"
        assert trainer.kwargs["config_path"] == "configs/tactical/phase1_open.json"
        assert trainer.kwargs["lr"] == 0.0003
        assert trainer.kwargs["n_envs"] == 2
        assert trainer.kwargs["timesteps"] == 100
        assert trainer.kwargs["phase"] == 1

    def test_weight_transfer_between_phases(self, no_threshold_yaml, tmp_path, monkeypatch):
        run_dir = str(tmp_path / "runs")
        runner = CurriculumRunner(no_threshold_yaml, run_dir=run_dir)

        monkeypatch.setattr(
            "training.curriculum._import_ppo_trainer",
            lambda: _MockPPOTrainer,
        )

        runner.run()

        # Phase 0 should have no resume
        assert _MockPPOTrainer.instances[0].kwargs["resume"] is None

        # Phase 1 should resume from phase 0's best model
        resume_path = _MockPPOTrainer.instances[1].kwargs["resume"]
        assert resume_path is not None
        assert "best_model.zip" in resume_path

    def test_start_phase_offset(self, arena3d_yaml, tmp_path, monkeypatch):
        run_dir = str(tmp_path / "runs")
        runner = CurriculumRunner(arena3d_yaml, run_dir=run_dir)

        monkeypatch.setattr(
            "training.curriculum._import_ppo_trainer",
            lambda: _MockPPOTrainer,
        )

        runner.run(start_phase=1)

        # Should only run phases 1 and 2 (tracking, combat)
        assert len(runner.results) == 2
        assert runner.results[0].phase_name == "tracking"
        assert runner.results[1].phase_name == "combat"

    def test_resume_model_passed_to_first_phase(self, single_phase_yaml, tmp_path, monkeypatch):
        run_dir = str(tmp_path / "runs")
        runner = CurriculumRunner(single_phase_yaml, run_dir=run_dir)

        monkeypatch.setattr(
            "training.curriculum._import_ppo_trainer",
            lambda: _MockPPOTrainer,
        )

        runner.run(resume_model="/some/pretrained.zip")

        # Even though phase has no resume_from, the explicit resume_model
        # should be passed when it's the first phase being run
        trainer = _MockPPOTrainer.instances[0]
        assert trainer.kwargs["resume"] == "/some/pretrained.zip"

    def test_advancement_detected(self, single_phase_yaml, tmp_path, monkeypatch):
        run_dir = str(tmp_path / "runs")
        runner = CurriculumRunner(single_phase_yaml, run_dir=run_dir)

        monkeypatch.setattr(
            "training.curriculum._import_ppo_trainer",
            lambda: _MockPPOTrainer,
        )

        runner.run()

        # MockPPOTrainer produces rewards of 50-80 mean, threshold is 10.0
        assert runner.results[0].advanced is True
        assert runner.results[0].peak_reward > 10.0

    def test_no_advancement_when_low_reward(self, single_phase_yaml, tmp_path, monkeypatch):
        run_dir = str(tmp_path / "runs")
        runner = CurriculumRunner(single_phase_yaml, run_dir=run_dir)

        monkeypatch.setattr(
            "training.curriculum._import_ppo_trainer",
            lambda: _MockPPOTrainerLowReward,
        )

        runner.run()

        # Low reward mock produces mean ~2.5, threshold is 10.0
        assert runner.results[0].advanced is False

    def test_progress_saved_after_each_phase(self, no_threshold_yaml, tmp_path, monkeypatch):
        run_dir = str(tmp_path / "runs")
        runner = CurriculumRunner(no_threshold_yaml, run_dir=run_dir)

        monkeypatch.setattr(
            "training.curriculum._import_ppo_trainer",
            lambda: _MockPPOTrainer,
        )

        runner.run()

        progress_path = os.path.join(run_dir, "curriculum_progress.json")
        assert os.path.exists(progress_path)

        with open(progress_path) as f:
            data = json.load(f)
        assert data["phases_completed"] == 2

    def test_self_play_passed_to_trainer(self, arena3d_yaml, tmp_path, monkeypatch):
        run_dir = str(tmp_path / "runs")
        runner = CurriculumRunner(arena3d_yaml, run_dir=run_dir)

        monkeypatch.setattr(
            "training.curriculum._import_ppo_trainer",
            lambda: _MockPPOTrainer,
        )

        runner.run()

        # Phase 2 (combat) has self_play=True
        combat_trainer = _MockPPOTrainer.instances[2]
        assert combat_trainer.kwargs["self_play"] is True
        assert combat_trainer.kwargs["swap_interval"] == 100
        assert combat_trainer.kwargs["scripted_warmup"] == 50

    def test_entropy_schedule_passed(self, single_phase_yaml, tmp_path, monkeypatch):
        # Modify the fixture data to include entropy
        with open(single_phase_yaml) as f:
            config = yaml.safe_load(f)
        config["phases"][0]["entropy_start"] = 0.02
        config["phases"][0]["entropy_end"] = 0.005
        with open(single_phase_yaml, "w") as f:
            yaml.dump(config, f)

        run_dir = str(tmp_path / "runs")
        runner = CurriculumRunner(single_phase_yaml, run_dir=run_dir)

        monkeypatch.setattr(
            "training.curriculum._import_ppo_trainer",
            lambda: _MockPPOTrainer,
        )

        runner.run()

        trainer = _MockPPOTrainer.instances[0]
        assert trainer.kwargs["entropy_schedule"] == (0.02, 0.005)

# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_phases_list(self, tmp_path):
        config = {"name": "empty", "scenario": "test", "phases": []}
        path = tmp_path / "empty.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        runner = CurriculumRunner(str(path), run_dir=str(tmp_path / "runs"))
        assert len(runner.phases) == 0

    def test_run_with_empty_phases(self, tmp_path, monkeypatch):
        config = {"name": "empty", "scenario": "test", "phases": []}
        path = tmp_path / "empty.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        runner = CurriculumRunner(str(path), run_dir=str(tmp_path / "runs"))

        monkeypatch.setattr(
            "training.curriculum._import_ppo_trainer",
            lambda: _MockPPOTrainer,
        )

        final = runner.run()
        assert final is None
        assert len(runner.results) == 0

    def test_start_phase_beyond_length(self, single_phase_yaml, tmp_path, monkeypatch):
        runner = CurriculumRunner(single_phase_yaml, run_dir=str(tmp_path / "runs"))

        monkeypatch.setattr(
            "training.curriculum._import_ppo_trainer",
            lambda: _MockPPOTrainer,
        )

        final = runner.run(start_phase=99)
        assert final is None
        assert len(runner.results) == 0

    def test_phase_dir_naming(self, arena3d_yaml, tmp_path, monkeypatch):
        run_dir = str(tmp_path / "runs")
        runner = CurriculumRunner(arena3d_yaml, run_dir=run_dir)

        monkeypatch.setattr(
            "training.curriculum._import_ppo_trainer",
            lambda: _MockPPOTrainer,
        )

        runner.run()

        # Check that phase directories were created with correct naming
        assert os.path.isdir(os.path.join(run_dir, "phase0_aim_targets"))
        assert os.path.isdir(os.path.join(run_dir, "phase1_tracking"))
        assert os.path.isdir(os.path.join(run_dir, "phase2_combat"))

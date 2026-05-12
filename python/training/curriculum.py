"""Auto-curriculum orchestrator for multi-phase training.

Reads a YAML curriculum config and runs phases sequentially,
auto-advancing when eval reward exceeds threshold. Each phase
can specify its own hyperparameters, config, and advancement
criteria.

The PPOTrainer dependency is imported lazily inside methods so
this module can be loaded and tested independently.

Usage
-----
::

    from training.curriculum import CurriculumRunner

    runner = CurriculumRunner("configs/arena3d/curriculum.yaml")
    final_model = runner.run()

YAML Config Format
------------------
::

    name: arena3d_full
    scenario: arena3d

    phases:
      - name: aim_targets
        config: configs/arena3d/phase1_aim_targets.json
        max_timesteps: 3000000
        advance_threshold: 15.0
        advance_patience: 3
        n_envs: 32
        lr: 0.0003
        ...
"""

import json
import os
from datetime import datetime

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Lazy PPOTrainer import -- allows testing without the full training stack.
# Tests can monkeypatch this function to inject a mock.
# ---------------------------------------------------------------------------

def _import_ppo_trainer():
    """Import and return the PPOTrainer class.

    Isolated into a function so tests can monkeypatch it with a mock
    trainer without needing the full training stack available.
    """
    from training.ppo_trainer import PPOTrainer
    return PPOTrainer


# ---------------------------------------------------------------------------
# PhaseResult
# ---------------------------------------------------------------------------

class PhaseResult:
    """Result from running a single curriculum phase.

    Attributes
    ----------
    phase_name : str
        Human-readable name of the phase (e.g. ``"aim_targets"``).
    model_path : str
        Path to the best (or final) model produced by this phase.
    advanced : bool
        Whether the phase's eval reward exceeded its advance threshold.
    peak_reward : float
        Highest mean eval reward observed during this phase.
    timesteps_used : int
        Total environment timesteps consumed by this phase.
    """

    def __init__(
        self,
        phase_name: str,
        model_path: str,
        advanced: bool,
        peak_reward: float,
        timesteps_used: int,
    ):
        self.phase_name = phase_name
        self.model_path = model_path
        self.advanced = advanced
        self.peak_reward = peak_reward
        self.timesteps_used = timesteps_used


# ---------------------------------------------------------------------------
# CurriculumRunner
# ---------------------------------------------------------------------------

class CurriculumRunner:
    """Run a multi-phase curriculum from a YAML config.

    Parameters
    ----------
    curriculum_path : str
        Path to the YAML curriculum config file.
    run_dir : str, optional
        Directory to store all phase outputs. If ``None``, a timestamped
        directory is created under ``runs/``.

    Raises
    ------
    FileNotFoundError
        If ``curriculum_path`` does not exist.
    yaml.YAMLError
        If the YAML file is malformed.
    KeyError
        If required fields (``name``, ``phases``) are missing.
    """

    def __init__(self, curriculum_path: str, run_dir: str | None = None):
        if not os.path.exists(curriculum_path):
            raise FileNotFoundError(
                f"Curriculum config not found: {curriculum_path}"
            )

        with open(curriculum_path) as f:
            self.config = yaml.safe_load(f)

        # Validate required fields -- raise KeyError early with clear context
        self.name = self.config["name"]
        self.scenario = self.config.get("scenario", "unknown")
        self.phases = self.config["phases"]

        if run_dir is not None:
            self.run_dir = run_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = f"runs/{self.name}_{timestamp}"

        os.makedirs(self.run_dir, exist_ok=True)
        self.results: list[PhaseResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        start_phase: int = 0,
        resume_model: str | None = None,
    ) -> str | None:
        """Run the full curriculum from ``start_phase``.

        Parameters
        ----------
        start_phase : int
            Zero-based index of the phase to start from.
        resume_model : str, optional
            Path to a model to resume from for the first phase being run.
            Overrides any ``resume_from`` setting in the YAML config for
            that phase.

        Returns
        -------
        str or None
            Path to the final model produced, or ``None`` if no phases
            were executed.
        """
        current_model = resume_model

        for i, phase_config in enumerate(self.phases[start_phase:], start=start_phase):
            print(f"\n{'='*60}")
            print(
                f"CURRICULUM: Phase {i+1}/{len(self.phases)} "
                f"-- {phase_config['name']}"
            )
            print(f"{'='*60}")

            # Determine resume model for this phase
            if i == start_phase and resume_model is not None:
                # Explicit resume_model overrides for the first phase
                resume = resume_model
            else:
                resume = self._resolve_resume_model(phase_config, current_model)

            result = self._run_phase(phase_config, resume_from=resume, phase_index=i)
            self.results.append(result)
            current_model = result.model_path

            if not result.advanced and phase_config.get("advance_threshold"):
                print(
                    f"Phase '{phase_config['name']}' did not advance "
                    f"(peak: {result.peak_reward:.2f}, "
                    f"threshold: {phase_config['advance_threshold']})"
                )
                # Continue to next phase anyway (max_timesteps reached)

            self._save_progress()

        # Summary
        print(f"\n{'='*60}")
        print(f"CURRICULUM COMPLETE: {len(self.results)} phases")
        for r in self.results:
            status = "ADVANCED" if r.advanced else "COMPLETED"
            print(
                f"  {r.phase_name}: {status} "
                f"(peak={r.peak_reward:.2f}, steps={r.timesteps_used:,})"
            )
        if current_model:
            print(f"Final model: {current_model}")
        print(f"{'='*60}")

        return current_model

    # ------------------------------------------------------------------
    # Internal: resume logic
    # ------------------------------------------------------------------

    def _resolve_resume_model(
        self,
        phase_config: dict,
        current_model: str | None,
    ) -> str | None:
        """Determine which model to resume from for a given phase.

        Logic:
        - If ``resume_from`` is ``"previous"`` and a current model exists,
          use the current model.
        - If ``resume_from`` is an explicit path (not ``"previous"``), use
          that path.
        - Otherwise, return ``None`` (train from scratch).

        Parameters
        ----------
        phase_config : dict
            The phase's configuration dict from the YAML.
        current_model : str or None
            Path to the model produced by the previous phase.

        Returns
        -------
        str or None
            Path to the model to resume from, or ``None``.
        """
        resume_from = phase_config.get("resume_from")

        if resume_from is None:
            return None

        if resume_from == "previous":
            return current_model  # May be None if no previous phase

        # Explicit path
        return resume_from

    # ------------------------------------------------------------------
    # Internal: run a single phase
    # ------------------------------------------------------------------

    def _run_phase(
        self,
        phase_config: dict,
        resume_from: str | None = None,
        phase_index: int = 0,
    ) -> PhaseResult:
        """Run a single curriculum phase.

        Constructs a PPOTrainer with the phase's hyperparameters, runs
        training, then checks whether the eval reward exceeded the
        advance threshold.

        Parameters
        ----------
        phase_config : dict
            Phase configuration from the YAML.
        resume_from : str or None
            Path to a model to resume from.
        phase_index : int
            Zero-based index of this phase in the curriculum.

        Returns
        -------
        PhaseResult
            Result object with model path, advancement status, and metrics.
        """
        PPOTrainer = _import_ppo_trainer()

        phase_dir = os.path.join(
            self.run_dir,
            f"phase{phase_index}_{phase_config['name']}",
        )

        # Build entropy schedule tuple if both start/end are specified
        entropy_schedule = None
        if "entropy_start" in phase_config and "entropy_end" in phase_config:
            entropy_schedule = (
                phase_config["entropy_start"],
                phase_config["entropy_end"],
            )

        # Handle BC warmstart (produces both SB3 model and KL reference)
        bc_ref = None
        if phase_config.get("bc_warmstart"):
            bc_ref, resume_from = self._do_bc_warmstart(
                phase_config, phase_dir
            )

        phase_scenario = phase_config.get("scenario", self.scenario)
        trainer = PPOTrainer(
            scenario=phase_scenario,
            config_path=phase_config["config"],
            name=f"{self.name}_p{phase_index}_{phase_config['name']}",
            lr=phase_config.get("lr", 3e-4),
            n_envs=phase_config.get("n_envs", 32),
            timesteps=phase_config.get("max_timesteps", 3_000_000),
            phase=phase_config.get("phase_mask"),
            resume=resume_from,
            kl_anchor=bc_ref if phase_config.get("kl_anchor") else None,
            kl_beta_start=phase_config.get("kl_beta_start", 0.5),
            kl_beta_end=phase_config.get("kl_beta_end", 0.0),
            kl_anneal_steps=phase_config.get("kl_anneal_steps", 2_000_000),
            entropy_schedule=entropy_schedule,
            self_play=phase_config.get("self_play", False),
            swap_interval=phase_config.get("swap_interval", 500_000),
            scripted_warmup=phase_config.get("scripted_warmup", 1_000_000),
            auto_stop=bool(phase_config.get("advance_threshold")),
            patience=phase_config.get("advance_patience", 15),
            run_dir=phase_dir,
            frame_skip=phase_config.get("frame_skip", 4),
        )

        final_model = trainer.train()

        # Check advancement
        eval_dir = os.path.join(phase_dir, "eval_logs")
        threshold = phase_config.get("advance_threshold")
        advanced, peak_reward = self._check_advancement(eval_dir, threshold)

        # Select best model (prefer best/ over final)
        model_path = self._select_best_model(phase_dir, final_model)

        return PhaseResult(
            phase_name=phase_config["name"],
            model_path=model_path,
            advanced=advanced,
            peak_reward=peak_reward,
            timesteps_used=trainer.timesteps,
        )

    # ------------------------------------------------------------------
    # Internal: BC warmstart
    # ------------------------------------------------------------------

    def _do_bc_warmstart(
        self,
        phase_config: dict,
        phase_dir: str,
    ) -> tuple[str, str]:
        """Run BC demo collection and pre-training.

        Parameters
        ----------
        phase_config : dict
            Phase configuration from the YAML.
        phase_dir : str
            Directory for this phase's outputs.

        Returns
        -------
        tuple[str, str]
            ``(bc_ref_path, model_path)`` -- the KL reference .pt file
            and the SB3 model .zip to resume from.
        """
        from training.bc_collector import collect_demonstrations
        from training.bc_pretrain import BCTrainer

        phase_scenario = phase_config.get("scenario", self.scenario)

        demos_path = phase_config.get("bc_demos")
        if not demos_path or not os.path.exists(demos_path):
            demos_path = os.path.join(phase_dir, "demos.npz")
            os.makedirs(os.path.dirname(demos_path), exist_ok=True)
            print(f"Collecting BC demonstrations -> {demos_path}")
            collect_demonstrations(
                config_path=phase_config["config"],
                scenario=phase_scenario,
                num_episodes=500,
                output_path=demos_path,
                phase=phase_config.get("phase_mask"),
            )

        # BC pre-train
        from training.ppo_trainer import _import_gym_class

        gym_class = _import_gym_class(phase_scenario)
        env = gym_class(
            config_path=phase_config["config"],
            scenario=phase_scenario,
            phase=phase_config.get("phase_mask"),
        )

        bc = BCTrainer(obs_dim=env.obs_size, act_dim=env.action_size)
        bc.train(demos_path, epochs=50)

        bc_model_path = os.path.join(phase_dir, "bc_model")
        bc.save_as_sb3(bc_model_path, env)

        bc_ref_path = os.path.join(phase_dir, "bc_ref.pt")
        bc.save_reference(bc_ref_path)

        env.close()

        return bc_ref_path, f"{bc_model_path}.zip"

    # ------------------------------------------------------------------
    # Internal: eval reward checking
    # ------------------------------------------------------------------

    @staticmethod
    def _check_advancement(
        eval_dir: str,
        threshold: float | None,
    ) -> tuple[bool, float]:
        """Check if eval reward exceeded the advance threshold.

        Reads the ``evaluations.npz`` file produced by SB3's
        ``EvalCallback`` and finds the peak mean reward.

        Parameters
        ----------
        eval_dir : str
            Directory containing ``evaluations.npz``.
        threshold : float or None
            Reward threshold for advancement. If ``None``, advancement
            is never triggered.

        Returns
        -------
        tuple[bool, float]
            ``(advanced, peak_reward)`` -- whether the threshold was
            exceeded, and the peak mean reward observed.
        """
        eval_path = os.path.join(eval_dir, "evaluations.npz")

        if not os.path.exists(eval_path):
            return False, 0.0

        try:
            data = np.load(eval_path)
            if "results" not in data:
                return False, 0.0
            results = data["results"]
            if results.size == 0:
                return False, 0.0
            rewards = [float(r.mean()) for r in results]
            peak_reward = max(rewards) if rewards else 0.0
        except Exception:
            return False, 0.0

        if threshold is not None and peak_reward >= threshold:
            return True, peak_reward

        return False, peak_reward

    # ------------------------------------------------------------------
    # Internal: best model selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_best_model(phase_dir: str, fallback: str) -> str:
        """Select the best model from a phase directory.

        Prefers ``{phase_dir}/best/best_model.zip`` if it exists,
        otherwise falls back to the provided path (typically the
        final model returned by the trainer).

        Parameters
        ----------
        phase_dir : str
            Directory for this phase's outputs.
        fallback : str
            Fallback model path if no best model exists.

        Returns
        -------
        str
            Path to the selected model.
        """
        best_path = os.path.join(phase_dir, "best", "best_model.zip")
        if os.path.exists(best_path):
            return best_path
        return fallback

    # ------------------------------------------------------------------
    # Internal: progress persistence
    # ------------------------------------------------------------------

    def _save_progress(self) -> None:
        """Save curriculum progress to a JSON file in the run directory.

        The progress file contains the curriculum name, number of
        completed phases, and details for each phase result (name,
        model path, advancement status, peak reward, timesteps).
        """
        progress = {
            "name": self.name,
            "phases_completed": len(self.results),
            "results": [
                {
                    "name": r.phase_name,
                    "model": r.model_path,
                    "advanced": r.advanced,
                    "peak_reward": r.peak_reward,
                    "timesteps": r.timesteps_used,
                }
                for r in self.results
            ],
        }

        path = os.path.join(self.run_dir, "curriculum_progress.json")
        with open(path, "w") as f:
            json.dump(progress, f, indent=2)

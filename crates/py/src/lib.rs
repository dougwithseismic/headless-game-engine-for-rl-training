use std::collections::HashMap;
use std::path::Path;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use ghostlobby_engine::config::GameConfig;
use ghostlobby_engine::scenario::Scenario;
use ghostlobby_engine::scenarios::cs_lite::CsLiteScenario;
use ghostlobby_engine::scenarios::tactical_deathmatch::TacticalDeathmatchScenario;
use ghostlobby_engine::tick::TickRunner;

fn make_scenario(name: &str) -> PyResult<Box<dyn Scenario>> {
    match name {
        "cs_lite" | "cs-lite" | "counterstrike" => Ok(Box::new(CsLiteScenario::default())),
        "cs_lite_dummy" | "cs-lite-dummy" => Ok(Box::new(CsLiteScenario { dummy_ai: true, ..Default::default() })),
        "tactical" | "tactical-deathmatch" => Ok(Box::new(TacticalDeathmatchScenario)),
        other => Err(PyValueError::new_err(format!("unknown scenario: {other}"))),
    }
}

fn detect_scenario(title: &str) -> &str {
    if title.contains("cs_lite") || title.contains("cs-lite") || title.contains("counterstrike") {
        "cs_lite"
    } else if title.contains("tactical") {
        "tactical"
    } else {
        "cs_lite"
    }
}

#[pyclass]
struct GhostLobbyEnv {
    runner: TickRunner,
    scenario_name: String,
    step_mul: Option<u32>,
}

#[pymethods]
impl GhostLobbyEnv {
    #[new]
    #[pyo3(signature = (config_path, scenario=None, step_mul=None))]
    fn new(config_path: &str, scenario: Option<&str>, step_mul: Option<u32>) -> PyResult<Self> {
        let config = GameConfig::from_file(config_path)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let scenario_name = scenario
            .unwrap_or_else(|| detect_scenario(&config.title))
            .to_string();

        let scenario_obj = make_scenario(&scenario_name)?;
        let runner = TickRunner::builder(config)
            .with_scenario_boxed(scenario_obj)
            .build();

        Ok(Self {
            runner,
            scenario_name,
            step_mul,
        })
    }

    fn reset<'py>(&mut self, py: Python<'py>) -> PyResult<(PyObject, PyObject)> {
        let scenario = make_scenario(&self.scenario_name)?;
        // step_mul is preserved across resets (stored on the struct, not the runner)
        self.runner = TickRunner::builder(self.runner.config().clone())
            .with_scenario_boxed(scenario)
            .build();

        let obs = self.build_obs(py)?;
        let infos = self.build_empty_infos(py)?;
        Ok((obs.into(), infos.into()))
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: HashMap<usize, Vec<f32>>,
    ) -> PyResult<(PyObject, PyObject, PyObject, PyObject, PyObject)> {
        let agents = &self.runner.agent_registry().agents;

        let mut engine_actions = HashMap::new();
        for (idx, action_vec) in actions {
            if idx < agents.len() {
                engine_actions.insert(agents[idx], action_vec);
            }
        }

        self.runner.apply_raw_actions(engine_actions);
        self.runner.tick();

        let obs = self.build_obs(py)?;
        let rewards = self.build_rewards(py)?;
        let terminated = self.build_dones(py)?;
        let truncated = self.build_truncated(py)?;
        let infos = self.build_empty_infos(py)?;

        Ok((
            obs.into(),
            rewards.into(),
            terminated.into(),
            truncated.into(),
            infos.into(),
        ))
    }

    /// Step the simulation using `step_mul` repeated ticks (frame-skipping).
    /// Actions are repeated for each sub-tick, rewards are summed, and
    /// observations come from the final tick. Uses the `step_mul` value
    /// passed to the constructor, or falls back to `config.step_mul`, or 1.
    fn step_mul<'py>(
        &mut self,
        py: Python<'py>,
        actions: HashMap<usize, Vec<f32>>,
    ) -> PyResult<(PyObject, PyObject, PyObject, PyObject, PyObject)> {
        let agents = &self.runner.agent_registry().agents;

        let mut engine_actions = HashMap::new();
        for (idx, action_vec) in actions {
            if idx < agents.len() {
                engine_actions.insert(agents[idx], action_vec);
            }
        }

        self.runner.apply_raw_actions(engine_actions);

        // Determine effective step_mul: constructor param > config > 1
        let mul = self
            .step_mul
            .or(self.runner.config().step_mul)
            .unwrap_or(1);
        self.runner.step_with_mul(mul);

        let obs = self.build_obs(py)?;
        let rewards = self.build_rewards(py)?;
        let terminated = self.build_dones(py)?;
        let truncated = self.build_truncated(py)?;
        let infos = self.build_empty_infos(py)?;

        Ok((
            obs.into(),
            rewards.into(),
            terminated.into(),
            truncated.into(),
            infos.into(),
        ))
    }

    /// Returns the new-style ActionSpace (function-call model) as a Python dict.
    fn new_action_space<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let space = self.runner.action_space();
        let json = serde_json::to_string(space)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let json_mod = py.import("json")?;
        let result = json_mod.call_method1("loads", (json,))?;
        Ok(result.into())
    }

    /// Get the current step_mul value (None if not set).
    fn get_step_mul(&self) -> Option<u32> {
        self.step_mul
    }

    /// Set the step_mul value at runtime.
    fn set_step_mul(&mut self, value: Option<u32>) {
        self.step_mul = value;
    }

    fn action_space<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let space = self.runner.action_space_def();
        let json = serde_json::to_string(space)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let json_mod = py.import("json")?;
        let result = json_mod.call_method1("loads", (json,))?;
        Ok(result.into())
    }

    fn observation_space<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let space = self.runner.observation_space_def();
        let json = serde_json::to_string(space)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let json_mod = py.import("json")?;
        let result = json_mod.call_method1("loads", (json,))?;
        Ok(result.into())
    }

    fn agents(&self) -> Vec<usize> {
        let registry = self.runner.agent_registry();
        (0..registry.agents.len()).collect()
    }

    fn agent_entity_id(&self, index: usize) -> Option<u64> {
        let registry = self.runner.agent_registry();
        registry.agents.get(index).map(|e| e.to_bits())
    }

    fn num_agents(&self) -> usize {
        self.runner.agent_registry().agents.len()
    }

    fn tick_count(&self) -> u64 {
        self.runner.tick_count()
    }

    fn get_actions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let actions = self.runner.raw_actions();
        let dict = PyDict::new(py);
        for (idx, action_vec) in actions {
            dict.set_item(idx, action_vec)?;
        }
        Ok(dict)
    }

    fn reward_breakdown<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let breakdown = self.runner.reward_breakdown();
        let dict = PyDict::new(py);
        for (idx, components) in breakdown {
            let inner = PyDict::new(py);
            for (category, value) in components {
                inner.set_item(category, value)?;
            }
            dict.set_item(idx, inner)?;
        }
        Ok(dict)
    }

    fn drain_telemetry(&mut self) -> PyResult<Vec<String>> {
        let events = self.runner.drain_telemetry();
        events
            .iter()
            .map(|e| serde_json::to_string(e).map_err(|e| PyValueError::new_err(e.to_string())))
            .collect()
    }

    // --- Replay recording ---

    /// Start recording replay frames. Clears any previously captured frames.
    fn start_recording(&mut self) -> PyResult<()> {
        self.runner.start_recording();
        Ok(())
    }

    /// Stop recording replay frames. Frames are retained for retrieval.
    fn stop_recording(&mut self) -> PyResult<()> {
        self.runner.stop_recording();
        Ok(())
    }

    /// Returns True if the replay recorder is actively capturing frames.
    fn is_recording(&self) -> bool {
        self.runner.is_recording()
    }

    /// Write all recorded frames to a JSON Lines file at the given path.
    fn save_replay(&self, path: &str) -> PyResult<()> {
        self.runner
            .save_replay(Path::new(path))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Drain all recorded replay frames as a list of dicts.
    /// Each dict has keys: tick, agent_obs, agent_actions, agent_rewards, agent_dones.
    fn replay_frames<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let frames = self.runner.drain_replay_frames();
        let list = PyList::empty(py);
        for frame in frames {
            let json = serde_json::to_string(&frame)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let json_mod = py.import("json")?;
            let dict = json_mod.call_method1("loads", (json,))?;
            list.append(dict)?;
        }
        Ok(list)
    }
}

impl GhostLobbyEnv {
    fn build_obs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let all_obs = self.runner.observe_all();
        let dict = PyDict::new(py);

        for (agent_idx, features) in all_obs {
            let agent_dict = PyDict::new(py);
            for (name, values) in features {
                agent_dict.set_item(name, values)?;
            }
            dict.set_item(agent_idx, agent_dict)?;
        }

        Ok(dict)
    }

    fn build_rewards<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let rewards = self.runner.rewards();
        let dict = PyDict::new(py);
        for (idx, reward) in rewards {
            dict.set_item(idx, reward)?;
        }
        Ok(dict)
    }

    fn build_dones<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dones = self.runner.dones();
        let dict = PyDict::new(py);
        for (idx, done) in dones {
            dict.set_item(idx, done)?;
        }
        Ok(dict)
    }

    fn build_truncated<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let registry = self.runner.agent_registry();
        for i in 0..registry.agents.len() {
            dict.set_item(i, false)?;
        }
        Ok(dict)
    }

    fn build_empty_infos<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let registry = self.runner.agent_registry();
        for i in 0..registry.agents.len() {
            let info = PyDict::new(py);
            dict.set_item(i, info)?;
        }
        Ok(dict)
    }
}

#[pymodule]
fn ghostlobby(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GhostLobbyEnv>()?;
    Ok(())
}

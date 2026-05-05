use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use ghostlobby_engine::config::GameConfig;
use ghostlobby_engine::scenario::{DeathmatchScenario, Scenario};
use ghostlobby_engine::scenarios::moba_lane::MobaLaneScenario;
use ghostlobby_engine::scenarios::racing::RacingScenario;
use ghostlobby_engine::tick::TickRunner;

fn make_scenario(name: &str) -> PyResult<Box<dyn Scenario>> {
    match name {
        "deathmatch" | "fps-deathmatch" | "fps" => Ok(Box::new(DeathmatchScenario)),
        "moba-lane" | "moba" | "lane" => Ok(Box::new(MobaLaneScenario)),
        "racing" | "race" | "oval-race" => Ok(Box::new(RacingScenario)),
        other => Err(PyValueError::new_err(format!("unknown scenario: {other}"))),
    }
}

fn detect_scenario(title: &str) -> &str {
    if title.contains("race") || title.contains("racing") {
        "racing"
    } else if title.contains("moba") || title.contains("lane") {
        "moba-lane"
    } else {
        "deathmatch"
    }
}

#[pyclass]
struct GhostLobbyEnv {
    runner: TickRunner,
    config: GameConfig,
    scenario_name: String,
}

#[pymethods]
impl GhostLobbyEnv {
    #[new]
    #[pyo3(signature = (config_path, scenario=None))]
    fn new(config_path: &str, scenario: Option<&str>) -> PyResult<Self> {
        let config = GameConfig::from_file(config_path)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let scenario_name = scenario
            .unwrap_or_else(|| detect_scenario(&config.title))
            .to_string();

        let scenario_obj = make_scenario(&scenario_name)?;
        let runner = TickRunner::builder(config.clone())
            .with_scenario_boxed(scenario_obj)
            .build();

        Ok(Self {
            runner,
            config,
            scenario_name,
        })
    }

    fn reset<'py>(&mut self, py: Python<'py>) -> PyResult<(PyObject, PyObject)> {
        let scenario = make_scenario(&self.scenario_name)?;
        self.runner = TickRunner::builder(self.config.clone())
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
        let registry = self.runner.agent_registry().clone();

        let mut engine_actions = HashMap::new();
        for (idx, action_vec) in actions {
            if idx < registry.agents.len() {
                engine_actions.insert(registry.agents[idx], action_vec);
            }
        }

        self.runner.apply_raw_actions(engine_actions);
        self.runner.tick();
        let _ = self.runner.drain_telemetry();

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

    fn num_agents(&self) -> usize {
        self.runner.agent_registry().agents.len()
    }

    fn tick_count(&self) -> u64 {
        self.runner.tick_count()
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

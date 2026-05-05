use std::sync::Mutex;

use bevy_ecs::prelude::*;

use crate::action_space::ActionSpaceDef;
use crate::config::GameConfig;
use crate::observation::{ObsWriter, ObservationSpaceDef};
use crate::physics::PhysicsState;
use crate::scenario::{DeathmatchScenario, Scenario};
use crate::tick::TickRunner;

type SystemConfigurator = Box<dyn FnOnce(&mut Schedule)>;
type ScenarioSetupFn =
    Mutex<Option<Box<dyn FnOnce(&mut World, &GameConfig, &mut PhysicsState) + Send + Sync>>>;

pub struct EngineBuilder {
    pub(crate) config: GameConfig,
    pub(crate) custom_systems: Vec<SystemConfigurator>,
    pub(crate) scenario: Box<dyn Scenario>,
}

impl EngineBuilder {
    pub fn new(config: GameConfig) -> Self {
        Self {
            config,
            custom_systems: Vec::new(),
            scenario: Box::new(DeathmatchScenario),
        }
    }

    pub fn add_system<M>(
        mut self,
        system: impl IntoScheduleConfigs<bevy_ecs::system::ScheduleSystem, M> + 'static,
    ) -> Self {
        self.custom_systems
            .push(Box::new(move |schedule: &mut Schedule| {
                schedule.add_systems(system);
            }));
        self
    }

    pub fn with_scenario(mut self, scenario: impl Scenario + 'static) -> Self {
        self.scenario = Box::new(scenario);
        self
    }

    pub fn with_scenario_boxed(mut self, scenario: Box<dyn Scenario>) -> Self {
        self.scenario = scenario;
        self
    }

    pub fn with_scenario_fn(
        mut self,
        setup: impl FnOnce(&mut World, &GameConfig, &mut PhysicsState) + Send + Sync + 'static,
    ) -> Self {
        self.scenario = Box::new(ClosureScenario {
            name: "custom".into(),
            inner: Mutex::new(Some(Box::new(setup))),
        });
        self
    }

    pub fn build(self) -> TickRunner {
        TickRunner::from_builder(self)
    }
}

struct ClosureScenario {
    name: String,
    inner: ScenarioSetupFn,
}

impl Scenario for ClosureScenario {
    fn name(&self) -> &str {
        &self.name
    }

    fn action_space(&self, _config: &GameConfig) -> ActionSpaceDef {
        ActionSpaceDef::new(vec![])
    }

    fn observation_space(&self, _config: &GameConfig) -> ObservationSpaceDef {
        ObservationSpaceDef { features: vec![] }
    }

    fn setup(&self, world: &mut World, config: &GameConfig, physics: &mut PhysicsState) {
        let setup = self
            .inner
            .lock()
            .expect("ClosureScenario mutex poisoned")
            .take()
            .expect("ClosureScenario::setup called more than once");
        setup(world, config, physics);
    }

    fn register_systems(&self, _schedule: &mut Schedule) {}

    fn observe(&self, _world: &World, _agent: Entity, _writer: &mut ObsWriter) {}
}

use std::sync::Mutex;

use bevy_ecs::prelude::*;

use crate::action_space::ActionSpaceDef;
use crate::actions::ActionTranslator;
use crate::config::GameConfig;
use crate::converters::ObsConverter;
use crate::features::{ExtractorSet, FeatureExtractor};
use crate::game_def::GameDef;
use crate::observation::{ObsWriter, ObservationSpaceDef};
use crate::physics::PhysicsState;
use crate::rewards::RewardFn;
use crate::scenario::Scenario;
use crate::scenarios::cs_lite::CsLiteScenario;
use crate::strategy::{StrategyBridge, StrategyProvider};
use crate::tick::TickRunner;

type SystemConfigurator = Box<dyn FnOnce(&mut Schedule)>;
type ScenarioSetupFn =
    Mutex<Option<Box<dyn FnOnce(&mut World, &GameConfig, &mut PhysicsState) + Send + Sync>>>;

/// What the builder is configured to run: a legacy Scenario or a new GameDef.
pub(crate) enum ScenarioOrGameDef {
    Scenario(Box<dyn Scenario>),
    GameDef(Box<dyn GameDef>),
}

pub struct EngineBuilder {
    pub(crate) config: GameConfig,
    pub(crate) custom_systems: Vec<SystemConfigurator>,
    pub(crate) inner: ScenarioOrGameDef,
    pub(crate) action_translator: Option<Box<dyn ActionTranslator>>,
    pub(crate) extractors: Option<ExtractorSet>,
    pub(crate) converter: Option<Box<dyn ObsConverter>>,
    pub(crate) reward_fn: Option<Box<dyn RewardFn>>,
    #[allow(clippy::type_complexity)]
    pub(crate) strategy: Option<(Box<dyn StrategyBridge>, Vec<Box<dyn StrategyProvider>>)>,
}

impl EngineBuilder {
    pub fn new(config: GameConfig) -> Self {
        Self {
            config,
            custom_systems: Vec::new(),
            inner: ScenarioOrGameDef::Scenario(Box::new(CsLiteScenario::default())),
            action_translator: None,
            extractors: None,
            converter: None,
            reward_fn: None,
            strategy: None,
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
        self.inner = ScenarioOrGameDef::Scenario(Box::new(scenario));
        self
    }

    pub fn with_scenario_boxed(mut self, scenario: Box<dyn Scenario>) -> Self {
        self.inner = ScenarioOrGameDef::Scenario(scenario);
        self
    }

    pub fn with_scenario_fn(
        mut self,
        setup: impl FnOnce(&mut World, &GameConfig, &mut PhysicsState) + Send + Sync + 'static,
    ) -> Self {
        self.inner = ScenarioOrGameDef::Scenario(Box::new(ClosureScenario {
            name: "custom".into(),
            inner: Mutex::new(Some(Box::new(setup))),
        }));
        self
    }

    /// Use a `GameDef` instead of a legacy `Scenario`.
    ///
    /// The `GameDef` path provides stub implementations for observation/reward
    /// (those will be composed separately in later phases). The engine handles
    /// physics setup based on `GameDef::needs_physics()`.
    pub fn with_game_def(mut self, game_def: impl GameDef + 'static) -> Self {
        self.inner = ScenarioOrGameDef::GameDef(Box::new(game_def));
        self
    }

    /// Use a boxed `GameDef` instead of a legacy `Scenario`.
    pub fn with_game_def_boxed(mut self, game_def: Box<dyn GameDef>) -> Self {
        self.inner = ScenarioOrGameDef::GameDef(game_def);
        self
    }

    /// Set an `ActionTranslator` to transform external actions before they
    /// reach the engine's `RawActionBuffer`. If not set, actions pass through
    /// as-is (equivalent to using `FlatTranslator`).
    pub fn with_action_translator(mut self, translator: impl ActionTranslator + 'static) -> Self {
        self.action_translator = Some(Box::new(translator));
        self
    }

    /// Set a complete `ExtractorSet` for composable observation extraction.
    /// Only used with the `GameDef` path; ignored for legacy `Scenario`.
    pub fn with_extractors(mut self, extractors: ExtractorSet) -> Self {
        self.extractors = Some(extractors);
        self
    }

    /// Add a single feature extractor. Builds the `ExtractorSet` incrementally.
    /// Only used with the `GameDef` path; ignored for legacy `Scenario`.
    pub fn with_extractor(mut self, extractor: impl FeatureExtractor + 'static) -> Self {
        let set = self.extractors.take().unwrap_or_default();
        self.extractors = Some(set.add(extractor));
        self
    }

    /// Set an observation converter to transform raw extractor output.
    /// Only used with the `GameDef` path; ignored for legacy `Scenario`.
    /// If not set, `DictConverter` (pass-through) is used by default.
    pub fn with_converter(mut self, converter: impl ObsConverter + 'static) -> Self {
        self.converter = Some(Box::new(converter));
        self
    }

    /// Set a composable reward function for the `GameDef` path.
    /// Only used with the `GameDef` path; ignored for legacy `Scenario`.
    pub fn with_reward(mut self, reward: impl RewardFn + 'static) -> Self {
        self.reward_fn = Some(Box::new(reward));
        self
    }

    /// Set a boxed reward function for the `GameDef` path.
    /// Only used with the `GameDef` path; ignored for legacy `Scenario`.
    pub fn with_reward_boxed(mut self, reward: Box<dyn RewardFn>) -> Self {
        self.reward_fn = Some(reward);
        self
    }

    pub fn with_strategy(
        mut self,
        bridge: impl StrategyBridge + 'static,
        providers: Vec<Box<dyn StrategyProvider>>,
    ) -> Self {
        self.strategy = Some((Box::new(bridge), providers));
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

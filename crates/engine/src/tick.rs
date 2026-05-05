use std::collections::HashMap;

use bevy_ecs::prelude::*;
use glam::Vec2;

use crate::action_space::{ActionDict, ActionMaskBuffer, ActionSpaceDef, RawActionBuffer};
use crate::builder::EngineBuilder;
use crate::config::GameConfig;
use crate::ecs::components::Agent;
use crate::ecs::resources::*;
use crate::ecs::systems;
use crate::observation::{AgentRegistry, ObsWriter, ObservationSpaceDef, RewardBuffer};
use crate::physics::PhysicsState;
use crate::scenario::{self, Scenario};
use crate::scripted_ai;
use crate::telemetry::TelemetryEvent;

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum EnginePhase {
    ClearBuffers,
    AiDecisions,
    PrePhysics,
    PhysicsStep,
    PostPhysics,
    GameLogic,
    StateTransitions,
    Telemetry,
}

#[derive(Debug, Clone)]
pub enum TickMode {
    Uncapped,
    RealTime { rate: u32 },
    Stepped,
}

pub struct TickRunner {
    world: World,
    schedule: Schedule,
    mode: TickMode,
    scenario: Box<dyn Scenario>,
}

impl TickRunner {
    pub fn new(config: GameConfig) -> Self {
        let scenario = Box::new(scenario::DeathmatchScenario) as Box<dyn Scenario>;
        Self::build_from_scenario(config, scenario)
    }

    pub fn builder(config: GameConfig) -> EngineBuilder {
        EngineBuilder::new(config)
    }

    pub(crate) fn from_builder(builder: EngineBuilder) -> Self {
        let mut runner = Self::build_from_scenario(builder.config, builder.scenario);
        for add_fn in builder.custom_systems {
            add_fn(&mut runner.schedule);
        }
        runner
    }

    fn build_from_scenario(config: GameConfig, scenario: Box<dyn Scenario>) -> Self {
        let dt = 1.0 / config.tick_rate as f32;
        let mut physics = PhysicsState::new(Vec2::ZERO, dt);
        let mut world = World::new();

        scenario.setup(&mut world, &config, &mut physics);
        world.insert_resource(physics);
        world.insert_resource(scenario.action_space(&config));
        world.insert_resource(scenario.observation_space(&config));
        world.insert_resource(RawActionBuffer::default());
        world.insert_resource(ActionMaskBuffer::default());
        world.insert_resource(RewardBuffer::default());

        let agents = Self::collect_agents(&mut world);
        let max_agents = agents.len();
        world.insert_resource(AgentRegistry::new(agents, max_agents));

        let mut schedule = Schedule::default();
        Self::configure_phases(&mut schedule);
        Self::add_core_systems(&mut schedule);
        scenario.register_systems(&mut schedule);

        Self {
            world,
            schedule,
            mode: TickMode::Uncapped,
            scenario,
        }
    }

    fn collect_agents(world: &mut World) -> Vec<Entity> {
        let mut agents: Vec<(u32, Entity)> = world
            .query::<(Entity, &Agent)>()
            .iter(world)
            .map(|(e, a)| (a.source_id, e))
            .collect();
        agents.sort_by_key(|(id, _)| *id);
        agents.into_iter().map(|(_, e)| e).collect()
    }

    fn configure_phases(schedule: &mut Schedule) {
        schedule.configure_sets(
            (
                EnginePhase::ClearBuffers,
                EnginePhase::AiDecisions,
                EnginePhase::PrePhysics,
                EnginePhase::PhysicsStep,
                EnginePhase::PostPhysics,
                EnginePhase::GameLogic,
                EnginePhase::StateTransitions,
                EnginePhase::Telemetry,
            )
                .chain(),
        );
    }

    fn add_core_systems(schedule: &mut Schedule) {
        schedule.add_systems(systems::clear_buffers.in_set(EnginePhase::ClearBuffers));
        schedule.add_systems(scripted_ai::run_scripted_ai.in_set(EnginePhase::AiDecisions));
        schedule.add_systems(
            systems::sync_actions_to_physics.in_set(EnginePhase::PrePhysics),
        );
        schedule.add_systems(
            (systems::physics_step, systems::sync_physics_to_ecs)
                .chain()
                .in_set(EnginePhase::PhysicsStep),
        );
        schedule.add_systems(
            systems::telemetry_snapshot_system.in_set(EnginePhase::Telemetry),
        );
    }

    pub fn set_mode(&mut self, mode: TickMode) {
        self.mode = mode;
    }

    pub fn mode(&self) -> &TickMode {
        &self.mode
    }

    pub fn tick(&mut self) {
        self.schedule.run(&mut self.world);
        let mut tick_state = self.world.resource_mut::<TickState>();
        tick_state.tick += 1;
    }

    pub fn drain_telemetry(&mut self) -> Vec<TelemetryEvent> {
        self.world.resource_mut::<TelemetryBuffer>().drain()
    }

    pub fn apply_raw_actions(&mut self, actions: HashMap<Entity, ActionDict>) {
        let mut buffer = self.world.resource_mut::<RawActionBuffer>();
        for (entity, action) in actions {
            buffer.insert(entity, action);
        }
    }

    pub fn observe_all(&self) -> HashMap<usize, HashMap<String, Vec<f32>>> {
        let registry = self.world.resource::<AgentRegistry>();
        let mut result = HashMap::new();
        for (idx, &entity) in registry.agents.iter().enumerate() {
            let mut writer = ObsWriter::new();
            self.scenario.observe(&self.world, entity, &mut writer);
            result.insert(idx, writer.buffers);
        }
        result
    }

    pub fn rewards(&self) -> HashMap<usize, f32> {
        let registry = self.world.resource::<AgentRegistry>();
        let mut result = HashMap::new();
        for (idx, &entity) in registry.agents.iter().enumerate() {
            result.insert(idx, self.scenario.reward(&self.world, entity));
        }
        result
    }

    pub fn dones(&self) -> HashMap<usize, bool> {
        let registry = self.world.resource::<AgentRegistry>();
        let mut result = HashMap::new();
        for (idx, &entity) in registry.agents.iter().enumerate() {
            result.insert(idx, self.scenario.is_done(&self.world, entity));
        }
        result
    }

    pub fn action_space_def(&self) -> &ActionSpaceDef {
        self.world.resource::<ActionSpaceDef>()
    }

    pub fn observation_space_def(&self) -> &ObservationSpaceDef {
        self.world.resource::<ObservationSpaceDef>()
    }

    pub fn agent_registry(&self) -> &AgentRegistry {
        self.world.resource::<AgentRegistry>()
    }

    pub fn tick_count(&self) -> u64 {
        self.world.resource::<TickState>().tick
    }

    pub fn config(&self) -> &GameConfig {
        &self.world.resource::<GameConfigResource>().0
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }
}

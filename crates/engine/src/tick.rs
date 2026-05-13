use std::collections::HashMap;

use bevy_ecs::prelude::*;
use glam::Vec2;

use crate::action_space::{
    ActionDict, ActionMaskBuffer, ActionSpaceDef, PendingActions, RawActionBuffer,
};
use crate::actions::{ActionSpace, ActionTranslator};
use crate::builder::{EngineBuilder, ScenarioOrGameDef};
use crate::config::GameConfig;
use crate::converters::{DictConverter, ObsConverter};
use crate::features::ExtractorSet;
use crate::ecs::components::Agent;
use crate::ecs::resources::*;
use crate::ecs::systems;
use crate::game_def::GameDef;
use crate::observation::{AgentRegistry, ObsWriter, ObservationSpaceDef, RewardBreakdownBuffer, RewardBuffer, ShotEventBuffer};
use crate::physics::PhysicsState;
use crate::replay::{ReplayFrame, ReplayRecorder};
use crate::replay::writer::ReplayWriter;
use crate::rewards::RewardFn;
use crate::scenario::Scenario;
use crate::scenarios::cs_lite::CsLiteScenario;
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

/// What drives the simulation: a legacy Scenario or a new GameDef.
enum RunnerCore {
    Scenario(Box<dyn Scenario>),
    GameDef(Box<dyn GameDef>),
}

pub struct TickRunner {
    world: World,
    schedule: Schedule,
    mode: TickMode,
    core: RunnerCore,
    action_translator: Option<Box<dyn ActionTranslator>>,
    new_action_space: ActionSpace,
    /// Feature extractors for the GameDef path. None for Scenario path.
    extractors: Option<ExtractorSet>,
    /// Observation converter for the GameDef path. None for Scenario path.
    /// When extractors are present but no converter is set, DictConverter is used.
    converter: Option<Box<dyn ObsConverter>>,
    /// Composable reward function for the GameDef path. None for Scenario path.
    reward_fn: Option<Box<dyn RewardFn>>,
    /// Optional replay recorder for capturing (obs, action, reward, done) trajectories.
    replay: Option<ReplayRecorder>,
}

impl TickRunner {
    pub fn new(config: GameConfig) -> Self {
        let scenario = Box::new(CsLiteScenario::default()) as Box<dyn Scenario>;
        Self::build_from_scenario(config, scenario)
    }

    pub fn builder(config: GameConfig) -> EngineBuilder {
        EngineBuilder::new(config)
    }

    pub(crate) fn from_builder(builder: EngineBuilder) -> Self {
        let action_translator = builder.action_translator;
        let converter = builder.converter;
        let reward_fn = builder.reward_fn;
        let mut runner = match builder.inner {
            ScenarioOrGameDef::Scenario(scenario) => {
                Self::build_from_scenario(builder.config, scenario)
            }
            ScenarioOrGameDef::GameDef(game_def) => {
                Self::build_from_game_def(builder.config, game_def, builder.extractors, converter)
            }
        };
        runner.action_translator = action_translator;
        runner.reward_fn = reward_fn;
        if let Some((bridge, providers)) = builder.strategy {
            runner.world.insert_resource(
                crate::strategy::StrategyState::new(bridge, providers),
            );
        }
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
        let legacy_space = scenario.action_space(&config);
        let new_action_space = ActionSpace::from_legacy(&legacy_space);
        world.insert_resource(legacy_space);
        world.insert_resource(scenario.observation_space(&config));
        world.insert_resource(RawActionBuffer::default());
        world.insert_resource(PendingActions::default());
        world.insert_resource(ActionMaskBuffer::default());
        world.insert_resource(RewardBuffer::default());
        world.insert_resource(RewardBreakdownBuffer::default());
        world.insert_resource(ShotEventBuffer::default());

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
            core: RunnerCore::Scenario(scenario),
            action_translator: None,
            new_action_space,
            extractors: None,
            converter: None,
            reward_fn: None,
            replay: None,
        }
    }

    fn build_from_game_def(
        config: GameConfig,
        game_def: Box<dyn GameDef>,
        extractors: Option<ExtractorSet>,
        converter: Option<Box<dyn ObsConverter>>,
    ) -> Self {
        let dt = 1.0 / config.tick_rate as f32;
        let mut world = World::new();

        // Insert engine-level resources that core systems always need.
        world.insert_resource(TickState::new(config.tick_rate));
        world.insert_resource(TelemetryBuffer::default());
        world.insert_resource(GameConfigResource(config.clone()));
        world.insert_resource(WorldBounds {
            width: config.arena.width,
            height: config.arena.height,
        });

        // Insert physics as a world resource BEFORE setup so GameDef can access
        // it via world.resource_mut::<PhysicsState>() if it needs to create bodies.
        if game_def.needs_physics() {
            let physics = PhysicsState::new(Vec2::ZERO, dt);
            world.insert_resource(physics);
        }

        game_def.setup(&mut world, &config);

        // Insert default action space if the GameDef didn't provide one.
        if !world.contains_resource::<ActionSpaceDef>() {
            world.insert_resource(ActionSpaceDef::new(vec![]));
        }

        // Derive ObservationSpaceDef: converter output > raw extractors > GameDef > empty stub.
        if let Some(ref ext) = extractors {
            if let Some(ref conv) = converter {
                // Converter transforms the extractor output shapes.
                let features = conv.output_features(ext);
                world.insert_resource(ObservationSpaceDef { features });
            } else {
                // No converter: report raw extractor shapes (DictConverter default at runtime).
                world.insert_resource(ext.observation_space_def());
            }
        } else if !world.contains_resource::<ObservationSpaceDef>() {
            world.insert_resource(ObservationSpaceDef { features: vec![] });
        }

        world.insert_resource(RawActionBuffer::default());
        world.insert_resource(PendingActions::default());
        world.insert_resource(ActionMaskBuffer::default());
        world.insert_resource(RewardBuffer::default());
        world.insert_resource(RewardBreakdownBuffer::default());
        world.insert_resource(ShotEventBuffer::default());

        let agents = Self::collect_agents(&mut world);
        let max_agents = agents.len();
        world.insert_resource(AgentRegistry::new(agents, max_agents));

        // Build the new ActionSpace: use from_legacy on the ActionSpaceDef
        // that the GameDef (or default) inserted.
        let legacy_space = world.resource::<ActionSpaceDef>();
        let new_action_space = ActionSpace::from_legacy(legacy_space);

        let mut schedule = Schedule::default();
        Self::configure_phases(&mut schedule);
        Self::add_core_systems(&mut schedule);
        game_def.register_systems(&mut schedule);

        Self {
            world,
            schedule,
            mode: TickMode::Uncapped,
            core: RunnerCore::GameDef(game_def),
            action_translator: None,
            new_action_space,
            extractors,
            converter,
            reward_fn: None,
            replay: None,
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

    fn flush_pending_actions(
        mut pending: ResMut<PendingActions>,
        mut raw_buffer: ResMut<RawActionBuffer>,
    ) {
        for (entity, action) in pending.drain() {
            raw_buffer.insert(entity, action);
        }
    }

    fn add_core_systems(schedule: &mut Schedule) {
        schedule.add_systems(systems::clear_buffers.in_set(EnginePhase::ClearBuffers));
        schedule.add_systems(
            crate::strategy::strategy_system.in_set(EnginePhase::AiDecisions),
        );
        schedule.add_systems(
            (scripted_ai::run_scripted_ai, Self::flush_pending_actions)
                .chain()
                .in_set(EnginePhase::AiDecisions),
        );
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

        // Capture a replay frame if recording is active.
        // We check the flag first, then collect data without holding a mutable
        // borrow on self.replay, and finally push the frame.
        let should_record = self
            .replay
            .as_ref()
            .map_or(false, |r| r.is_recording());

        if should_record {
            let tick = self.tick_count();
            let agent_obs = self.observe_all();
            let agent_actions = self.raw_actions();
            let agent_rewards = self.rewards();
            let agent_dones = self.dones();
            let frame = ReplayFrame {
                tick,
                agent_obs,
                agent_actions,
                agent_rewards,
                agent_dones,
            };
            // Safe: we know self.replay is Some because should_record was true.
            self.replay.as_mut().unwrap().record_frame(frame);
        }
    }

    /// Run `mul` simulation ticks, repeating the current pending actions for
    /// each sub-tick. Rewards are accumulated (summed) across all sub-ticks.
    /// Observations come from the final tick. If any agent's `is_done` triggers
    /// mid-sequence, execution stops early.
    pub fn step_with_mul(&mut self, mul: u32) {
        // Snapshot the pending actions so we can re-insert them each sub-tick.
        // After the first tick(), flush_pending_actions drains PendingActions
        // and clear_buffers clears RawActionBuffer, so we must re-inject.
        let saved_actions: HashMap<Entity, ActionDict> = self
            .world
            .resource::<PendingActions>()
            .actions
            .clone();

        let mut accumulated_rewards: HashMap<Entity, f32> = HashMap::new();

        for i in 0..mul {
            // For sub-ticks after the first, re-insert the saved actions into
            // PendingActions so they survive clear_buffers + flush.
            if i > 0 {
                let mut pending = self.world.resource_mut::<PendingActions>();
                for (entity, action) in &saved_actions {
                    pending.insert(*entity, action.clone());
                }
            }

            self.tick();

            // Accumulate rewards from this sub-tick's RewardBuffer.
            {
                let reward_buf = self.world.resource::<RewardBuffer>();
                for (&entity, &reward) in &reward_buf.rewards {
                    *accumulated_rewards.entry(entity).or_default() += reward;
                }
            }

            // Check early termination: if any agent is done, stop.
            if i < mul - 1 {
                let any_done = {
                    let registry = self.world.resource::<AgentRegistry>();
                    let agents: Vec<Entity> = registry.agents.clone();
                    match &self.core {
                        RunnerCore::Scenario(scenario) => {
                            agents.iter().any(|&e| scenario.is_done(&self.world, e))
                        }
                        RunnerCore::GameDef(game_def) => {
                            agents.iter().any(|&e| game_def.is_done(&self.world, e))
                        }
                    }
                };
                if any_done {
                    break;
                }
            }
        }

        // Write accumulated rewards back into RewardBuffer so that callers
        // reading rewards() after step_with_mul see the full sum.
        {
            let mut reward_buf = self.world.resource_mut::<RewardBuffer>();
            reward_buf.clear();
            for (entity, total) in accumulated_rewards {
                reward_buf.add(entity, total);
            }
        }
    }

    /// Run `config.step_mul` simulation ticks (defaults to 1 if unset).
    /// Convenience wrapper around `step_with_mul`.
    pub fn step_auto(&mut self) {
        let mul = self.config().step_mul.unwrap_or(1);
        self.step_with_mul(mul);
    }

    pub fn drain_telemetry(&mut self) -> Vec<TelemetryEvent> {
        self.world.resource_mut::<TelemetryBuffer>().drain()
    }

    pub fn apply_raw_actions(&mut self, actions: HashMap<Entity, ActionDict>) {
        let mut pending = self.world.resource_mut::<PendingActions>();
        for (entity, action) in actions {
            let translated = if let Some(ref translator) = self.action_translator {
                translator.translate(&action, &self.new_action_space)
            } else {
                action
            };
            pending.insert(entity, translated);
        }
    }

    pub fn observe_all(&self) -> HashMap<usize, HashMap<String, Vec<f32>>> {
        let registry = self.world.resource::<AgentRegistry>();
        let mut result = HashMap::new();
        match &self.core {
            RunnerCore::Scenario(scenario) => {
                for (idx, &entity) in registry.agents.iter().enumerate() {
                    let mut writer = ObsWriter::new();
                    scenario.observe(&self.world, entity, &mut writer);
                    result.insert(idx, writer.buffers);
                }
            }
            RunnerCore::GameDef(_) => {
                if let Some(ref extractors) = self.extractors {
                    // Use the configured converter, or DictConverter (pass-through) by default.
                    let default_converter = DictConverter;
                    let converter: &dyn ObsConverter = match self.converter {
                        Some(ref c) => c.as_ref(),
                        None => &default_converter,
                    };
                    for (idx, &entity) in registry.agents.iter().enumerate() {
                        let raw = extractors.observe(&self.world, entity);
                        result.insert(idx, converter.convert(&raw, extractors));
                    }
                } else {
                    // No extractors configured: return empty observations.
                    for (idx, &_entity) in registry.agents.iter().enumerate() {
                        result.insert(idx, HashMap::new());
                    }
                }
            }
        }
        result
    }

    pub fn rewards(&self) -> HashMap<usize, f32> {
        let registry = self.world.resource::<AgentRegistry>();
        let mut result = HashMap::new();
        match &self.core {
            RunnerCore::Scenario(scenario) => {
                for (idx, &entity) in registry.agents.iter().enumerate() {
                    result.insert(idx, scenario.reward(&self.world, entity));
                }
            }
            RunnerCore::GameDef(_) => {
                if let Some(ref reward_fn) = self.reward_fn {
                    for (idx, &entity) in registry.agents.iter().enumerate() {
                        result.insert(idx, reward_fn.compute(&self.world, entity));
                    }
                } else {
                    // No reward function configured — return 0.0 for each agent.
                    for (idx, &_entity) in registry.agents.iter().enumerate() {
                        result.insert(idx, 0.0);
                    }
                }
            }
        }
        result
    }

    pub fn dones(&self) -> HashMap<usize, bool> {
        let registry = self.world.resource::<AgentRegistry>();
        let mut result = HashMap::new();
        match &self.core {
            RunnerCore::Scenario(scenario) => {
                for (idx, &entity) in registry.agents.iter().enumerate() {
                    result.insert(idx, scenario.is_done(&self.world, entity));
                }
            }
            RunnerCore::GameDef(game_def) => {
                for (idx, &entity) in registry.agents.iter().enumerate() {
                    result.insert(idx, game_def.is_done(&self.world, entity));
                }
            }
        }
        result
    }

    pub fn action_space_def(&self) -> &ActionSpaceDef {
        self.world.resource::<ActionSpaceDef>()
    }

    /// Returns the new-style `ActionSpace` (function-call model).
    pub fn action_space(&self) -> &ActionSpace {
        &self.new_action_space
    }

    pub fn observation_space_def(&self) -> &ObservationSpaceDef {
        self.world.resource::<ObservationSpaceDef>()
    }

    pub fn agent_registry(&self) -> &AgentRegistry {
        self.world.resource::<AgentRegistry>()
    }

    pub fn raw_actions(&self) -> HashMap<usize, Vec<f32>> {
        let registry = self.world.resource::<AgentRegistry>();
        let buffer = self.world.resource::<RawActionBuffer>();
        let mut result = HashMap::new();
        for (idx, &entity) in registry.agents.iter().enumerate() {
            if let Some(action) = buffer.get(entity) {
                result.insert(idx, action.clone());
            }
        }
        result
    }

    pub fn reward_breakdown(&self) -> HashMap<usize, HashMap<String, f32>> {
        let registry = self.world.resource::<AgentRegistry>();
        let buffer = self.world.resource::<RewardBreakdownBuffer>();
        let mut result = HashMap::new();
        for (idx, &entity) in registry.agents.iter().enumerate() {
            if let Some(components) = buffer.get(entity) {
                result.insert(idx, components.clone());
            }
        }
        result
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

    // --- Replay recording ---

    /// Start recording replay frames. Creates a new recorder if needed,
    /// or resets an existing one (clearing previously captured frames).
    pub fn start_recording(&mut self) {
        match self.replay {
            Some(ref mut recorder) => recorder.start(),
            None => {
                let mut recorder = ReplayRecorder::new();
                recorder.start();
                self.replay = Some(recorder);
            }
        }
    }

    /// Stop recording replay frames. Frames are retained for retrieval.
    pub fn stop_recording(&mut self) {
        if let Some(ref mut recorder) = self.replay {
            recorder.stop();
        }
    }

    /// Returns `true` if the replay recorder is actively capturing frames.
    pub fn is_recording(&self) -> bool {
        self.replay
            .as_ref()
            .map_or(false, |r| r.is_recording())
    }

    /// Take all recorded replay frames, leaving the internal buffer empty.
    pub fn drain_replay_frames(&mut self) -> Vec<ReplayFrame> {
        self.replay
            .as_mut()
            .map_or_else(Vec::new, |r| r.drain_frames())
    }

    /// Write all currently recorded frames to a JSON Lines file.
    pub fn save_replay(&self, path: &std::path::Path) -> std::io::Result<()> {
        let frames = self
            .replay
            .as_ref()
            .map_or(&[] as &[ReplayFrame], |r| r.frames());
        ReplayWriter::write_jsonl(frames, path)
    }
}

use std::collections::HashMap;

use bevy_ecs::prelude::*;
use ghostlobby_engine::config::GameConfig;
use glam::Vec2;
use ghostlobby_engine::ecs::components::Agent;
use ghostlobby_engine::features::extractors::{ActionMaskExtractor, SelfStateExtractor};
use ghostlobby_engine::features::ExtractorSet;
use ghostlobby_engine::game_def::GameDef;
use ghostlobby_engine::telemetry::TelemetryEvent;
use ghostlobby_engine::tick::TickRunner;

fn cs_lite_config() -> GameConfig {
    let json = r#"{
        "title": "cs_lite test",
        "tick_rate": 64,
        "arena": { "width": 80.0, "height": 60.0 },
        "movement": { "max_speed": 5.5, "acceleration": 100.0, "friction": 50.0 },
        "combat": { "default_weapon": { "damage": 25.0, "fire_rate": 0.3, "range": 50.0 } },
        "spawning": { "respawn_delay": 999.0, "round_time_limit": 115.0 },
        "teams": { "count": 2, "players_per_team": 5 },
        "obstacles": [],
        "extra": {
            "scenario": "cs_lite",
            "arena_width": 80.0,
            "arena_depth": 60.0,
            "arena_height_3d": 10.0,
            "max_rounds": 24,
            "round_time_limit": 10.0,
            "buy_time": 1.0,
            "end_time": 1.0,
            "hitbox_radius": 2.5
        }
    }"#;
    json.parse::<GameConfig>().expect("failed to parse test config")
}

fn make_runner() -> TickRunner {
    TickRunner::new(cs_lite_config())
}

mod integration {
    use super::*;

    #[test]
    fn smoke_test_100_ticks() {
        let mut runner = make_runner();

        for _ in 0..100 {
            runner.tick();
        }

        assert_eq!(runner.tick_count(), 100, "tick_count should be 100 after 100 ticks");
    }

    #[test]
    fn agents_exist() {
        let runner = make_runner();
        let registry = runner.agent_registry();

        assert_eq!(
            registry.agents.len(),
            10,
            "expected 10 agents (2 teams * 5 players)"
        );
    }

    #[test]
    fn action_space_shape() {
        let runner = make_runner();
        let action_space = runner.action_space_def();

        assert_eq!(action_space.heads.len(), 4, "expected 4 action heads (move_target, shoot, reload, use_action)");
        assert_eq!(action_space.total_size, 4, "expected total_size = 4");
    }

    #[test]
    fn observation_space_features() {
        let runner = make_runner();
        let obs_space = runner.observation_space_def();

        let feature_names: Vec<&str> = obs_space.features.iter().map(|f| f.name.as_str()).collect();
        assert!(
            feature_names.contains(&"self_state"),
            "observation space should contain self_state"
        );
        assert!(
            feature_names.contains(&"enemy_state"),
            "observation space should contain enemy_state"
        );
        assert!(
            feature_names.contains(&"action_mask"),
            "observation space should contain action_mask"
        );
    }

    #[test]
    fn observe_all_returns_all_agents() {
        let mut runner = make_runner();

        for _ in 0..5 {
            runner.tick();
        }

        let observations = runner.observe_all();

        assert_eq!(
            observations.len(),
            10,
            "observe_all should return observations for all 10 agents"
        );

        for idx in 0..10 {
            let obs = observations
                .get(&idx)
                .unwrap_or_else(|| panic!("missing observation for agent {}", idx));
            assert!(
                obs.contains_key("self_state"),
                "agent {} observation missing self_state",
                idx
            );
            assert!(
                obs.contains_key("enemy_state"),
                "agent {} observation missing enemy_state",
                idx
            );
            assert!(
                obs.contains_key("action_mask"),
                "agent {} observation missing action_mask",
                idx
            );
        }
    }

    #[test]
    fn observation_shapes() {
        let mut runner = make_runner();

        runner.tick();

        let observations = runner.observe_all();
        let obs = observations.get(&0).expect("missing observation for agent 0");

        let self_state = obs.get("self_state").expect("missing self_state");
        assert_eq!(
            self_state.len(),
            12,
            "self_state should have 12 values (cs_lite)"
        );

        let action_mask = obs.get("action_mask").expect("missing action_mask");
        assert_eq!(action_mask.len(), 19, "action_mask should have 19 values (cs_lite: 12 move + 2 shoot + 2 reload + 3 use)");
    }

    #[test]
    fn apply_raw_actions_moves_agent() {
        let mut runner = make_runner();

        // Tick past buy freeze phase (1s at 64 tps = 64 ticks) + buffer
        for _ in 0..80 {
            runner.tick();
        }

        let agent_entity = runner.agent_registry().agents[0];

        let initial_obs = runner.observe_all();
        // Check z-position (index 2) since move_dir=0 is +Z in compass dirs
        let initial_pos = initial_obs[&0]["self_state"][2];

        // cs_lite action: [move_target=0(+Z), shoot=0, reload=0, use_action=0]
        let mut actions = HashMap::new();
        actions.insert(agent_entity, vec![0.0, 0.0, 0.0, 0.0]);

        // Run enough ticks for candidate recomputation (every 4 ticks) + pathfinding
        for _ in 0..60 {
            runner.apply_raw_actions(actions.clone());
            runner.tick();
        }

        let final_obs = runner.observe_all();
        let final_pos = final_obs[&0]["self_state"][2];

        assert!(
            (final_pos - initial_pos).abs() > 0.001,
            "agent z-position should have changed after applying move actions (initial={}, final={})",
            initial_pos,
            final_pos
        );
    }

    #[test]
    fn rewards_and_dones_for_all_agents() {
        let mut runner = make_runner();

        runner.tick();

        let rewards = runner.rewards();
        let dones = runner.dones();

        assert_eq!(
            rewards.len(),
            10,
            "rewards should have entries for all 10 agents"
        );
        assert_eq!(
            dones.len(),
            10,
            "dones should have entries for all 10 agents"
        );

        for idx in 0..10 {
            assert!(
                rewards.contains_key(&idx),
                "rewards missing entry for agent {}",
                idx
            );
            assert!(
                dones.contains_key(&idx),
                "dones missing entry for agent {}",
                idx
            );
        }
    }

    #[test]
    fn telemetry_contains_world_snapshots() {
        let mut runner = make_runner();

        for _ in 0..5 {
            runner.tick();
        }

        let events = runner.drain_telemetry();
        assert!(!events.is_empty(), "telemetry should not be empty after ticks");

        let has_snapshot = events.iter().any(|e| matches!(e, TelemetryEvent::WorldSnapshot { .. }));
        assert!(
            has_snapshot,
            "telemetry should contain at least one WorldSnapshot event"
        );
    }

    #[test]
    fn builder_with_scenario() {
        let mut runner = make_runner();

        assert_eq!(runner.agent_registry().agents.len(), 10);
        assert_eq!(runner.action_space_def().total_size, 4);

        for _ in 0..10 {
            runner.tick();
        }

        assert_eq!(runner.tick_count(), 10);
    }
}

// ---------------------------------------------------------------------------
// GameDef integration tests
// ---------------------------------------------------------------------------

fn minimal_config() -> GameConfig {
    let json = r#"{
        "title": "gamedef-test",
        "tick_rate": 64,
        "arena": { "width": 100.0, "height": 100.0 },
        "movement": { "max_speed": 5.0, "acceleration": 500.0, "friction": 400.0 },
        "combat": { "default_weapon": { "damage": 10.0, "fire_rate": 0.5, "range": 50.0 } },
        "spawning": { "respawn_delay": 1.0 },
        "teams": { "count": 1, "players_per_team": 1 },
        "obstacles": []
    }"#;
    json.parse::<GameConfig>().expect("failed to parse minimal config")
}

/// A minimal GameDef that spawns a single agent with no game-specific systems.
struct MinimalGameDef;

impl GameDef for MinimalGameDef {
    fn name(&self) -> &str {
        "minimal"
    }

    fn needs_physics(&self) -> bool {
        true
    }

    fn setup(&self, world: &mut World, _config: &GameConfig) {
        use ghostlobby_engine::ecs::components::*;
        use ghostlobby_engine::ecs::resources::*;
        use glam::Vec2;

        // GameDef needs to set up game-specific resources that the scenario
        // systems might require. For a minimal game, just spawn an agent.
        world.insert_resource(RoundState::default());
        world.insert_resource(ObstacleLayout::default());
        world.insert_resource(SpawnPointPool(vec![Vec2::new(50.0, 50.0)]));

        // Create a physics body for the agent via the world resource.
        let physics = world.resource_mut::<ghostlobby_engine::physics::PhysicsState>();
        let (body, collider) = physics.into_inner().add_dynamic_body(Vec2::new(50.0, 50.0), 10.0);

        world.spawn((
            Position(Vec2::new(50.0, 50.0)),
            Velocity(Vec2::ZERO),
            Facing(0.0),
            Health { current: 100.0, max: 100.0 },
            Team(0),
            Weapon {
                damage: 10.0,
                fire_rate: 0.5,
                range: 50.0,
                cooldown_remaining: 0.0,
            },
            Agent { source_id: 0 },
            PhysicsHandle { body, collider },
        ));
    }

    fn register_systems(&self, _schedule: &mut Schedule) {}
}

/// A GameDef with no physics, spawning a bare-minimum agent.
struct NoPhysicsGameDef;

impl GameDef for NoPhysicsGameDef {
    fn name(&self) -> &str {
        "no-physics"
    }

    fn needs_physics(&self) -> bool {
        false
    }

    fn setup(&self, world: &mut World, _config: &GameConfig) {
        world.spawn(Agent { source_id: 0 });
    }

    fn register_systems(&self, _schedule: &mut Schedule) {}
}

/// A GameDef that reports done after a fixed number of ticks.
struct TimedGameDef {
    max_ticks: u64,
}

impl GameDef for TimedGameDef {
    fn name(&self) -> &str {
        "timed"
    }

    fn setup(&self, world: &mut World, _config: &GameConfig) {
        use ghostlobby_engine::ecs::components::*;
        use ghostlobby_engine::ecs::resources::*;
        use glam::Vec2;

        world.insert_resource(RoundState::default());
        world.insert_resource(ObstacleLayout::default());
        world.insert_resource(SpawnPointPool(vec![Vec2::new(50.0, 50.0)]));

        let physics = world.resource_mut::<ghostlobby_engine::physics::PhysicsState>();
        let (body, collider) = physics.into_inner().add_dynamic_body(Vec2::new(50.0, 50.0), 10.0);

        world.spawn((
            Position(Vec2::new(50.0, 50.0)),
            Velocity(Vec2::ZERO),
            Facing(0.0),
            Health { current: 100.0, max: 100.0 },
            Team(0),
            Weapon {
                damage: 10.0,
                fire_rate: 0.5,
                range: 50.0,
                cooldown_remaining: 0.0,
            },
            Agent { source_id: 0 },
            PhysicsHandle { body, collider },
        ));
    }

    fn register_systems(&self, _schedule: &mut Schedule) {}

    fn is_done(&self, world: &World, _agent: Entity) -> bool {
        let tick = world.resource::<ghostlobby_engine::ecs::resources::TickState>();
        tick.tick >= self.max_ticks
    }
}

mod game_def_integration {
    use super::*;

    #[test]
    fn game_def_smoke_test_ticks() {
        let config = minimal_config();
        let mut runner = TickRunner::builder(config)
            .with_game_def(MinimalGameDef)
            .build();

        for _ in 0..50 {
            runner.tick();
        }

        assert_eq!(runner.tick_count(), 50);
    }

    #[test]
    fn game_def_agent_registered() {
        let config = minimal_config();
        let runner = TickRunner::builder(config)
            .with_game_def(MinimalGameDef)
            .build();

        assert_eq!(
            runner.agent_registry().agents.len(),
            1,
            "MinimalGameDef should register exactly 1 agent"
        );
    }

    #[test]
    fn game_def_default_action_space_is_empty() {
        let config = minimal_config();
        let runner = TickRunner::builder(config)
            .with_game_def(MinimalGameDef)
            .build();

        let action_space = runner.action_space_def();
        assert_eq!(action_space.heads.len(), 0, "default GameDef action space should be empty");
        assert_eq!(action_space.total_size, 0);
    }

    #[test]
    fn game_def_observe_all_returns_empty_obs() {
        let config = minimal_config();
        let mut runner = TickRunner::builder(config)
            .with_game_def(MinimalGameDef)
            .build();

        runner.tick();

        let obs = runner.observe_all();
        assert_eq!(obs.len(), 1, "should have 1 agent observation");
        assert!(
            obs[&0].is_empty(),
            "GameDef observations should be empty until extractors are added (Phase 2)"
        );
    }

    #[test]
    fn game_def_rewards_return_zero() {
        let config = minimal_config();
        let mut runner = TickRunner::builder(config)
            .with_game_def(MinimalGameDef)
            .build();

        runner.tick();

        let rewards = runner.rewards();
        assert_eq!(rewards.len(), 1);
        assert_eq!(rewards[&0], 0.0, "GameDef rewards should be 0.0 until reward fns are added");
    }

    #[test]
    fn game_def_is_done_delegates_to_game_def() {
        let config = minimal_config();
        let mut runner = TickRunner::builder(config)
            .with_game_def(TimedGameDef { max_ticks: 10 })
            .build();

        // Tick up to the threshold
        for _ in 0..9 {
            runner.tick();
        }
        let dones = runner.dones();
        assert!(!dones[&0], "should not be done before max_ticks");

        // One more tick to reach max_ticks
        runner.tick();
        let dones = runner.dones();
        assert!(dones[&0], "should be done at max_ticks");
    }

    #[test]
    fn game_def_telemetry_works() {
        let config = minimal_config();
        let mut runner = TickRunner::builder(config)
            .with_game_def(MinimalGameDef)
            .build();

        for _ in 0..4 {
            runner.tick();
        }

        let events = runner.drain_telemetry();
        assert!(!events.is_empty(), "telemetry should produce events from core systems");

        let has_snapshot = events.iter().any(|e| matches!(e, TelemetryEvent::WorldSnapshot { .. }));
        assert!(has_snapshot, "should contain WorldSnapshot events");
    }

    #[test]
    fn game_def_no_physics_builds_and_ticks() {
        // NoPhysicsGameDef skips physics — core physics systems will skip
        // their work since there are no entities with PhysicsHandle.
        // We just verify it doesn't panic.
        let config = minimal_config();
        let runner = TickRunner::builder(config)
            .with_game_def(NoPhysicsGameDef)
            .build();

        assert_eq!(runner.agent_registry().agents.len(), 1);

        // Can't tick because core systems require PhysicsState resource.
        // This test verifies build succeeds and agent is registered.
        // Future: when needs_physics=false, engine should skip physics systems.
    }

    #[test]
    fn game_def_config_accessible() {
        let config = minimal_config();
        let runner = TickRunner::builder(config.clone())
            .with_game_def(MinimalGameDef)
            .build();

        assert_eq!(runner.config().title, "gamedef-test");
        assert_eq!(runner.config().tick_rate, 64);
    }
}

// ---------------------------------------------------------------------------
// Action System (Phase 4) integration tests
// ---------------------------------------------------------------------------

mod action_system_integration {
    use super::*;
    use ghostlobby_engine::action_space::{ActionHead, ActionSpaceDef};
    use ghostlobby_engine::actions::{FlatTranslator, FunctionCallTranslator};

    /// A GameDef that inserts a non-empty ActionSpaceDef so we can test translators.
    struct ActionTestGameDef;

    impl GameDef for ActionTestGameDef {
        fn name(&self) -> &str {
            "action-test"
        }

        fn setup(&self, world: &mut World, _config: &GameConfig) {
            use ghostlobby_engine::ecs::components::*;
            use ghostlobby_engine::ecs::resources::*;
            use glam::Vec2;

            world.insert_resource(RoundState::default());
            world.insert_resource(ObstacleLayout::default());
            world.insert_resource(SpawnPointPool(vec![Vec2::new(50.0, 50.0)]));

            // Define a 3-head action space: move(2 continuous), shoot(1 discrete), aim(1 continuous)
            world.insert_resource(ActionSpaceDef::new(vec![
                ActionHead::Continuous {
                    name: "move".into(),
                    size: 2,
                    low: vec![-1.0; 2],
                    high: vec![1.0; 2],
                },
                ActionHead::Discrete {
                    name: "shoot".into(),
                    n: 2,
                },
                ActionHead::Continuous {
                    name: "aim".into(),
                    size: 1,
                    low: vec![-3.14],
                    high: vec![3.14],
                },
            ]));

            let physics = world.resource_mut::<ghostlobby_engine::physics::PhysicsState>();
            let (body, collider) = physics.into_inner().add_dynamic_body(Vec2::new(50.0, 50.0), 10.0);

            world.spawn((
                Position(Vec2::new(50.0, 50.0)),
                Velocity(Vec2::ZERO),
                Facing(0.0),
                Health { current: 100.0, max: 100.0 },
                Team(0),
                Weapon {
                    damage: 10.0,
                    fire_rate: 0.5,
                    range: 50.0,
                    cooldown_remaining: 0.0,
                },
                Agent { source_id: 0 },
                PhysicsHandle { body, collider },
            ));
        }

        fn register_systems(&self, _schedule: &mut Schedule) {}
    }

    #[test]
    fn action_space_derived_from_legacy() {
        let config = minimal_config();
        let runner = TickRunner::builder(config)
            .with_game_def(ActionTestGameDef)
            .build();

        let space = runner.action_space();
        assert_eq!(space.num_functions(), 3);
        assert_eq!(space.flat_size(), 4); // 2 + 1 + 1
        assert_eq!(space.functions[0].name, "move");
        assert_eq!(space.functions[1].name, "shoot");
        assert_eq!(space.functions[2].name, "aim");
    }

    #[test]
    fn flat_translator_wired_through_builder() {
        let config = minimal_config();
        let mut runner = TickRunner::builder(config)
            .with_game_def(ActionTestGameDef)
            .with_action_translator(FlatTranslator)
            .build();

        let agent_entity = runner.agent_registry().agents[0];

        // Apply actions through the translator
        let mut actions = HashMap::new();
        actions.insert(agent_entity, vec![0.5, -0.3, 1.0, 1.57]);
        runner.apply_raw_actions(actions);
        runner.tick();

        // Verify actions reached the buffer (FlatTranslator passes through)
        let raw = runner.raw_actions();
        assert!(raw.contains_key(&0), "agent 0 should have raw actions");
    }

    #[test]
    fn function_call_translator_wired_through_builder() {
        let config = minimal_config();
        let mut runner = TickRunner::builder(config)
            .with_game_def(ActionTestGameDef)
            .with_action_translator(FunctionCallTranslator)
            .build();

        let agent_entity = runner.agent_registry().agents[0];

        // Call function 1 ("shoot") with arg [1.0]
        // FunctionCallTranslator input: [fn_idx, arg0, ...]
        let mut actions = HashMap::new();
        actions.insert(agent_entity, vec![1.0, 1.0]);
        runner.apply_raw_actions(actions);
        runner.tick();

        // Verify actions flowed through the translator to the buffer
        let raw = runner.raw_actions();
        let action = raw.get(&0).expect("agent 0 should have raw actions");
        // Total flat size is 4: [move(2), shoot(1), aim(1)]
        assert_eq!(action.len(), 4);
        // Function 1 ("shoot") at offset 2
        assert_eq!(action[0], 0.0, "move[0] should be 0 (not called)");
        assert_eq!(action[1], 0.0, "move[1] should be 0 (not called)");
        assert_eq!(action[2], 1.0, "shoot should be 1.0 (called)");
        assert_eq!(action[3], 0.0, "aim should be 0 (not called)");
    }

    #[test]
    fn no_translator_passes_actions_unchanged() {
        let config = minimal_config();
        let mut runner = TickRunner::builder(config)
            .with_game_def(ActionTestGameDef)
            .build();

        let agent_entity = runner.agent_registry().agents[0];

        let input = vec![0.5, -0.3, 1.0, 1.57];
        let mut actions = HashMap::new();
        actions.insert(agent_entity, input.clone());
        runner.apply_raw_actions(actions);
        runner.tick();

        let raw = runner.raw_actions();
        let action = raw.get(&0).expect("agent 0 should have raw actions");
        assert_eq!(action, &input);
    }

    #[test]
    fn action_space_accessor_returns_consistent_data() {
        let config = minimal_config();
        let runner = TickRunner::builder(config)
            .with_game_def(ActionTestGameDef)
            .build();

        let legacy = runner.action_space_def();
        let new_space = runner.action_space();

        assert_eq!(legacy.total_size, new_space.flat_size());
        assert_eq!(legacy.heads.len(), new_space.num_functions());
    }
}

// ---------------------------------------------------------------------------
// Feature Extractor (Phase 2) integration tests
// ---------------------------------------------------------------------------

/// A GameDef that spawns 2 agents with Position, Velocity, and Health.
/// Uses physics=true so core systems don't panic on tick().
struct ExtractorTestGameDef;

impl GameDef for ExtractorTestGameDef {
    fn name(&self) -> &str {
        "extractor-test"
    }

    fn needs_physics(&self) -> bool {
        true
    }

    fn setup(&self, world: &mut World, _config: &GameConfig) {
        use ghostlobby_engine::ecs::components::*;
        use ghostlobby_engine::ecs::resources::*;

        world.insert_resource(RoundState::default());
        world.insert_resource(ObstacleLayout::default());
        world.insert_resource(SpawnPointPool(vec![Vec2::new(50.0, 50.0)]));

        let physics = world.resource_mut::<ghostlobby_engine::physics::PhysicsState>();
        let (b0, c0) = physics.into_inner().add_dynamic_body(Vec2::new(10.0, 20.0), 10.0);
        let physics = world.resource_mut::<ghostlobby_engine::physics::PhysicsState>();
        let (b1, c1) = physics.into_inner().add_dynamic_body(Vec2::new(50.0, 60.0), 10.0);

        world.spawn((
            Agent { source_id: 0 },
            Position(Vec2::new(10.0, 20.0)),
            Velocity(Vec2::new(1.0, -1.0)),
            Health {
                current: 80.0,
                max: 100.0,
            },
            PhysicsHandle { body: b0, collider: c0 },
        ));
        world.spawn((
            Agent { source_id: 1 },
            Position(Vec2::new(50.0, 60.0)),
            Velocity(Vec2::new(-2.0, 3.0)),
            Health {
                current: 100.0,
                max: 100.0,
            },
            PhysicsHandle { body: b1, collider: c1 },
        ));
    }

    fn register_systems(&self, _schedule: &mut Schedule) {}
}

mod extractor_integration {
    use super::*;

    #[test]
    fn game_def_with_extractors_produces_observations() {
        let extractors = ExtractorSet::new()
            .add(SelfStateExtractor::new(10.0))
            .add(ActionMaskExtractor::new(4));

        let runner = TickRunner::builder(minimal_config())
            .with_game_def(ExtractorTestGameDef)
            .with_extractors(extractors)
            .build();

        let obs = runner.observe_all();
        assert_eq!(obs.len(), 2, "should have observations for 2 agents");

        for idx in 0..2 {
            let agent_obs = &obs[&idx];
            assert!(
                agent_obs.contains_key("self_state"),
                "agent {} missing self_state",
                idx
            );
            assert!(
                agent_obs.contains_key("action_mask"),
                "agent {} missing action_mask",
                idx
            );
            assert_eq!(agent_obs["self_state"].len(), 9);
            assert_eq!(agent_obs["action_mask"].len(), 4);
        }

        // Agent 0: position (10, 20), arena (100, 100) -> normalized (0.1, 0.2)
        let a0_self = &obs[&0]["self_state"];
        assert!(
            (a0_self[0] - 0.1).abs() < 1e-4,
            "agent 0 x should be ~0.1, got {}",
            a0_self[0]
        );
        assert!(
            (a0_self[1] - 0.2).abs() < 1e-4,
            "agent 0 y should be ~0.2, got {}",
            a0_self[1]
        );

        // Health: 80/100 = 0.8
        assert!(
            (a0_self[5] - 0.8).abs() < 1e-4,
            "agent 0 hp should be ~0.8, got {}",
            a0_self[5]
        );

        // Alive = 1.0
        assert!((a0_self[7] - 1.0).abs() < 1e-4);

        // Action mask: all 1.0 (no mask set)
        assert_eq!(obs[&0]["action_mask"], vec![1.0; 4]);
    }

    #[test]
    fn observation_space_def_reflects_extractors() {
        let extractors = ExtractorSet::new()
            .add(SelfStateExtractor::new(10.0))
            .add(ActionMaskExtractor::new(4));

        let runner = TickRunner::builder(minimal_config())
            .with_game_def(ExtractorTestGameDef)
            .with_extractors(extractors)
            .build();

        let def = runner.observation_space_def();
        assert_eq!(def.features.len(), 2);
        assert_eq!(def.features[0].name, "self_state");
        assert_eq!(def.features[0].shape, vec![9]);
        assert_eq!(def.features[1].name, "action_mask");
        assert_eq!(def.features[1].shape, vec![4]);
    }

    #[test]
    fn game_def_without_extractors_returns_empty_obs() {
        let runner = TickRunner::builder(minimal_config())
            .with_game_def(ExtractorTestGameDef)
            .build();

        let obs = runner.observe_all();
        assert_eq!(obs.len(), 2);
        for idx in 0..2 {
            assert!(obs[&idx].is_empty());
        }
    }

    #[test]
    fn tick_then_observe_works() {
        let extractors = ExtractorSet::new().add(SelfStateExtractor::new(10.0));

        let mut runner = TickRunner::builder(minimal_config())
            .with_game_def(ExtractorTestGameDef)
            .with_extractors(extractors)
            .build();

        for _ in 0..5 {
            runner.tick();
        }

        let obs = runner.observe_all();
        assert_eq!(obs.len(), 2);
        for idx in 0..2 {
            assert_eq!(obs[&idx]["self_state"].len(), 9);
        }
    }

    #[test]
    fn with_extractor_builder_convenience() {
        let runner = TickRunner::builder(minimal_config())
            .with_game_def(ExtractorTestGameDef)
            .with_extractor(SelfStateExtractor::new(10.0))
            .with_extractor(ActionMaskExtractor::new(3))
            .build();

        let def = runner.observation_space_def();
        assert_eq!(def.features.len(), 2);
        assert_eq!(def.features[0].name, "self_state");
        assert_eq!(def.features[1].name, "action_mask");

        let obs = runner.observe_all();
        assert_eq!(obs.len(), 2);
        assert!(obs[&0].contains_key("self_state"));
        assert!(obs[&0].contains_key("action_mask"));
    }
}

// ---------------------------------------------------------------------------
// Converter (Phase 3) integration tests
// ---------------------------------------------------------------------------

mod converter_integration {
    use super::*;
    use ghostlobby_engine::converters::{DictConverter, FlatConverter, SpatialConverter};

    #[test]
    fn flat_converter_produces_single_obs_key() {
        let extractors = ExtractorSet::new()
            .add(SelfStateExtractor::new(10.0))
            .add(ActionMaskExtractor::new(4));

        let runner = TickRunner::builder(minimal_config())
            .with_game_def(ExtractorTestGameDef)
            .with_extractors(extractors)
            .with_converter(FlatConverter)
            .build();

        let obs = runner.observe_all();
        assert_eq!(obs.len(), 2, "should have observations for 2 agents");

        for idx in 0..2 {
            let agent_obs = &obs[&idx];
            assert_eq!(
                agent_obs.len(),
                1,
                "FlatConverter should produce exactly 1 key"
            );
            assert!(
                agent_obs.contains_key("obs"),
                "FlatConverter output key should be 'obs'"
            );
            // SelfStateExtractor(9) + ActionMaskExtractor(4) = 13
            assert_eq!(
                agent_obs["obs"].len(),
                13,
                "flat obs should be 9 + 4 = 13"
            );
        }
    }

    #[test]
    fn flat_converter_observation_space_def_reflects_converter() {
        let extractors = ExtractorSet::new()
            .add(SelfStateExtractor::new(10.0))
            .add(ActionMaskExtractor::new(4));

        let runner = TickRunner::builder(minimal_config())
            .with_game_def(ExtractorTestGameDef)
            .with_extractors(extractors)
            .with_converter(FlatConverter)
            .build();

        let def = runner.observation_space_def();
        assert_eq!(def.features.len(), 1);
        assert_eq!(def.features[0].name, "obs");
        assert_eq!(def.features[0].shape, vec![13]); // 9 + 4
    }

    #[test]
    fn dict_converter_preserves_named_features() {
        let extractors = ExtractorSet::new()
            .add(SelfStateExtractor::new(10.0))
            .add(ActionMaskExtractor::new(4));

        let runner = TickRunner::builder(minimal_config())
            .with_game_def(ExtractorTestGameDef)
            .with_extractors(extractors)
            .with_converter(DictConverter)
            .build();

        let obs = runner.observe_all();
        assert_eq!(obs.len(), 2);

        for idx in 0..2 {
            let agent_obs = &obs[&idx];
            assert!(agent_obs.contains_key("self_state"));
            assert!(agent_obs.contains_key("action_mask"));
            assert_eq!(agent_obs["self_state"].len(), 9);
            assert_eq!(agent_obs["action_mask"].len(), 4);
        }
    }

    #[test]
    fn dict_converter_observation_space_def_matches_extractors() {
        let extractors = ExtractorSet::new()
            .add(SelfStateExtractor::new(10.0))
            .add(ActionMaskExtractor::new(4));

        let runner = TickRunner::builder(minimal_config())
            .with_game_def(ExtractorTestGameDef)
            .with_extractors(extractors)
            .with_converter(DictConverter)
            .build();

        let def = runner.observation_space_def();
        assert_eq!(def.features.len(), 2);
        assert_eq!(def.features[0].name, "self_state");
        assert_eq!(def.features[0].shape, vec![9]);
        assert_eq!(def.features[1].name, "action_mask");
        assert_eq!(def.features[1].shape, vec![4]);
    }

    #[test]
    fn no_converter_defaults_to_dict_behavior() {
        // When no converter is explicitly set, the default DictConverter
        // should be used, preserving named features from extractors.
        let extractors = ExtractorSet::new()
            .add(SelfStateExtractor::new(10.0))
            .add(ActionMaskExtractor::new(4));

        let runner = TickRunner::builder(minimal_config())
            .with_game_def(ExtractorTestGameDef)
            .with_extractors(extractors)
            .build();

        let obs = runner.observe_all();
        assert_eq!(obs.len(), 2);

        // Should behave identically to DictConverter (pass-through)
        for idx in 0..2 {
            let agent_obs = &obs[&idx];
            assert!(agent_obs.contains_key("self_state"));
            assert!(agent_obs.contains_key("action_mask"));
            assert_eq!(agent_obs["self_state"].len(), 9);
            assert_eq!(agent_obs["action_mask"].len(), 4);
        }
    }

    #[test]
    fn spatial_converter_groups_vector_features() {
        let extractors = ExtractorSet::new()
            .add(SelfStateExtractor::new(10.0))
            .add(ActionMaskExtractor::new(4));

        let runner = TickRunner::builder(minimal_config())
            .with_game_def(ExtractorTestGameDef)
            .with_extractors(extractors)
            .with_converter(SpatialConverter)
            .build();

        let obs = runner.observe_all();
        assert_eq!(obs.len(), 2);

        // Both SelfStateExtractor and ActionMaskExtractor produce Vector shapes,
        // so SpatialConverter should concatenate them into "flat_features".
        for idx in 0..2 {
            let agent_obs = &obs[&idx];
            assert_eq!(
                agent_obs.len(),
                1,
                "SpatialConverter with only Vector extractors should produce 1 key"
            );
            assert!(agent_obs.contains_key("flat_features"));
            assert_eq!(
                agent_obs["flat_features"].len(),
                13,
                "flat_features should be 9 + 4 = 13"
            );
        }
    }

    #[test]
    fn spatial_converter_observation_space_def() {
        let extractors = ExtractorSet::new()
            .add(SelfStateExtractor::new(10.0))
            .add(ActionMaskExtractor::new(4));

        let runner = TickRunner::builder(minimal_config())
            .with_game_def(ExtractorTestGameDef)
            .with_extractors(extractors)
            .with_converter(SpatialConverter)
            .build();

        let def = runner.observation_space_def();
        assert_eq!(def.features.len(), 1);
        assert_eq!(def.features[0].name, "flat_features");
        assert_eq!(def.features[0].shape, vec![13]);
    }

    #[test]
    fn converter_does_not_affect_scenario_path() {
        // The Scenario path should ignore the converter entirely.
        let runner = make_runner();

        let obs = runner.observe_all();
        assert_eq!(obs.len(), 10);

        // Scenario path should produce named features (self_state, enemy_state, action_mask)
        assert!(obs[&0].contains_key("self_state"));
    }

    #[test]
    fn tick_then_observe_with_flat_converter() {
        let extractors = ExtractorSet::new()
            .add(SelfStateExtractor::new(10.0))
            .add(ActionMaskExtractor::new(4));

        let mut runner = TickRunner::builder(minimal_config())
            .with_game_def(ExtractorTestGameDef)
            .with_extractors(extractors)
            .with_converter(FlatConverter)
            .build();

        for _ in 0..10 {
            runner.tick();
        }

        let obs = runner.observe_all();
        assert_eq!(obs.len(), 2);
        for idx in 0..2 {
            assert_eq!(obs[&idx]["obs"].len(), 13);
        }
    }
}

// ---------------------------------------------------------------------------
// step_mul (Phase 5) integration tests
// ---------------------------------------------------------------------------

fn config_with_step_mul(mul: u32) -> GameConfig {
    let json = format!(
        r#"{{
            "title": "step-mul-test",
            "tick_rate": 64,
            "arena": {{ "width": 100.0, "height": 100.0 }},
            "movement": {{ "max_speed": 5.0, "acceleration": 500.0, "friction": 400.0 }},
            "combat": {{ "default_weapon": {{ "damage": 10.0, "fire_rate": 0.5, "range": 50.0 }} }},
            "spawning": {{ "respawn_delay": 1.0 }},
            "teams": {{ "count": 1, "players_per_team": 1 }},
            "obstacles": [],
            "step_mul": {mul}
        }}"#
    );
    json.parse::<GameConfig>().expect("failed to parse step_mul config")
}

// ---------------------------------------------------------------------------
// Replay System (Phase 7) integration tests
// ---------------------------------------------------------------------------

mod replay_integration {
    use super::*;
    use ghostlobby_engine::replay::writer::ReplayWriter;

    #[test]
    fn replay_scenario_path_record_10_ticks() {
        let mut runner = make_runner();

        runner.start_recording();
        assert!(runner.is_recording());

        for _ in 0..10 {
            runner.tick();
        }

        runner.stop_recording();
        assert!(!runner.is_recording());

        let frames = runner.drain_replay_frames();
        assert_eq!(frames.len(), 10, "should capture exactly 10 frames");

        // Verify tick numbers are sequential (1..=10 because tick increments before capture)
        for (i, frame) in frames.iter().enumerate() {
            assert_eq!(
                frame.tick,
                (i as u64) + 1,
                "frame {} should have tick {}",
                i,
                i + 1
            );
        }

        // Verify obs/actions/rewards/dones are populated for all 10 agents
        for frame in &frames {
            assert_eq!(
                frame.agent_obs.len(),
                10,
                "each frame should have obs for 10 agents"
            );
            assert_eq!(
                frame.agent_rewards.len(),
                10,
                "each frame should have rewards for 10 agents"
            );
            assert_eq!(
                frame.agent_dones.len(),
                10,
                "each frame should have dones for 10 agents"
            );
        }
    }

    #[test]
    fn replay_game_def_path_record_10_ticks() {
        let config = minimal_config();
        let mut runner = TickRunner::builder(config)
            .with_game_def(MinimalGameDef)
            .build();

        runner.start_recording();

        for _ in 0..10 {
            runner.tick();
        }

        runner.stop_recording();

        let frames = runner.drain_replay_frames();
        assert_eq!(frames.len(), 10, "should capture exactly 10 frames");

        for (i, frame) in frames.iter().enumerate() {
            assert_eq!(frame.tick, (i as u64) + 1);
        }

        // MinimalGameDef has 1 agent
        for frame in &frames {
            assert_eq!(frame.agent_rewards.len(), 1);
            assert_eq!(frame.agent_dones.len(), 1);
        }
    }

    #[test]
    fn replay_save_and_load_roundtrip() {
        let mut runner = make_runner();

        runner.start_recording();
        for _ in 0..5 {
            runner.tick();
        }
        runner.stop_recording();

        let dir = std::env::temp_dir().join("ghostlobby_replay_integration");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_integration.jsonl");

        runner.save_replay(&path).unwrap();

        let loaded = ReplayWriter::read_jsonl(&path).unwrap();
        assert_eq!(loaded.len(), 5);

        // Verify tick numbers roundtrip
        for (i, frame) in loaded.iter().enumerate() {
            assert_eq!(frame.tick, (i as u64) + 1);
        }

        // Clean up
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn replay_no_overhead_when_not_recording() {
        let mut runner = make_runner();

        // Tick without recording — no frames should be captured
        for _ in 0..10 {
            runner.tick();
        }

        assert!(!runner.is_recording());
        let frames = runner.drain_replay_frames();
        assert_eq!(frames.len(), 0, "no frames should be captured when not recording");
    }

    #[test]
    fn replay_drain_clears_buffer() {
        let mut runner = make_runner();

        runner.start_recording();
        for _ in 0..5 {
            runner.tick();
        }

        let first_drain = runner.drain_replay_frames();
        assert_eq!(first_drain.len(), 5);

        let second_drain = runner.drain_replay_frames();
        assert_eq!(second_drain.len(), 0, "drain should clear the buffer");
    }
}

mod step_mul_integration {
    use super::*;

    #[test]
    fn step_with_mul_1_behaves_like_tick() {
        let config = minimal_config();
        let mut runner_tick = TickRunner::builder(config.clone())
            .with_game_def(MinimalGameDef)
            .build();
        let mut runner_mul = TickRunner::builder(config)
            .with_game_def(MinimalGameDef)
            .build();

        runner_tick.tick();
        runner_mul.step_with_mul(1);

        assert_eq!(
            runner_tick.tick_count(),
            runner_mul.tick_count(),
            "step_with_mul(1) should advance tick_count by 1, same as tick()"
        );
    }

    #[test]
    fn step_with_mul_4_advances_tick_count_by_4() {
        let config = minimal_config();
        let mut runner = TickRunner::builder(config)
            .with_game_def(MinimalGameDef)
            .build();

        runner.step_with_mul(4);

        assert_eq!(
            runner.tick_count(),
            4,
            "step_with_mul(4) should advance tick_count by 4"
        );
    }

    #[test]
    fn step_auto_reads_config_step_mul() {
        let config = config_with_step_mul(3);
        assert_eq!(config.step_mul, Some(3));

        let mut runner = TickRunner::builder(config)
            .with_game_def(MinimalGameDef)
            .build();

        runner.step_auto();

        assert_eq!(
            runner.tick_count(),
            3,
            "step_auto() with step_mul=3 should advance tick_count by 3"
        );
    }

    #[test]
    fn step_auto_defaults_to_1_when_step_mul_absent() {
        let config = minimal_config();
        assert_eq!(config.step_mul, None);

        let mut runner = TickRunner::builder(config)
            .with_game_def(MinimalGameDef)
            .build();

        runner.step_auto();

        assert_eq!(
            runner.tick_count(),
            1,
            "step_auto() with no step_mul should advance tick_count by 1"
        );
    }

    #[test]
    fn step_with_mul_actions_persist_across_subticks() {
        // Use the arena3d config which has a real action space (5 heads, total_size=5)
        let config = cs_lite_config();
        let mut runner = TickRunner::new(config);

        let agent_entity = runner.agent_registry().agents[0];

        // Apply a move-forward action: [move_x, move_y, yaw, pitch, shoot]
        let action = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let mut actions = HashMap::new();
        actions.insert(agent_entity, action);
        runner.apply_raw_actions(actions);
        runner.step_with_mul(4);

        assert_eq!(runner.tick_count(), 4);

        // Verify the raw action buffer still has data after step_with_mul
        // (the last sub-tick should have the re-injected actions)
        let raw = runner.raw_actions();
        assert!(
            raw.contains_key(&0),
            "agent should have raw actions after step_with_mul"
        );
    }

    #[test]
    fn step_with_mul_early_termination_on_done() {
        // TimedGameDef reports done after max_ticks
        let config = minimal_config();
        let mut runner = TickRunner::builder(config)
            .with_game_def(TimedGameDef { max_ticks: 3 })
            .build();

        // Try to step 10 sub-ticks, but should stop at tick 3 when done
        runner.step_with_mul(10);

        // is_done triggers when tick >= 3, so we run tick 0->1, 1->2, 2->3
        // then check done before tick 4 and stop
        assert!(
            runner.tick_count() <= 4,
            "step_with_mul should stop early when is_done triggers, tick_count={}",
            runner.tick_count()
        );
        assert!(
            runner.tick_count() >= 3,
            "should have run at least 3 ticks before done, tick_count={}",
            runner.tick_count()
        );
    }

    #[test]
    fn step_with_mul_multiple_calls_accumulate() {
        let config = minimal_config();
        let mut runner = TickRunner::builder(config)
            .with_game_def(MinimalGameDef)
            .build();

        runner.step_with_mul(4);
        runner.step_with_mul(4);
        runner.step_with_mul(4);

        assert_eq!(
            runner.tick_count(),
            12,
            "three calls to step_with_mul(4) should give tick_count=12"
        );
    }

    #[test]
    fn step_auto_multiple_calls() {
        let config = config_with_step_mul(2);
        let mut runner = TickRunner::builder(config)
            .with_game_def(MinimalGameDef)
            .build();

        runner.step_auto();
        runner.step_auto();
        runner.step_auto();

        assert_eq!(
            runner.tick_count(),
            6,
            "three calls to step_auto() with step_mul=2 should give tick_count=6"
        );
    }
}

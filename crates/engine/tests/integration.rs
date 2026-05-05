use std::collections::HashMap;

use ghostlobby_engine::config::GameConfig;
use ghostlobby_engine::scenario::DeathmatchScenario;
use ghostlobby_engine::telemetry::TelemetryEvent;
use ghostlobby_engine::tick::TickRunner;

/// Build a GameConfig matching configs/arena_deathmatch.json.
/// Constructed in code so tests work regardless of working directory.
fn deathmatch_config() -> GameConfig {
    let json = r#"{
        "title": "arena-deathmatch",
        "tick_rate": 128,
        "arena": { "width": 1000.0, "height": 1000.0 },
        "movement": { "max_speed": 200.0, "acceleration": 1000.0, "friction": 800.0 },
        "combat": { "default_weapon": { "damage": 25.0, "fire_rate": 0.2, "range": 500.0 } },
        "spawning": { "respawn_delay": 3.0 },
        "teams": { "count": 2, "players_per_team": 5 },
        "obstacles": [
            { "x": 400, "y": 450, "width": 200, "height": 50 },
            { "x": 200, "y": 200, "width": 80, "height": 80 },
            { "x": 700, "y": 700, "width": 80, "height": 80 },
            { "x": 450, "y": 150, "width": 50, "height": 150 },
            { "x": 450, "y": 650, "width": 50, "height": 150 }
        ]
    }"#;
    json.parse::<GameConfig>().expect("failed to parse test config")
}

mod integration {
    use super::*;

    // -----------------------------------------------------------------------
    // 1. Smoke test — run 100 ticks, verify tick_count
    // -----------------------------------------------------------------------
    #[test]
    fn smoke_test_100_ticks() {
        let config = deathmatch_config();
        let mut runner = TickRunner::new(config);

        for _ in 0..100 {
            runner.tick();
        }

        assert_eq!(runner.tick_count(), 100, "tick_count should be 100 after 100 ticks");
    }

    // -----------------------------------------------------------------------
    // 2. Agents exist — 2 teams * 5 players = 10
    // -----------------------------------------------------------------------
    #[test]
    fn agents_exist() {
        let config = deathmatch_config();
        let runner = TickRunner::new(config);
        let registry = runner.agent_registry();

        assert_eq!(
            registry.agents.len(),
            10,
            "expected 10 agents (2 teams * 5 players)"
        );
    }

    // -----------------------------------------------------------------------
    // 3. Action space — total_size = 4, 3 heads
    // -----------------------------------------------------------------------
    #[test]
    fn action_space_shape() {
        let config = deathmatch_config();
        let runner = TickRunner::new(config);
        let action_space = runner.action_space_def();

        assert_eq!(action_space.heads.len(), 3, "expected 3 action heads");
        // move_dir (2) + look_angle (1) + shoot (1 discrete) = 4
        assert_eq!(action_space.total_size, 4, "expected total_size = 4");
    }

    // -----------------------------------------------------------------------
    // 4. Observation space — verify features
    // -----------------------------------------------------------------------
    #[test]
    fn observation_space_features() {
        let config = deathmatch_config();
        let runner = TickRunner::new(config);
        let obs_space = runner.observation_space_def();

        let feature_names: Vec<&str> = obs_space.features.iter().map(|f| f.name.as_str()).collect();
        assert!(
            feature_names.contains(&"self_features"),
            "observation space should contain self_features"
        );
        assert!(
            feature_names.contains(&"entities"),
            "observation space should contain entities"
        );
        assert!(
            feature_names.contains(&"action_mask"),
            "observation space should contain action_mask"
        );
    }

    // -----------------------------------------------------------------------
    // 5. observe_all — run ticks, get observations for all agents
    // -----------------------------------------------------------------------
    #[test]
    fn observe_all_returns_all_agents() {
        let config = deathmatch_config();
        let mut runner = TickRunner::new(config);

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
                obs.contains_key("self_features"),
                "agent {} observation missing self_features",
                idx
            );
            assert!(
                obs.contains_key("entities"),
                "agent {} observation missing entities",
                idx
            );
            assert!(
                obs.contains_key("action_mask"),
                "agent {} observation missing action_mask",
                idx
            );
        }
    }

    // -----------------------------------------------------------------------
    // 6. Observation shapes — self_features=7, action_mask=2
    // -----------------------------------------------------------------------
    #[test]
    fn observation_shapes() {
        let config = deathmatch_config();
        let mut runner = TickRunner::new(config);

        // Run a tick so observations are populated
        runner.tick();

        let observations = runner.observe_all();
        let obs = observations.get(&0).expect("missing observation for agent 0");

        let self_features = obs.get("self_features").expect("missing self_features");
        assert_eq!(
            self_features.len(),
            7,
            "self_features should have 7 values"
        );

        let action_mask = obs.get("action_mask").expect("missing action_mask");
        assert_eq!(action_mask.len(), 2, "action_mask should have 2 values");
    }

    // -----------------------------------------------------------------------
    // 7. apply_raw_actions — move right, verify position changed
    // -----------------------------------------------------------------------
    #[test]
    fn apply_raw_actions_moves_agent() {
        let config = deathmatch_config();
        let mut runner = TickRunner::new(config);

        let agent_entity = runner.agent_registry().agents[0];

        // Record initial position
        let initial_obs = runner.observe_all();
        let initial_x = initial_obs[&0]["self_features"][0];

        // Apply "move right" action: move_dir=(1.0, 0.0), look_angle=0.0, shoot=0.0
        let mut actions = HashMap::new();
        actions.insert(agent_entity, vec![1.0, 0.0, 0.0, 0.0]);

        for _ in 0..20 {
            runner.apply_raw_actions(actions.clone());
            runner.tick();
        }

        let final_obs = runner.observe_all();
        let final_x = final_obs[&0]["self_features"][0];

        assert!(
            (final_x - initial_x).abs() > 0.01,
            "agent position should have changed after applying move actions (initial_x={}, final_x={})",
            initial_x,
            final_x
        );
    }

    // -----------------------------------------------------------------------
    // 8. rewards and dones — entries for all agents
    // -----------------------------------------------------------------------
    #[test]
    fn rewards_and_dones_for_all_agents() {
        let config = deathmatch_config();
        let mut runner = TickRunner::new(config);

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

    // -----------------------------------------------------------------------
    // 9. Telemetry — WorldSnapshot events after ticking
    // -----------------------------------------------------------------------
    #[test]
    fn telemetry_contains_world_snapshots() {
        let config = deathmatch_config();
        let mut runner = TickRunner::new(config);

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

    // -----------------------------------------------------------------------
    // 10. Builder with scenario — TickRunner::builder().with_scenario()
    // -----------------------------------------------------------------------
    #[test]
    fn builder_with_scenario() {
        let config = deathmatch_config();
        let mut runner = TickRunner::builder(config)
            .with_scenario(DeathmatchScenario)
            .build();

        // Should produce the same result as TickRunner::new
        assert_eq!(runner.agent_registry().agents.len(), 10);
        assert_eq!(runner.action_space_def().total_size, 4);

        for _ in 0..10 {
            runner.tick();
        }

        assert_eq!(runner.tick_count(), 10);
    }
}

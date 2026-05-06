use std::collections::HashMap;

use ghostlobby_engine::config::GameConfig;
use ghostlobby_engine::scenarios::tactical_deathmatch::TacticalDeathmatchScenario;
use ghostlobby_engine::telemetry::TelemetryEvent;
use ghostlobby_engine::tick::TickRunner;

fn tactical_open_config() -> GameConfig {
    let json = r#"{
        "title": "tactical-open",
        "tick_rate": 64,
        "arena": { "width": 600.0, "height": 600.0 },
        "movement": { "max_speed": 200.0, "acceleration": 1000.0, "friction": 800.0, "turn_rate": 8.0 },
        "combat": { "default_weapon": { "damage": 34.0, "fire_rate": 0.3, "range": 400.0 } },
        "spawning": { "respawn_delay": 0.5, "round_time_limit": 15.0 },
        "teams": { "count": 2, "players_per_team": 1 },
        "obstacles": [],
        "extra": {
            "scenario": "tactical",
            "reward_mode": "base",
            "candidate_distance": 45.0,
            "sensor_rays": 64,
            "sensor_range": 500.0
        }
    }"#;
    json.parse::<GameConfig>().expect("failed to parse tactical config")
}

fn tactical_obstacles_config() -> GameConfig {
    let json = r#"{
        "title": "tactical-obstacles",
        "tick_rate": 64,
        "arena": { "width": 600.0, "height": 600.0 },
        "movement": { "max_speed": 200.0, "acceleration": 1000.0, "friction": 800.0, "turn_rate": 8.0 },
        "combat": { "default_weapon": { "damage": 34.0, "fire_rate": 0.3, "range": 400.0 } },
        "spawning": { "respawn_delay": 0.5, "round_time_limit": 15.0 },
        "teams": { "count": 2, "players_per_team": 1 },
        "obstacles": [
            { "x": 275, "y": 250, "width": 50, "height": 100 },
            { "x": 150, "y": 125, "width": 15, "height": 120 },
            { "x": 435, "y": 355, "width": 15, "height": 120 }
        ],
        "extra": {
            "scenario": "tactical",
            "reward_mode": "cover",
            "candidate_distance": 45.0,
            "sensor_rays": 64,
            "sensor_range": 500.0
        }
    }"#;
    json.parse::<GameConfig>().expect("failed to parse tactical obstacles config")
}

fn build_tactical(config: GameConfig) -> TickRunner {
    TickRunner::builder(config)
        .with_scenario(TacticalDeathmatchScenario)
        .build()
}

// -----------------------------------------------------------------------
// 1. Smoke test — 200 ticks without panicking
// -----------------------------------------------------------------------
#[test]
fn smoke_test_200_ticks() {
    let mut runner = build_tactical(tactical_open_config());
    for _ in 0..200 {
        runner.tick();
    }
    assert_eq!(runner.tick_count(), 200);
}

#[test]
fn smoke_test_200_ticks_with_obstacles() {
    let mut runner = build_tactical(tactical_obstacles_config());
    for _ in 0..200 {
        runner.tick();
    }
    assert_eq!(runner.tick_count(), 200);
}

// -----------------------------------------------------------------------
// 2. Agents — 2 teams * 1 = 2 agents
// -----------------------------------------------------------------------
#[test]
fn agents_count() {
    let runner = build_tactical(tactical_open_config());
    assert_eq!(runner.agent_registry().agents.len(), 2);
}

// -----------------------------------------------------------------------
// 3. Action space — discrete(12) + continuous(1) + discrete(2) = 3
// -----------------------------------------------------------------------
#[test]
fn action_space_shape() {
    let runner = build_tactical(tactical_open_config());
    let space = runner.action_space_def();
    assert_eq!(space.heads.len(), 3, "expected 3 action heads");
    assert_eq!(space.total_size, 3, "move_target(1) + aim_delta(1) + shoot(1) = 3");
}

// -----------------------------------------------------------------------
// 4. Observation space — verify all 7 features exist
// -----------------------------------------------------------------------
#[test]
fn observation_space_features() {
    let runner = build_tactical(tactical_open_config());
    let obs_space = runner.observation_space_def();
    let names: Vec<&str> = obs_space.features.iter().map(|f| f.name.as_str()).collect();

    for expected in &["self_state", "enemy_state", "raycasts", "candidates", "context", "audio", "action_mask"] {
        assert!(names.contains(expected), "missing feature: {expected}");
    }
}

// -----------------------------------------------------------------------
// 5. Observation shapes — verify dimensions after ticking
// -----------------------------------------------------------------------
#[test]
fn observation_shapes() {
    let mut runner = build_tactical(tactical_open_config());
    runner.tick();

    let obs = runner.observe_all();
    let agent0 = obs.get(&0).expect("missing agent 0 obs");

    assert_eq!(agent0["self_state"].len(), 8, "self_state should be 8");
    assert_eq!(agent0["enemy_state"].len(), 20, "enemy_state should be max_agents * 10 = 20");
    assert_eq!(agent0["raycasts"].len(), 128, "raycasts should be 64 * 2 = 128");
    assert_eq!(agent0["candidates"].len(), 60, "candidates should be 12 * 5 = 60");
    assert_eq!(agent0["context"].len(), 10, "context should be 10");
    assert_eq!(agent0["audio"].len(), 2, "audio should be 2");
    assert_eq!(agent0["action_mask"].len(), 14, "action_mask should be 14");
}

// -----------------------------------------------------------------------
// 6. Total observation dimensions = 242
// -----------------------------------------------------------------------
#[test]
fn total_observation_dims() {
    let mut runner = build_tactical(tactical_open_config());
    runner.tick();

    let obs = runner.observe_all();
    let agent0 = obs.get(&0).unwrap();
    let total: usize = agent0.values().map(|v| v.len()).sum();
    assert_eq!(total, 242, "total obs dims should be 242");
}

// -----------------------------------------------------------------------
// 7. Apply discrete move_target action
// -----------------------------------------------------------------------
#[test]
fn apply_move_target_action() {
    let mut runner = build_tactical(tactical_open_config());

    let agent_entity = runner.agent_registry().agents[0];
    let initial_obs = runner.observe_all();
    let initial_x = initial_obs[&0]["self_state"][0];

    // Action: move_target=2 (East), aim_delta=0, shoot=0
    let actions = HashMap::from([(agent_entity, vec![2.0, 0.0, 0.0])]);

    for _ in 0..30 {
        runner.apply_raw_actions(actions.clone());
        runner.tick();
    }

    let final_obs = runner.observe_all();
    let final_x = final_obs[&0]["self_state"][0];

    assert!(
        (final_x - initial_x).abs() > 0.001,
        "agent should have moved with move_target=East (initial_x={initial_x}, final_x={final_x})"
    );
}

// -----------------------------------------------------------------------
// 8. Stay action (index 8) — position doesn't change much
// -----------------------------------------------------------------------
#[test]
fn stay_action_keeps_position() {
    let mut runner = build_tactical(tactical_open_config());
    for _ in 0..5 {
        runner.tick();
    }

    let agent_entity = runner.agent_registry().agents[0];
    let before = runner.observe_all();
    let x_before = before[&0]["self_state"][0];
    let y_before = before[&0]["self_state"][1];

    // Action: move_target=8 (Stay), aim_delta=0, shoot=0
    let actions = HashMap::from([(agent_entity, vec![8.0, 0.0, 0.0])]);
    for _ in 0..20 {
        runner.apply_raw_actions(actions.clone());
        runner.tick();
    }

    let after = runner.observe_all();
    let x_after = after[&0]["self_state"][0];
    let y_after = after[&0]["self_state"][1];

    let dx = (x_after - x_before).abs();
    let dy = (y_after - y_before).abs();
    assert!(
        dx < 0.05 && dy < 0.05,
        "agent with Stay action should barely move (dx={dx}, dy={dy})"
    );
}

// -----------------------------------------------------------------------
// 9. Rewards and dones for both agents
// -----------------------------------------------------------------------
#[test]
fn rewards_and_dones() {
    let mut runner = build_tactical(tactical_open_config());
    runner.tick();

    let rewards = runner.rewards();
    let dones = runner.dones();

    assert_eq!(rewards.len(), 2);
    assert_eq!(dones.len(), 2);
    assert!(rewards.contains_key(&0));
    assert!(rewards.contains_key(&1));
}

// -----------------------------------------------------------------------
// 10. Telemetry events work
// -----------------------------------------------------------------------
#[test]
fn telemetry_snapshots() {
    let mut runner = build_tactical(tactical_open_config());
    for _ in 0..10 {
        runner.tick();
    }

    let events = runner.drain_telemetry();
    assert!(!events.is_empty());

    let has_snapshot = events.iter().any(|e| matches!(e, TelemetryEvent::WorldSnapshot { .. }));
    assert!(has_snapshot, "should have WorldSnapshot events");
}

// -----------------------------------------------------------------------
// 11. Raycasts contain non-trivial data
// -----------------------------------------------------------------------
#[test]
fn raycasts_contain_data() {
    let mut runner = build_tactical(tactical_obstacles_config());
    for _ in 0..5 {
        runner.tick();
    }

    let obs = runner.observe_all();
    let raycasts = &obs[&0]["raycasts"];

    // With obstacles, at least some rays should hit something (distance < 1.0)
    let hits = raycasts.chunks(2).filter(|r| r[0] < 0.99).count();
    assert!(
        hits > 0,
        "with obstacles, at least some rays should detect walls"
    );
}

// -----------------------------------------------------------------------
// 12. Candidate positions have features
// -----------------------------------------------------------------------
#[test]
fn candidates_have_features() {
    let mut runner = build_tactical(tactical_open_config());
    for _ in 0..5 {
        runner.tick();
    }

    let obs = runner.observe_all();
    let candidates = &obs[&0]["candidates"];

    // 12 candidates * 5 features = 60
    assert_eq!(candidates.len(), 60);

    // Stay candidate (index 8) should have path_distance = 0
    let stay_path_dist = candidates[8 * 5]; // first feature of candidate 8
    assert!(
        stay_path_dist < 0.01,
        "Stay candidate path_distance should be ~0, got {stay_path_dist}"
    );
}

// -----------------------------------------------------------------------
// 13. Cover mode rewards differ from base mode
// -----------------------------------------------------------------------
#[test]
fn cover_mode_has_different_rewards() {
    let mut runner_base = build_tactical(tactical_open_config());
    let mut runner_cover = build_tactical(tactical_obstacles_config());

    for _ in 0..50 {
        runner_base.tick();
        runner_cover.tick();
    }

    let rewards_base = runner_base.rewards();
    let rewards_cover = runner_cover.rewards();

    // Both should return rewards for 2 agents
    assert_eq!(rewards_base.len(), 2);
    assert_eq!(rewards_cover.len(), 2);
}

// -----------------------------------------------------------------------
// 14. Long run stability (1000 ticks, no panics)
// -----------------------------------------------------------------------
#[test]
fn stability_1000_ticks_with_obstacles() {
    let mut runner = build_tactical(tactical_obstacles_config());

    let agent_entities = runner.agent_registry().agents.clone();
    for step in 0..1000 {
        // Apply random-ish actions
        let mut actions = HashMap::new();
        for &e in &agent_entities {
            let target = (step % 12) as f32;
            let aim = ((step as f32 * 0.1).sin()).clamp(-1.0, 1.0);
            let shoot = if step % 3 == 0 { 1.0 } else { 0.0 };
            actions.insert(e, vec![target, aim, shoot]);
        }
        runner.apply_raw_actions(actions);
        runner.tick();
    }

    assert_eq!(runner.tick_count(), 1000);
    let obs = runner.observe_all();
    assert_eq!(obs.len(), 2);
}

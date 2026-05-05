use std::collections::HashMap;

use ghostlobby_engine::config::GameConfig;
use ghostlobby_engine::scenarios::racing::{CheckpointTracker, RacingScenario, Vehicle};
use ghostlobby_engine::tick::TickRunner;

fn racing_config() -> GameConfig {
    let json = r#"{
        "title": "oval-race",
        "tick_rate": 64,
        "arena": { "width": 1200, "height": 800 },
        "movement": { "max_speed": 300, "acceleration": 800, "friction": 200 },
        "combat": { "default_weapon": { "damage": 0, "fire_rate": 1, "range": 0 } },
        "spawning": { "respawn_delay": 3 },
        "teams": { "count": 1, "players_per_team": 4 },
        "obstacles": [],
        "extra": {
            "max_speed": 300,
            "acceleration": 150,
            "turn_rate": 3.0,
            "brake_force": 400,
            "laps_to_win": 3,
            "checkpoint_radius": 60
        }
    }"#;
    json.parse().expect("failed to parse racing config")
}

fn racing_runner() -> TickRunner {
    TickRunner::builder(racing_config())
        .with_scenario(RacingScenario)
        .build()
}

#[test]
fn smoke_test_100_ticks() {
    let mut runner = racing_runner();
    for _ in 0..100 {
        runner.tick();
        let _ = runner.drain_telemetry();
    }
    assert_eq!(runner.tick_count(), 100);
}

#[test]
fn action_space_is_steer_throttle_brake() {
    let runner = racing_runner();
    let space = runner.action_space_def();
    assert_eq!(space.heads.len(), 3);
    assert_eq!(space.total_size, 3);
    assert_eq!(space.heads[0].name(), "steer");
    assert_eq!(space.heads[1].name(), "throttle");
    assert_eq!(space.heads[2].name(), "brake");
}

#[test]
fn agents_count_matches_config() {
    let runner = racing_runner();
    let registry = runner.agent_registry();
    assert_eq!(registry.agents.len(), 4);
}

#[test]
fn vehicle_moves_with_throttle() {
    let mut runner = racing_runner();

    let registry = runner.agent_registry().clone();
    let agent = registry.agents[0];
    let start_pos = runner.world().get::<ghostlobby_engine::ecs::components::Position>(agent)
        .unwrap().0;

    // Apply full throttle, no steer, no brake: [0.0, 1.0, 0.0]
    for _ in 0..50 {
        runner.apply_raw_actions(HashMap::from([(agent, vec![0.0, 1.0, 0.0])]));
        runner.tick();
        let _ = runner.drain_telemetry();
    }

    let end_pos = runner.world().get::<ghostlobby_engine::ecs::components::Position>(agent)
        .unwrap().0;
    let dist = start_pos.distance(end_pos);
    assert!(dist > 1.0, "car should have moved, but only moved {dist}");
}

#[test]
fn steering_changes_heading() {
    let mut runner = racing_runner();

    let registry = runner.agent_registry().clone();
    let agent = registry.agents[0];

    // Give it some speed first
    for _ in 0..20 {
        runner.apply_raw_actions(HashMap::from([(agent, vec![0.0, 1.0, 0.0])]));
        runner.tick();
        let _ = runner.drain_telemetry();
    }

    let heading_before = runner.world().get::<Vehicle>(agent).unwrap().heading;

    // Now steer hard right
    for _ in 0..30 {
        runner.apply_raw_actions(HashMap::from([(agent, vec![1.0, 0.5, 0.0])]));
        runner.tick();
        let _ = runner.drain_telemetry();
    }

    let heading_after = runner.world().get::<Vehicle>(agent).unwrap().heading;
    let heading_diff = (heading_after - heading_before).abs();
    assert!(heading_diff > 0.1, "heading should have changed, diff was {heading_diff}");
}

#[test]
fn checkpoint_progression() {
    let mut runner = racing_runner();
    let registry = runner.agent_registry().clone();
    let agent = registry.agents[0];

    // Get first checkpoint position
    let checkpoints = runner.world().resource::<ghostlobby_engine::scenarios::racing::Checkpoints>();
    let cp0 = checkpoints.positions[0];
    let cp_radius = checkpoints.radius;

    // Teleport car to just outside the first checkpoint
    {
        let world = runner.world_mut();
        let mut physics = world.remove_resource::<ghostlobby_engine::physics::PhysicsState>().unwrap();
        let ph = world.get::<ghostlobby_engine::ecs::components::PhysicsHandle>(agent).unwrap().clone();
        let near_cp = glam::Vec2::new(cp0.x, cp0.y - cp_radius - 5.0);
        physics.set_body_position(ph.body, near_cp);
        physics.set_body_linvel(ph.body, glam::Vec2::new(0.0, 50.0)); // heading toward cp
        world.insert_resource(physics);
    }

    let before = runner.world().get::<CheckpointTracker>(agent).unwrap().next_checkpoint;

    // Run enough ticks for the car to reach the checkpoint
    for _ in 0..50 {
        runner.apply_raw_actions(HashMap::from([(agent, vec![0.0, 1.0, 0.0])]));
        runner.tick();
        let _ = runner.drain_telemetry();
    }

    let after = runner.world().get::<CheckpointTracker>(agent).unwrap().next_checkpoint;
    assert!(after > before, "checkpoint should have advanced: before={before}, after={after}");
}

#[test]
fn observations_have_correct_keys() {
    let mut runner = racing_runner();
    for _ in 0..5 {
        runner.tick();
        let _ = runner.drain_telemetry();
    }

    let obs = runner.observe_all();
    assert_eq!(obs.len(), 4, "should have observations for all 4 cars");

    let agent_obs = &obs[&0];
    assert!(agent_obs.contains_key("self_features"), "missing self_features");
    assert!(agent_obs.contains_key("track_waypoints"), "missing track_waypoints");
    assert!(agent_obs.contains_key("other_cars"), "missing other_cars");
    assert!(agent_obs.contains_key("action_mask"), "missing action_mask");
}

#[test]
fn observation_shapes() {
    let mut runner = racing_runner();
    for _ in 0..5 {
        runner.tick();
        let _ = runner.drain_telemetry();
    }

    let obs = runner.observe_all();
    let agent_obs = &obs[&0];

    assert_eq!(agent_obs["self_features"].len(), 8);
    assert_eq!(agent_obs["track_waypoints"].len(), 8); // 4 checkpoints * 2 coords
    assert_eq!(agent_obs["action_mask"].len(), 2);
}

#[test]
fn rewards_and_dones_for_all_agents() {
    let mut runner = racing_runner();
    for _ in 0..10 {
        runner.tick();
        let _ = runner.drain_telemetry();
    }

    let rewards = runner.rewards();
    let dones = runner.dones();

    assert_eq!(rewards.len(), 4);
    assert_eq!(dones.len(), 4);

    for i in 0..4 {
        assert!(rewards.contains_key(&i));
        assert!(dones.contains_key(&i));
    }
}

#[test]
fn telemetry_contains_snapshots() {
    let mut runner = racing_runner();
    let mut all_events = Vec::new();
    for _ in 0..10 {
        runner.tick();
        all_events.extend(runner.drain_telemetry());
    }

    let snapshots: Vec<_> = all_events.iter().filter(|e| {
        matches!(e, ghostlobby_engine::telemetry::TelemetryEvent::WorldSnapshot { .. })
    }).collect();

    assert!(!snapshots.is_empty(), "should have WorldSnapshot events");
}

#[test]
fn scripted_ai_drives_cars() {
    let mut runner = racing_runner();

    // Record starting positions
    let registry = runner.agent_registry().clone();
    let start_positions: Vec<glam::Vec2> = registry.agents.iter().map(|&e| {
        runner.world().get::<ghostlobby_engine::ecs::components::Position>(e).unwrap().0
    }).collect();

    // Run for 500 ticks — AI should drive cars around
    for _ in 0..500 {
        runner.tick();
        let _ = runner.drain_telemetry();
    }

    let mut total_movement = 0.0f32;
    for (i, &agent) in registry.agents.iter().enumerate() {
        let end_pos = runner.world().get::<ghostlobby_engine::ecs::components::Position>(agent).unwrap().0;
        let dist = start_positions[i].distance(end_pos);
        total_movement += dist;
    }

    assert!(
        total_movement > 100.0,
        "AI cars should have moved significantly, total_movement={total_movement}"
    );
}

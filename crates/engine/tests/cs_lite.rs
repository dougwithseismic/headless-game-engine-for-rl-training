use ghostlobby_engine::config::GameConfig;
use ghostlobby_engine::scenarios::cs_lite::{CsLiteScenario, CsRoundState, RoundPhase};
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
            "ray_h_count": 5,
            "ray_v_count": 3,
            "hitbox_radius": 0.5
        }
    }"#;
    json.parse::<GameConfig>().expect("failed to parse cs_lite test config")
}

fn make_cs_runner() -> TickRunner {
    TickRunner::builder(cs_lite_config())
        .with_scenario(CsLiteScenario::default())
        .build()
}

#[test]
fn cs_lite_smoke_test_200_ticks() {
    let mut runner = make_cs_runner();
    for _ in 0..200 {
        runner.tick();
    }
    assert_eq!(runner.tick_count(), 200);
}

#[test]
fn cs_lite_has_10_agents() {
    let runner = make_cs_runner();
    let registry = runner.agent_registry();
    assert_eq!(registry.agents.len(), 10, "expected 10 agents (2 teams * 5)");
}

#[test]
fn cs_lite_starts_in_buy_freeze() {
    let runner = make_cs_runner();
    let round = runner.world().resource::<CsRoundState>();
    assert_eq!(round.phase, RoundPhase::BuyFreeze);
    assert_eq!(round.round_number, 1);
    assert_eq!(round.t_score, 0);
    assert_eq!(round.ct_score, 0);
}

#[test]
fn cs_lite_transitions_to_active() {
    let mut runner = make_cs_runner();
    // buy_time is 1.0s, tick_rate is 64, so ~64 ticks to transition
    for _ in 0..100 {
        runner.tick();
    }
    let round = runner.world().resource::<CsRoundState>();
    assert_eq!(round.phase, RoundPhase::Active, "should transition to Active after buy freeze");
}

#[test]
fn cs_lite_observations_have_correct_keys() {
    let runner = make_cs_runner();
    let obs = runner.observe_all();
    assert_eq!(obs.len(), 10);

    let agent_obs = obs.get(&0).expect("agent 0 obs missing");
    assert!(agent_obs.contains_key("self_state"), "missing self_state");
    assert!(agent_obs.contains_key("weapon_state"), "missing weapon_state");
    assert!(agent_obs.contains_key("teammate_state"), "missing teammate_state");
    assert!(agent_obs.contains_key("enemy_state"), "missing enemy_state");
    assert!(agent_obs.contains_key("round_info"), "missing round_info");
    assert!(agent_obs.contains_key("bomb_state"), "missing bomb_state");
    assert!(agent_obs.contains_key("candidates"), "missing candidates");
    assert!(agent_obs.contains_key("raycasts_3d"), "missing raycasts_3d");
    assert!(agent_obs.contains_key("audio_3d"), "missing audio_3d");
    assert!(agent_obs.contains_key("action_mask"), "missing action_mask");
}

#[test]
fn cs_lite_audio_3d_has_6_elements() {
    let runner = make_cs_runner();
    let obs = runner.observe_all();
    let agent_obs = obs.get(&0).expect("agent 0 obs missing");
    let audio = agent_obs.get("audio_3d").expect("audio_3d missing");
    assert_eq!(audio.len(), 6, "audio_3d should have 6 elements, got {}", audio.len());
}

#[test]
fn cs_lite_round_timeout_ct_wins() {
    let mut runner = make_cs_runner();
    // buy_time(1s) + round_time(10s) + end_time(1s) = 12s @ 64tps = 768 ticks
    // Run enough ticks to get through buy + full round + end
    for _ in 0..900 {
        runner.tick();
    }
    let round = runner.world().resource::<CsRoundState>();
    // After timeout, CT should win (defenders), round should advance
    assert!(round.round_number >= 2, "round should advance after timeout, got round {}", round.round_number);
    assert!(round.ct_score >= 1, "CT should win on timeout, ct_score={}", round.ct_score);
}

#[test]
fn cs_lite_config_driven_rewards() {
    let json = r#"{
        "title": "cs_lite reward test",
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
            "ray_h_count": 5,
            "ray_v_count": 3,
            "hitbox_radius": 0.5,
            "reward_kill": 99.0,
            "reward_near_miss": 0.5
        }
    }"#;
    let config = json.parse::<GameConfig>().expect("failed to parse");
    let mut runner = TickRunner::builder(config)
        .with_scenario(CsLiteScenario::default())
        .build();

    // Run past buy freeze into active combat
    for _ in 0..500 {
        runner.tick();
    }

    // Check that reward breakdown uses config values (not hardcoded 3.0)
    let breakdown = runner.reward_breakdown();
    for (_, components) in &breakdown {
        if let Some(&kill_val) = components.get("kill") {
            assert!(
                (kill_val - 99.0).abs() < 0.01 || (kill_val - 198.0).abs() < 0.01,
                "kill reward should be 99.0 per kill (from config), got {}",
                kill_val
            );
        }
        if let Some(&nm_val) = components.get("near_miss") {
            assert!(
                (nm_val - 0.5).abs() < 0.01,
                "near_miss reward should be 0.5 (from config), got {}",
                nm_val
            );
        }
    }
}

#[test]
fn cs_lite_1000_ticks_stable() {
    let mut runner = make_cs_runner();
    for _ in 0..1000 {
        runner.tick();
    }
    let round = runner.world().resource::<CsRoundState>();
    assert!(round.round_number >= 1, "should have completed at least 1 round");
    // Should not panic or crash
}

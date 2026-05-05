use std::collections::HashMap;
use std::time::Instant;

use tokio::sync::mpsc;
use tracing::{info, warn};

use ghostlobby_engine::config::GameConfig;
use ghostlobby_engine::scenarios::moba_lane::MobaLaneScenario;
use ghostlobby_engine::scenarios::racing::RacingScenario;
use ghostlobby_engine::tick::{TickMode, TickRunner};
use ghostlobby_telemetry::ws_sink::WsSink;
use ghostlobby_telemetry::TelemetrySink;

use crate::api::MatchResponse;
use crate::state::EngineCommand;

pub async fn run_tick_loop(
    config: GameConfig,
    mut command_rx: mpsc::Receiver<EngineCommand>,
    telemetry_ws: WsSink,
    file_sink: Option<ghostlobby_telemetry::file_sink::FileSink>,
    mode: TickMode,
) {
    let mut runner = build_runner(&config);
    runner.set_mode(mode.clone());

    let tick_interval = match &mode {
        TickMode::RealTime { rate } => {
            Some(tokio::time::interval(std::time::Duration::from_secs_f64(
                1.0 / *rate as f64,
            )))
        }
        _ => None,
    };

    let mut interval = tick_interval;
    let start = Instant::now();
    let mut last_log = Instant::now();

    info!(
        title = config.title,
        tick_rate = config.tick_rate,
        "engine started"
    );

    loop {
        if let Some(ref mut iv) = interval {
            iv.tick().await;
        }

        while let Ok(cmd) = command_rx.try_recv() {
            match cmd {
                EngineCommand::InjectRawAction { source_id, actions } => {
                    let world = runner.world_mut();
                    let mut target = None;
                    let mut query = world.query::<(
                        bevy_ecs::prelude::Entity,
                        &ghostlobby_engine::ecs::components::Agent,
                    )>();
                    for (entity, agent) in query.iter(world) {
                        if agent.source_id == source_id {
                            target = Some(entity);
                            break;
                        }
                    }
                    if let Some(entity) = target {
                        runner.apply_raw_actions(HashMap::from([(entity, actions)]));
                    } else {
                        warn!(source_id, "no agent with this source_id");
                    }
                }
                EngineCommand::GetStatus { reply } => {
                    let _ = reply.send(MatchResponse {
                        title: runner.config().title.clone(),
                        tick: runner.tick_count(),
                        tick_rate: runner.config().tick_rate,
                        status: "running".into(),
                    });
                }
                EngineCommand::Reset => {
                    info!("resetting engine");
                    runner = build_runner(&config);
                    runner.set_mode(mode.clone());
                }
            }
        }

        runner.tick();

        let events = runner.drain_telemetry();
        if !events.is_empty() {
            telemetry_ws.emit(&events);
            if let Some(ref fs) = file_sink {
                fs.emit(&events);
            }
        }

        if last_log.elapsed().as_secs() >= 5 {
            let elapsed = start.elapsed().as_secs_f64();
            let tps = runner.tick_count() as f64 / elapsed;
            info!(
                tick = runner.tick_count(),
                tps = format!("{:.0}", tps),
                "engine status"
            );
            last_log = Instant::now();
        }

        if matches!(mode, TickMode::Uncapped) && runner.tick_count().is_multiple_of(1000) {
            tokio::task::yield_now().await;
        }
    }
}

fn build_runner(config: &GameConfig) -> TickRunner {
    if config.title.contains("race") || config.title.contains("racing") {
        info!("using racing scenario");
        TickRunner::builder(config.clone())
            .with_scenario(RacingScenario)
            .build()
    } else if config.title.contains("moba") || config.title.contains("lane") {
        info!("using MOBA lane scenario");
        TickRunner::builder(config.clone())
            .with_scenario(MobaLaneScenario)
            .build()
    } else {
        TickRunner::new(config.clone())
    }
}

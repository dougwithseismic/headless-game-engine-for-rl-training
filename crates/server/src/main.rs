use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;
use tracing::info;

use ghostlobby_engine::config::GameConfig;
use ghostlobby_engine::tick::TickMode;
use ghostlobby_telemetry::file_sink::FileSink;
use ghostlobby_telemetry::ws_sink::WsSink;

mod api;
mod state;
mod tick_loop;
mod ws;

use state::AppState;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "configs/arena_deathmatch.json".into());

    let config = GameConfig::from_file(&config_path).unwrap_or_else(|e| {
        eprintln!("Failed to load config from {}: {}", config_path, e);
        std::process::exit(1);
    });

    info!(
        title = config.title,
        tick_rate = config.tick_rate,
        "loaded config"
    );

    let telemetry_ws = WsSink::new(4096);
    let file_sink = FileSink::new("telemetry.jsonl").ok();

    let (command_tx, command_rx) = tokio::sync::mpsc::channel(256);

    let app_state = Arc::new(AppState {
        command_tx,
        telemetry_sink: telemetry_ws,
        config: config.clone(),
    });

    let tick_telemetry = app_state.telemetry_sink.clone();

    tokio::spawn(tick_loop::run_tick_loop(
        config.clone(),
        command_rx,
        tick_telemetry,
        file_sink,
        TickMode::RealTime {
            rate: config.tick_rate,
        },
    ));

    let app = Router::new()
        .route("/api/health", get(api::health))
        .route("/api/match", get(api::match_status))
        .route("/api/match/reset", post(api::match_reset))
        .route("/api/config", get(api::config))
        .route("/ws/observe", get(ws::ws_observe))
        .route("/ws/play", get(ws::ws_play))
        .fallback_service(ServeDir::new("web-app/dist"))
        .layer(CorsLayer::permissive())
        .with_state(app_state);

    let addr = "0.0.0.0:3000";
    info!(addr, "server starting");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

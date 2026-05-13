use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use clap::Parser;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;
use tracing::info;

use ghostlobby_engine::config::GameConfig;
use ghostlobby_engine::tick::TickMode;
use ghostlobby_telemetry::file_sink::FileSink;
use ghostlobby_telemetry::ws_sink::WsSink;

mod api;
mod session_registry;
mod state;
mod tick_loop;
mod ws;

use state::AppState;

#[derive(Parser)]
#[command(name = "ghostlobby-server")]
struct Cli {
    /// Path to game config JSON
    #[arg(default_value = "configs/arena_deathmatch.json")]
    config: String,

    /// Port to listen on (0 = OS-assigned)
    #[arg(long, default_value_t = 3000)]
    port: u16,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();

    let config = GameConfig::from_file(&cli.config).unwrap_or_else(|e| {
        eprintln!("Failed to load config from {}: {}", cli.config, e);
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
        .route("/api/obstacles", get(api::obstacles))
        .route("/api/training", get(api::training))
        .route("/api/strategy", post(api::post_strategy))
        .route("/api/strategy/state", get(api::get_strategy_state))
        .route("/api/strategy/intents", get(api::get_strategy_intents))
        .route("/ws/observe", get(ws::ws_observe))
        .route("/ws/play", get(ws::ws_play))
        .fallback_service(ServeDir::new("web-app/dist"))
        .layer(CorsLayer::permissive())
        .with_state(app_state);

    let bind_addr = format!("0.0.0.0:{}", cli.port);
    let listener = tokio::net::TcpListener::bind(&bind_addr).await.unwrap();
    let actual_port = listener.local_addr().unwrap().port();
    info!(port = actual_port, "server starting");

    let scenario = config
        .extra
        .get("scenario")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let config_path = std::fs::canonicalize(&cli.config)
        .unwrap_or_else(|_| cli.config.clone().into())
        .display()
        .to_string();

    let pid = std::process::id();
    session_registry::register(&session_registry::SessionEntry {
        pid,
        port: actual_port,
        title: config.title.clone(),
        config_path,
        scenario,
        started_at: chrono_now(),
    });

    let server = axum::serve(listener, app);
    tokio::select! {
        result = server => { result.unwrap(); },
        _ = tokio::signal::ctrl_c() => {
            info!("shutting down");
        }
    }

    session_registry::unregister(pid);
}

fn chrono_now() -> String {
    use std::time::SystemTime;
    let d = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    let (days, rem) = (secs / 86400, secs % 86400);
    let (hours, rem) = (rem / 3600, rem % 3600);
    let (mins, s) = (rem / 60, rem % 60);
    // Approximate year/month/day from epoch days (good enough for display)
    let mut y = 1970u64;
    let mut d = days;
    loop {
        let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
        let ydays = if leap { 366 } else { 365 };
        if d < ydays {
            break;
        }
        d -= ydays;
        y += 1;
    }
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let mdays = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut m = 0u64;
    for md in mdays {
        if d < md {
            break;
        }
        d -= md;
        m += 1;
    }
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y,
        m + 1,
        d + 1,
        hours,
        mins,
        s
    )
}

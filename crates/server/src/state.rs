use tokio::sync::mpsc;

use ghostlobby_engine::config::GameConfig;
use ghostlobby_telemetry::ws_sink::WsSink;

use crate::api::MatchResponse;

pub enum EngineCommand {
    InjectRawAction {
        source_id: u32,
        actions: Vec<f32>,
    },
    GetStatus {
        reply: tokio::sync::oneshot::Sender<MatchResponse>,
    },
    Reset,
}

pub struct AppState {
    pub command_tx: mpsc::Sender<EngineCommand>,
    pub telemetry_sink: WsSink,
    pub config: GameConfig,
}

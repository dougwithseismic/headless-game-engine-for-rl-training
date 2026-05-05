use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use serde::Serialize;

use crate::state::{AppState, EngineCommand};

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
}

pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
    })
}

#[derive(Serialize)]
pub struct MatchResponse {
    pub title: String,
    pub tick: u64,
    pub tick_rate: u32,
    pub status: String,
}

pub async fn match_status(State(state): State<Arc<AppState>>) -> Json<MatchResponse> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let _ = state.command_tx.send(EngineCommand::GetStatus { reply: tx }).await;

    match rx.await {
        Ok(status) => Json(status),
        Err(_) => Json(MatchResponse {
            title: state.config.title.clone(),
            tick: 0,
            tick_rate: state.config.tick_rate,
            status: "unknown".into(),
        }),
    }
}

pub async fn match_reset(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let _ = state.command_tx.send(EngineCommand::Reset).await;
    Json(serde_json::json!({"status": "reset"}))
}

pub async fn config(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::to_value(&state.config).unwrap_or_default())
}

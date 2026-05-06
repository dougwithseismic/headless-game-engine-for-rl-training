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

#[derive(Serialize)]
pub struct ObstaclesResponse {
    pub obstacles: Vec<ObstacleRectDto>,
    pub spawn_points: Vec<[f32; 2]>,
}

#[derive(Serialize)]
pub struct ObstacleRectDto {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

pub async fn obstacles(State(state): State<Arc<AppState>>) -> Json<ObstaclesResponse> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let _ = state
        .command_tx
        .send(EngineCommand::GetObstacles { reply: tx })
        .await;

    match rx.await {
        Ok(resp) => Json(resp),
        Err(_) => Json(ObstaclesResponse {
            obstacles: vec![],
            spawn_points: vec![],
        }),
    }
}

pub async fn config(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::to_value(&state.config).unwrap_or_default())
}

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Serialize;

use ghostlobby_engine::strategy::{Directive, IntentSpec};

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

pub async fn training() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "model_version": 0,
        "phase": null,
        "phase_desc": "standalone",
        "last_reload_ago": 0
    }))
}

pub async fn post_strategy(
    State(state): State<Arc<AppState>>,
    Json(directive): Json<Directive>,
) -> StatusCode {
    let _ = state
        .command_tx
        .send(EngineCommand::InjectDirective { directive })
        .await;
    StatusCode::ACCEPTED
}

pub async fn get_strategy_state(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let _ = state
        .command_tx
        .send(EngineCommand::GetStrategyState { reply: tx })
        .await;
    match rx.await {
        Ok(Some(response)) => Json(response).into_response(),
        Ok(None) => StatusCode::NOT_FOUND.into_response(),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

pub async fn get_strategy_intents(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let _ = state
        .command_tx
        .send(EngineCommand::GetStrategyState { reply: tx })
        .await;
    match rx.await {
        Ok(Some(response)) => Json(response.intents).into_response(),
        Ok(None) => Json(Vec::<IntentSpec>::new()).into_response(),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

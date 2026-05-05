use glam::Vec2;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityState {
    pub id: u64,
    pub position: Vec2,
    pub velocity: Vec2,
    pub health: f32,
    pub max_health: f32,
    pub team: u8,
    pub is_dead: bool,
    pub facing: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TelemetryEvent {
    WorldSnapshot {
        tick: u64,
        entities: Vec<EntityState>,
    },
    Damage {
        tick: u64,
        source: u64,
        target: u64,
        amount: f32,
    },
    Kill {
        tick: u64,
        killer: u64,
        victim: u64,
    },
    Spawn {
        tick: u64,
        entity: u64,
        position: Vec2,
        team: u8,
    },
    ShotFired {
        tick: u64,
        shooter: u64,
        origin: Vec2,
        direction: Vec2,
        hit_target: Option<u64>,
    },
    TickComplete {
        tick: u64,
        entity_count: usize,
    },
}

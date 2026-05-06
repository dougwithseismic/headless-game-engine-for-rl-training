use std::collections::HashMap;

use bevy_ecs::prelude::*;
use glam::Vec2;

use crate::config::GameConfig;
use crate::telemetry::TelemetryEvent;

#[derive(Resource, Debug, Clone)]
pub struct WorldBounds {
    pub width: f32,
    pub height: f32,
}

impl WorldBounds {
    pub fn diagonal(&self) -> f32 {
        (self.width * self.width + self.height * self.height).sqrt()
    }
}

#[derive(Resource, Debug, Clone)]
pub struct TickState {
    pub tick: u64,
    pub delta: f32,
}

impl TickState {
    pub fn new(tick_rate: u32) -> Self {
        Self {
            tick: 0,
            delta: 1.0 / tick_rate as f32,
        }
    }
}

#[derive(Resource, Debug, Clone, Default)]
pub struct TelemetryBuffer {
    pub events: Vec<TelemetryEvent>,
}

impl TelemetryBuffer {
    pub fn push(&mut self, event: TelemetryEvent) {
        self.events.push(event);
    }

    pub fn drain(&mut self) -> Vec<TelemetryEvent> {
        std::mem::take(&mut self.events)
    }
}

#[derive(Resource, Debug, Clone)]
pub struct GameConfigResource(pub GameConfig);

#[derive(Resource, Debug, Clone, Default)]
pub struct RoundState {
    pub reset_timer: Option<f32>,
    pub round_clock: f32,
}

#[derive(Resource, Debug, Clone, Default)]
pub struct ObstacleLayout(pub Vec<crate::ecs::components::ObstacleRect>);

#[derive(Resource, Debug, Clone, Default)]
pub struct SpawnPointPool(pub Vec<glam::Vec2>);

#[derive(Resource, Debug, Clone, Default)]
pub struct ObstacleColliders(pub Vec<rapier2d::prelude::ColliderHandle>);

// --- Tactical scenario resources ---

#[derive(Debug, Clone, Copy)]
pub struct CandidatePosition {
    pub world_pos: Vec2,
    pub path_distance: f32,
    pub has_los_to_enemy: bool,
    pub dist_to_enemy: f32,
    pub enemies_with_los: f32,
}

impl Default for CandidatePosition {
    fn default() -> Self {
        Self {
            world_pos: Vec2::ZERO,
            path_distance: 1.0,
            has_los_to_enemy: false,
            dist_to_enemy: 1.0,
            enemies_with_los: 0.0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CandidateSet {
    pub positions: [CandidatePosition; 12],
}

impl CandidateSet {
    pub fn as_obs_features(&self) -> [f32; 60] {
        let mut out = [0.0f32; 60];
        for (i, c) in self.positions.iter().enumerate() {
            let base = i * 5;
            out[base] = c.path_distance;
            out[base + 1] = if c.has_los_to_enemy { 1.0 } else { 0.0 };
            out[base + 2] = if c.has_los_to_enemy { 0.0 } else { 1.0 };
            out[base + 3] = c.dist_to_enemy;
            out[base + 4] = c.enemies_with_los;
        }
        out
    }

    pub fn walkable_mask(&self) -> [f32; 12] {
        let mut mask = [1.0f32; 12];
        for (i, c) in self.positions.iter().enumerate() {
            if c.path_distance >= 1.0 && i != 8 {
                mask[i] = 0.0;
            }
        }
        mask
    }
}

#[derive(Resource, Debug, Clone, Default)]
pub struct CandidatePositionBuffer {
    pub candidates: HashMap<Entity, CandidateSet>,
}

impl CandidatePositionBuffer {
    pub fn get(&self, entity: Entity) -> Option<&CandidateSet> {
        self.candidates.get(&entity)
    }

    pub fn insert(&mut self, entity: Entity, set: CandidateSet) {
        self.candidates.insert(entity, set);
    }

    pub fn clear(&mut self) {
        self.candidates.clear();
    }
}

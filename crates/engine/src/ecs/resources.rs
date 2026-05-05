use bevy_ecs::prelude::*;

use crate::config::GameConfig;
use crate::telemetry::TelemetryEvent;

#[derive(Resource, Debug, Clone)]
pub struct WorldBounds {
    pub width: f32,
    pub height: f32,
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

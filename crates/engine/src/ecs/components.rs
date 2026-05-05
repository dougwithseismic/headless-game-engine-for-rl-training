use bevy_ecs::prelude::*;
use glam::Vec2;
use rapier2d::prelude as rapier;
use serde::{Deserialize, Serialize};

pub type ActionSourceId = u32;

#[derive(Component, Debug, Clone)]
pub struct Position(pub Vec2);

#[derive(Component, Debug, Clone)]
pub struct Velocity(pub Vec2);

#[derive(Component, Debug, Clone)]
pub struct Facing(pub f32);

#[derive(Component, Debug, Clone)]
pub struct Health {
    pub current: f32,
    pub max: f32,
}

#[derive(Component, Debug, Clone)]
pub struct Team(pub u8);

#[derive(Component, Debug, Clone)]
pub struct Weapon {
    pub damage: f32,
    pub fire_rate: f32,
    pub range: f32,
    pub cooldown_remaining: f32,
}

#[derive(Component, Debug, Clone)]
pub struct Agent {
    pub source_id: ActionSourceId,
}

#[derive(Component, Debug, Clone)]
pub struct Respawning {
    pub timer: f32,
}

#[derive(Component, Debug, Clone)]
pub struct Dead;

#[derive(Component, Debug, Clone)]
pub struct LastDamageSource(pub Entity);

#[derive(Component, Debug, Clone)]
pub struct PhysicsHandle {
    pub body: rapier::RigidBodyHandle,
    pub collider: rapier::ColliderHandle,
}

#[derive(Component, Debug, Clone)]
pub struct Obstacle;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ObstacleRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

use std::collections::{HashMap, VecDeque};

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

// --- Tactical scenario components ---

#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub last_seen_tick: u64,
    pub last_known_pos: Vec2,
}

#[derive(Component, Debug, Clone, Default)]
pub struct EnemyMemory {
    pub entries: HashMap<Entity, MemoryEntry>,
}

#[derive(Component, Debug, Clone, Default)]
pub struct PathState {
    pub waypoints: Vec<Vec2>,
    pub current_index: usize,
    pub target_candidate: Option<usize>,
}

impl PathState {
    pub fn current_waypoint(&self) -> Option<Vec2> {
        self.waypoints.get(self.current_index).copied()
    }

    pub fn advance(&mut self) {
        if self.current_index < self.waypoints.len() {
            self.current_index += 1;
        }
    }

    pub fn is_complete(&self) -> bool {
        self.current_index >= self.waypoints.len()
    }

    pub fn clear(&mut self) {
        self.waypoints.clear();
        self.current_index = 0;
        self.target_candidate = None;
    }
}

#[derive(Component, Debug, Clone, Default)]
pub struct VisitedCells {
    pub cells: VecDeque<(u16, u16, u64)>,
}

impl VisitedCells {
    pub fn record(&mut self, gx: u16, gy: u16, tick: u64) {
        if !self.cells.iter().any(|(x, y, _)| *x == gx && *y == gy) {
            self.cells.push_back((gx, gy, tick));
        }
    }

    pub fn prune(&mut self, current_tick: u64, window: u64) {
        while let Some(&(_, _, t)) = self.cells.front() {
            if current_tick - t > window {
                self.cells.pop_front();
            } else {
                break;
            }
        }
    }

    pub fn unique_count(&self) -> usize {
        self.cells.len()
    }
}

// --- Inventory / multi-weapon components ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeaponType {
    Rifle,
    Shotgun,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeaponSlot {
    pub weapon_type: WeaponType,
    pub damage: f32,
    pub fire_rate: f32,
    pub range: f32,
    pub cooldown_remaining: f32,
    pub ammo: u16,
    pub max_ammo: u16,
    pub reload_time: f32,
    pub reload_remaining: f32,
    pub is_reloading: bool,
}

impl WeaponSlot {
    pub fn ammo_fraction(&self) -> f32 {
        if self.max_ammo == 0 {
            1.0
        } else {
            self.ammo as f32 / self.max_ammo as f32
        }
    }

    pub fn cooldown_fraction(&self) -> f32 {
        if self.fire_rate <= 0.0 {
            0.0
        } else {
            self.cooldown_remaining / self.fire_rate
        }
    }

    pub fn reload_fraction(&self) -> f32 {
        if self.reload_time <= 0.0 {
            0.0
        } else {
            self.reload_remaining / self.reload_time
        }
    }
}

#[derive(Component, Debug, Clone)]
pub struct Inventory {
    pub weapons: Vec<WeaponSlot>,
    pub active: usize,
}

impl Inventory {
    pub fn active_weapon(&self) -> Option<&WeaponSlot> {
        self.weapons.get(self.active)
    }

    pub fn active_weapon_mut(&mut self) -> Option<&mut WeaponSlot> {
        self.weapons.get_mut(self.active)
    }
}

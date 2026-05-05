use bevy_ecs::prelude::*;
use glam::Vec2;

use crate::action_space::{ActionDict, RawActionBuffer};
use crate::ecs::components::*;
use crate::ecs::resources::*;

pub struct AiContext {
    pub my_position: Vec2,
    pub my_velocity: Vec2,
    pub my_team: u8,
    pub my_facing: f32,
    pub my_health: f32,
    pub weapon_range: f32,
    pub weapon_cooldown: f32,
    pub tick: u64,
    pub entity_bits: u64,
    pub arena_width: f32,
    pub arena_height: f32,
    pub enemies: Vec<TargetInfo>,
    pub allies: Vec<TargetInfo>,
}

pub struct TargetInfo {
    pub position: Vec2,
    pub distance: f32,
    pub health: f32,
}

pub type AiFn = Box<dyn Fn(&AiContext) -> ActionDict + Send + Sync>;

#[derive(Component)]
pub struct ScriptedAi(pub AiFn);

#[allow(clippy::type_complexity)]
pub fn run_scripted_ai(
    bots: Query<
        (Entity, &ScriptedAi, &Position, &Velocity, &Team, &Facing, &Health, &Weapon),
        Without<Dead>,
    >,
    targets: Query<(Entity, &Position, &Health, &Team), Without<Dead>>,
    bounds: Res<WorldBounds>,
    tick: Res<TickState>,
    mut raw_buffer: ResMut<RawActionBuffer>,
) {
    let all_entities: Vec<(Entity, Vec2, f32, u8)> = targets
        .iter()
        .map(|(e, p, h, t)| (e, p.0, h.current, t.0))
        .collect();

    for (entity, ai, pos, vel, team, facing, health, weapon) in &bots {
        let mut enemies = Vec::new();
        let mut allies = Vec::new();

        for &(e, e_pos, e_health, e_team) in &all_entities {
            if e == entity {
                continue;
            }
            let distance = pos.0.distance(e_pos);
            let info = TargetInfo {
                position: e_pos,
                distance,
                health: e_health,
            };
            if e_team == team.0 {
                allies.push(info);
            } else {
                enemies.push(info);
            }
        }

        let ctx = AiContext {
            my_position: pos.0,
            my_velocity: vel.0,
            my_team: team.0,
            my_facing: facing.0,
            my_health: health.current,
            weapon_range: weapon.range,
            weapon_cooldown: weapon.cooldown_remaining,
            tick: tick.tick,
            entity_bits: entity.to_bits(),
            arena_width: bounds.width,
            arena_height: bounds.height,
            enemies,
            allies,
        };

        let action_dict = (ai.0)(&ctx);
        raw_buffer.insert(entity, action_dict);
    }
}

// --- Built-in behaviors producing ActionDict ---
// Action layout (FPS): [move_x, move_y, look_angle, shoot]

pub fn aggressive_ai() -> AiFn {
    Box::new(|ctx: &AiContext| {
        let nearest = ctx
            .enemies
            .iter()
            .min_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        if let Some(enemy) = nearest {
            let dist = enemy.distance;
            let to_enemy = (enemy.position - ctx.my_position).normalize_or_zero();
            let angle = to_enemy.y.atan2(to_enemy.x);

            let mut repulsion = Vec2::ZERO;
            for ally in &ctx.allies {
                let to_other = ctx.my_position - ally.position;
                let d = to_other.length();
                if d < 60.0 && d > 0.1 {
                    repulsion += to_other.normalize() * 2.0 * (60.0 - d) / 60.0;
                }
            }
            for enemy_info in &ctx.enemies {
                let to_other = ctx.my_position - enemy_info.position;
                let d = to_other.length();
                if d < 60.0 && d > 0.1 {
                    repulsion += to_other.normalize() * 0.5 * (60.0 - d) / 60.0;
                }
            }

            let mut move_dir = if dist > ctx.weapon_range * 0.6 {
                to_enemy
            } else if dist < ctx.weapon_range * 0.2 {
                -to_enemy
            } else {
                let perp = Vec2::new(-to_enemy.y, to_enemy.x);
                let strafe_sign = if ctx.entity_bits.is_multiple_of(2) { 1.0 } else { -1.0 };
                perp * strafe_sign
            };

            move_dir = (move_dir + repulsion * 0.5).normalize_or_zero();

            if ctx.my_velocity.length() < 3.0 {
                let wander_phase = (ctx.tick as f32 * 0.02) + (ctx.entity_bits as f32);
                move_dir = Vec2::new(wander_phase.cos(), wander_phase.sin());
            }

            let move_dir = move_dir.normalize_or_zero();
            let shoot = if dist <= ctx.weapon_range && ctx.weapon_cooldown <= 0.0 {
                1.0
            } else {
                0.0
            };

            vec![move_dir.x, move_dir.y, angle, shoot]
        } else {
            let wander_phase = (ctx.tick as f32 * 0.01) + (ctx.entity_bits as f32);
            let dir = Vec2::new(wander_phase.cos(), wander_phase.sin());
            let angle = dir.y.atan2(dir.x);
            vec![dir.x, dir.y, angle, 0.0]
        }
    })
}

pub fn creep_ai(march_direction: f32) -> AiFn {
    Box::new(move |ctx: &AiContext| {
        let march_dir = Vec2::new(march_direction.cos(), march_direction.sin());

        let nearest_in_range = ctx
            .enemies
            .iter()
            .filter(|e| e.distance < ctx.weapon_range)
            .min_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        if let Some(enemy) = nearest_in_range {
            let to_enemy = (enemy.position - ctx.my_position).normalize_or_zero();
            let angle = to_enemy.y.atan2(to_enemy.x);
            let shoot = if ctx.weapon_cooldown <= 0.0 { 1.0 } else { 0.0 };
            vec![0.0, 0.0, angle, shoot]
        } else {
            vec![march_dir.x, march_dir.y, march_direction, 0.0]
        }
    })
}

pub fn passive_ai() -> AiFn {
    Box::new(|ctx: &AiContext| {
        let wander_phase = (ctx.tick as f32 * 0.005) + (ctx.entity_bits as f32);
        let dir = Vec2::new(wander_phase.cos(), wander_phase.sin());
        let angle = dir.y.atan2(dir.x);
        vec![dir.x, dir.y, angle, 0.0]
    })
}

pub fn racing_ai(checkpoints: Vec<Vec2>) -> AiFn {
    Box::new(move |ctx: &AiContext| {
        // Action layout: [steer, throttle, brake]
        // Pick the nearest checkpoint as the target (simple strategy)
        let target = checkpoints
            .iter()
            .min_by(|a, b| {
                let da = ctx.my_position.distance(**a);
                let db = ctx.my_position.distance(**b);
                da.partial_cmp(&db).unwrap()
            })
            .copied()
            .unwrap_or(ctx.my_position);

        let to_target = target - ctx.my_position;
        let desired_heading = to_target.y.atan2(to_target.x);

        let mut angle_diff = desired_heading - ctx.my_facing;
        // Normalize to [-PI, PI]
        while angle_diff > std::f32::consts::PI {
            angle_diff -= 2.0 * std::f32::consts::PI;
        }
        while angle_diff < -std::f32::consts::PI {
            angle_diff += 2.0 * std::f32::consts::PI;
        }

        let steer = angle_diff.clamp(-1.0, 1.0);
        let throttle = if angle_diff.abs() > 1.5 { 0.3 } else { 0.8 };
        let brake = 0.0;

        vec![steer, throttle, brake]
    })
}

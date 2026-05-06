use bevy_ecs::prelude::*;
use glam::Vec2;

use crate::action_space::{ActionDict, RawActionBuffer};
use crate::ecs::components::*;
use crate::ecs::resources::*;
use crate::physics::PhysicsState;

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
    pub wall_rays: [f32; 8],
}

pub struct TargetInfo {
    pub position: Vec2,
    pub distance: f32,
    pub health: f32,
}

pub type AiFn = Box<dyn Fn(&AiContext) -> ActionDict + Send + Sync>;

fn angle_to_turn_delta(current_facing: f32, desired_angle: f32) -> f32 {
    let mut diff = desired_angle - current_facing;
    if diff > std::f32::consts::PI {
        diff -= 2.0 * std::f32::consts::PI;
    }
    if diff < -std::f32::consts::PI {
        diff += 2.0 * std::f32::consts::PI;
    }
    (diff / std::f32::consts::PI).clamp(-1.0, 1.0)
}

#[derive(Component)]
pub struct ScriptedAi(pub AiFn);

#[allow(clippy::type_complexity)]
pub fn run_scripted_ai(
    bots: Query<
        (Entity, &ScriptedAi, &Position, &Velocity, &Team, &Facing, &Health, &Weapon, &PhysicsHandle),
        Without<Dead>,
    >,
    targets: Query<(Entity, &Position, &Health, &Team, &PhysicsHandle), Without<Dead>>,
    bounds: Res<WorldBounds>,
    tick: Res<TickState>,
    physics: Res<PhysicsState>,
    mut raw_buffer: ResMut<RawActionBuffer>,
) {
    let all_entities: Vec<_> = targets
        .iter()
        .map(|(e, p, h, t, ph)| (e, p.0, h.current, t.0, ph.collider))
        .collect();

    for (entity, ai, pos, vel, team, facing, health, weapon, bot_ph) in &bots {
        let mut enemies = Vec::new();
        let mut allies = Vec::new();

        for &(e, e_pos, e_health, e_team, e_collider) in &all_entities {
            if e == entity {
                continue;
            }
            let distance = pos.0.distance(e_pos);

            let visible = if distance > 0.1 {
                let dir = (e_pos - pos.0).normalize_or_zero();
                match physics.cast_ray(pos.0, dir, distance, Some(bot_ph.collider)) {
                    Some((hit_col, _)) => hit_col == e_collider,
                    None => true,
                }
            } else {
                true
            };

            if !visible {
                continue;
            }

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

        let scan_range = 120.0;
        let mut wall_rays = [1.0f32; 8];
        for (i, ray) in wall_rays.iter_mut().enumerate() {
            let angle = i as f32 * std::f32::consts::FRAC_PI_4;
            let dir = Vec2::new(angle.cos(), angle.sin());
            if let Some((_, toi)) = physics.cast_ray(pos.0, dir, scan_range, Some(bot_ph.collider))
            {
                *ray = toi / scan_range;
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
            wall_rays,
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
                let phase = (ctx.tick / 90 + ctx.entity_bits) % 3;
                let strafe_sign = if phase == 0 { 1.0 } else if phase == 1 { -1.0 } else { 0.3 };
                perp * strafe_sign + to_enemy * 0.2
            };

            move_dir = (move_dir + repulsion * 0.5).normalize_or_zero();

            let desired = move_dir;
            let mut best_score = f32::MIN;
            let mut best_dir = desired;
            for i in 0..8 {
                let a = i as f32 * std::f32::consts::FRAC_PI_4;
                let dir = Vec2::new(a.cos(), a.sin());
                let interest = dir.dot(desired);
                let danger = (1.0 - ctx.wall_rays[i]).max(0.0);
                let score = interest - danger * 3.0;
                if score > best_score {
                    best_score = score;
                    best_dir = dir;
                }
            }

            let move_dir = best_dir;
            let shoot = if dist <= ctx.weapon_range && ctx.weapon_cooldown <= 0.0 {
                1.0
            } else {
                0.0
            };

            let turn_delta = angle_to_turn_delta(ctx.my_facing, angle);
            vec![move_dir.x, move_dir.y, turn_delta, shoot]
        } else {
            let epoch = ctx.tick / 128;
            let hash = (epoch.wrapping_mul(2654435761) ^ ctx.entity_bits) as f32;
            let margin = 80.0;
            let goal = Vec2::new(
                margin + (hash % 1000.0) / 1000.0 * (ctx.arena_width - margin * 2.0),
                margin + ((hash / 7.0) % 1000.0) / 1000.0 * (ctx.arena_height - margin * 2.0),
            );
            let to_goal = (goal - ctx.my_position).normalize_or_zero();

            let mut best_score = f32::MIN;
            let mut best_dir = to_goal;
            for i in 0..8 {
                let a = i as f32 * std::f32::consts::FRAC_PI_4;
                let dir = Vec2::new(a.cos(), a.sin());
                let interest = dir.dot(to_goal);
                let danger = (1.0 - ctx.wall_rays[i]).max(0.0);
                let score = interest - danger * 3.0;
                if score > best_score {
                    best_score = score;
                    best_dir = dir;
                }
            }

            let desired_angle = best_dir.y.atan2(best_dir.x);
            let turn_delta = angle_to_turn_delta(ctx.my_facing, desired_angle);
            vec![best_dir.x, best_dir.y, turn_delta, 0.0]
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
            let turn_delta = angle_to_turn_delta(ctx.my_facing, angle);
            let shoot = if ctx.weapon_cooldown <= 0.0 { 1.0 } else { 0.0 };
            vec![0.0, 0.0, turn_delta, shoot]
        } else {
            let turn_delta = angle_to_turn_delta(ctx.my_facing, march_direction);
            vec![march_dir.x, march_dir.y, turn_delta, 0.0]
        }
    })
}

pub fn passive_ai() -> AiFn {
    Box::new(|ctx: &AiContext| {
        let wander_phase = (ctx.tick as f32 * 0.005) + (ctx.entity_bits as f32);
        let dir = Vec2::new(wander_phase.cos(), wander_phase.sin());
        let desired = dir.y.atan2(dir.x);
        let turn_delta = angle_to_turn_delta(ctx.my_facing, desired);
        vec![dir.x, dir.y, turn_delta, 0.0]
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

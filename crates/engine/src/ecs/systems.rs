use bevy_ecs::prelude::*;
use glam::Vec2;
use rand::Rng;

use crate::action_space::{ActionMaskBuffer, ActionSpaceDef, RawActionBuffer};
use crate::ecs::components::*;
use crate::ecs::resources::*;
use crate::observation::RewardBuffer;
use crate::physics::PhysicsState;
use crate::telemetry::{EntityState, TelemetryEvent};

pub fn clear_buffers(
    mut raw_buffer: ResMut<RawActionBuffer>,
    mut mask_buffer: ResMut<ActionMaskBuffer>,
    mut reward_buffer: ResMut<RewardBuffer>,
) {
    raw_buffer.clear();
    mask_buffer.clear();
    reward_buffer.clear();
}

pub fn sync_actions_to_physics(
    movers: Query<(Entity, &PhysicsHandle), Without<Dead>>,
    raw_buffer: Res<RawActionBuffer>,
    action_space: Res<ActionSpaceDef>,
    config: Res<GameConfigResource>,
    mut physics: ResMut<PhysicsState>,
) {
    let mc = &config.0.movement;

    for (entity, ph) in &movers {
        if let Some(raw) = raw_buffer.get(entity)
            && raw.len() >= action_space.total_size
        {
            let move_slice = action_space.extract_head(raw, 0);
            if move_slice.len() >= 2 {
                let dir = Vec2::new(move_slice[0], move_slice[1]);
                if dir.length_squared() > 0.001 {
                    let target_vel = dir.normalize_or_zero() * mc.max_speed;
                    physics.set_body_linvel(ph.body, target_vel);
                }
            }
        }
    }
}

pub fn physics_step(mut physics: ResMut<PhysicsState>) {
    physics.step();
}

pub fn sync_physics_to_ecs(
    mut query: Query<(&PhysicsHandle, &mut Position, &mut Velocity)>,
    physics: Res<PhysicsState>,
) {
    for (ph, mut pos, mut vel) in &mut query {
        if let Some(p) = physics.body_position(ph.body) {
            pos.0 = p;
        }
        if let Some(v) = physics.body_velocity(ph.body) {
            vel.0 = v;
        }
    }
}

pub fn facing_system(
    mut query: Query<(Entity, &mut Facing), Without<Dead>>,
    raw_buffer: Res<RawActionBuffer>,
    action_space: Res<ActionSpaceDef>,
) {
    for (entity, mut facing) in &mut query {
        if let Some(raw) = raw_buffer.get(entity)
            && raw.len() >= action_space.total_size
            && action_space.heads.len() > 1
        {
            let look_slice = action_space.extract_head(raw, 1);
            if !look_slice.is_empty() {
                facing.0 = look_slice[0];
            }
        }
    }
}

pub fn weapon_cooldown_system(mut query: Query<&mut Weapon>, tick: Res<TickState>) {
    for mut weapon in &mut query {
        weapon.cooldown_remaining = (weapon.cooldown_remaining - tick.delta).max(0.0);
    }
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn combat_system(
    mut commands: Commands,
    mut shooters: Query<
        (Entity, &Position, &Facing, &mut Weapon, &Team, &PhysicsHandle),
        Without<Dead>,
    >,
    mut targets: Query<(Entity, &Position, &mut Health, &Team, &PhysicsHandle), Without<Dead>>,
    raw_buffer: Res<RawActionBuffer>,
    action_space: Res<ActionSpaceDef>,
    physics: Res<PhysicsState>,
    tick: Res<TickState>,
    mut telemetry: ResMut<TelemetryBuffer>,
    mut rewards: ResMut<RewardBuffer>,
) {
    let mut hits: Vec<(Entity, f32, Entity)> = Vec::new();

    for (shooter_entity, shooter_pos, facing, mut weapon, shooter_team, shooter_ph) in
        &mut shooters
    {
        let mut wants_shoot = false;

        if let Some(raw) = raw_buffer.get(shooter_entity)
            && raw.len() >= action_space.total_size
            && action_space.heads.len() > 2
        {
            let shoot_slice = action_space.extract_head(raw, 2);
            if !shoot_slice.is_empty() && shoot_slice[0] > 0.5 {
                wants_shoot = true;
            }
        }

        if !wants_shoot || weapon.cooldown_remaining > 0.0 {
            continue;
        }

        weapon.cooldown_remaining = weapon.fire_rate;

        let dir = Vec2::new(facing.0.cos(), facing.0.sin());
        let range = weapon.range;
        let damage = weapon.damage;

        let mut best_hit: Option<(Entity, f32)> = None;

        for (target_entity, target_pos, _health, target_team, _target_ph) in &targets {
            if target_entity == shooter_entity || target_team.0 == shooter_team.0 {
                continue;
            }

            let to_target = target_pos.0 - shooter_pos.0;
            let dist = to_target.length();
            if dist > range {
                continue;
            }

            let proj = to_target.dot(dir);
            if proj < 0.0 {
                continue;
            }
            let perp_dist = (to_target - dir * proj).length();
            let hitbox_radius = 15.0;

            if perp_dist <= hitbox_radius {
                let occluded = if let Some((hit_collider, hit_toi)) =
                    physics.cast_ray(shooter_pos.0, dir, proj, Some(shooter_ph.collider))
                {
                    let target_collider = targets
                        .get(target_entity)
                        .ok()
                        .and_then(|t| physics.collider_for_body(t.4.body));
                    target_collider.is_none_or(|tc| hit_collider != tc) && hit_toi < proj
                } else {
                    false
                };

                if !occluded
                    && (best_hit.is_none() || proj < best_hit.unwrap().1)
                {
                    best_hit = Some((target_entity, proj));
                }
            }
        }

        telemetry.push(TelemetryEvent::ShotFired {
            tick: tick.tick,
            shooter: shooter_entity.to_bits(),
            origin: shooter_pos.0,
            direction: dir,
            hit_target: best_hit.map(|(e, _)| e.to_bits()),
        });

        if let Some((hit_entity, _)) = best_hit {
            hits.push((hit_entity, damage, shooter_entity));
        }
    }

    for &(hit_entity, damage, shooter_entity) in &hits {
        if let Ok((_entity, _pos, mut health, _team, _ph)) = targets.get_mut(hit_entity) {
            let max_hp = health.max;
            health.current -= damage;
            telemetry.push(TelemetryEvent::Damage {
                tick: tick.tick,
                source: shooter_entity.to_bits(),
                target: hit_entity.to_bits(),
                amount: damage,
            });
            rewards.add(shooter_entity, 0.1 * damage / max_hp);
        }
        commands
            .entity(hit_entity)
            .insert(LastDamageSource(shooter_entity));
    }
}

pub fn death_system(
    mut commands: Commands,
    query: Query<(Entity, &Health, Option<&LastDamageSource>), Without<Dead>>,
    config: Res<GameConfigResource>,
    tick: Res<TickState>,
    mut telemetry: ResMut<TelemetryBuffer>,
    mut rewards: ResMut<RewardBuffer>,
) {
    for (entity, health, last_source) in &query {
        if health.current <= 0.0 {
            let killer_bits = last_source.map(|s| s.0.to_bits()).unwrap_or(0);
            if let Some(source) = last_source {
                rewards.add(source.0, 1.0);
            }
            rewards.add(entity, -1.0);
            commands.entity(entity).insert((
                Dead,
                Respawning {
                    timer: config.0.spawning.respawn_delay,
                },
            ));
            telemetry.push(TelemetryEvent::Kill {
                tick: tick.tick,
                killer: killer_bits,
                victim: entity.to_bits(),
            });
        }
    }
}

#[allow(clippy::type_complexity)]
pub fn respawn_system(
    mut commands: Commands,
    mut query: Query<(
        Entity,
        &mut Respawning,
        &mut Health,
        &mut Position,
        &Team,
        &PhysicsHandle,
    )>,
    tick: Res<TickState>,
    bounds: Res<WorldBounds>,
    mut physics: ResMut<PhysicsState>,
    mut telemetry: ResMut<TelemetryBuffer>,
) {
    let mut rng = rand::rng();

    for (entity, mut respawning, mut health, mut pos, team, ph) in &mut query {
        respawning.timer -= tick.delta;
        if respawning.timer <= 0.0 {
            health.current = health.max;

            let spawn_x = if team.0 == 0 {
                rng.random_range(50.0..150.0)
            } else {
                rng.random_range((bounds.width - 150.0)..(bounds.width - 50.0))
            };
            let spawn_y = rng.random_range(100.0..(bounds.height - 100.0));
            let new_pos = Vec2::new(spawn_x, spawn_y);
            pos.0 = new_pos;
            physics.set_body_position(ph.body, new_pos);
            physics.set_body_linvel(ph.body, Vec2::ZERO);

            commands.entity(entity).remove::<(Dead, Respawning)>();

            telemetry.push(TelemetryEvent::Spawn {
                tick: tick.tick,
                entity: entity.to_bits(),
                position: pos.0,
                team: team.0,
            });
        }
    }
}

#[allow(clippy::type_complexity)]
pub fn telemetry_snapshot_system(
    query: Query<(
        Entity,
        &Position,
        &Velocity,
        &Health,
        &Team,
        &Facing,
        Option<&Dead>,
    )>,
    tick: Res<TickState>,
    mut telemetry: ResMut<TelemetryBuffer>,
) {
    if !tick.tick.is_multiple_of(2) {
        return;
    }

    let entities: Vec<EntityState> = query
        .iter()
        .map(|(entity, pos, vel, health, team, facing, dead)| EntityState {
            id: entity.to_bits(),
            position: pos.0,
            velocity: vel.0,
            health: health.current,
            max_health: health.max,
            team: team.0,
            is_dead: dead.is_some(),
            facing: facing.0,
        })
        .collect();

    telemetry.push(TelemetryEvent::WorldSnapshot {
        tick: tick.tick,
        entities,
    });
}

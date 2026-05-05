use bevy_ecs::prelude::*;
use glam::Vec2;

use crate::action_space::{ActionHead, ActionSpaceDef};
use crate::config::GameConfig;
use crate::ecs::components::*;
use crate::ecs::resources::*;
use crate::ecs::systems;
use crate::observation::{ObsFeature, ObsWriter, ObservationSpaceDef, ShotEventBuffer};
use crate::physics::PhysicsState;
use crate::scripted_ai::{aggressive_ai, ScriptedAi};
use crate::tick::EnginePhase;

pub trait Scenario: Send + Sync {
    fn name(&self) -> &str;
    fn action_space(&self, config: &GameConfig) -> ActionSpaceDef;
    fn observation_space(&self, config: &GameConfig) -> ObservationSpaceDef;
    fn setup(&self, world: &mut World, config: &GameConfig, physics: &mut PhysicsState);
    fn register_systems(&self, schedule: &mut Schedule);
    fn observe(&self, world: &World, agent: Entity, writer: &mut ObsWriter);
    fn reward(&self, _world: &World, _agent: Entity) -> f32 {
        0.0
    }
    fn is_done(&self, _world: &World, _agent: Entity) -> bool {
        false
    }
}

pub struct DeathmatchScenario;

impl Scenario for DeathmatchScenario {
    fn name(&self) -> &str {
        "deathmatch"
    }

    fn action_space(&self, _config: &GameConfig) -> ActionSpaceDef {
        ActionSpaceDef::new(vec![
            ActionHead::Continuous {
                name: "move_dir".into(),
                size: 2,
                low: vec![-1.0, -1.0],
                high: vec![1.0, 1.0],
            },
            ActionHead::Continuous {
                name: "look_angle".into(),
                size: 1,
                low: vec![-std::f32::consts::PI],
                high: vec![std::f32::consts::PI],
            },
            ActionHead::Discrete {
                name: "shoot".into(),
                n: 2,
            },
        ])
    }

    fn observation_space(&self, config: &GameConfig) -> ObservationSpaceDef {
        let max_agents =
            (config.teams.count as usize) * (config.teams.players_per_team as usize);
        ObservationSpaceDef {
            features: vec![
                ObsFeature {
                    name: "self_features".into(),
                    shape: vec![11],
                },
                ObsFeature {
                    name: "entities".into(),
                    shape: vec![max_agents, 10],
                },
                ObsFeature {
                    name: "audio".into(),
                    shape: vec![2],
                },
                ObsFeature {
                    name: "action_mask".into(),
                    shape: vec![2],
                },
            ],
        }
    }

    fn setup(&self, world: &mut World, config: &GameConfig, physics: &mut PhysicsState) {
        setup_world(world, config, physics);
    }

    fn register_systems(&self, schedule: &mut Schedule) {
        schedule.add_systems(
            (
                systems::facing_system,
                systems::weapon_cooldown_system,
                systems::combat_system,
            )
                .chain()
                .in_set(EnginePhase::GameLogic),
        );
        schedule.add_systems(
            (systems::death_system, systems::respawn_system)
                .chain()
                .in_set(EnginePhase::StateTransitions),
        );
    }

    fn observe(&self, world: &World, agent: Entity, writer: &mut ObsWriter) {
        let bounds = world.resource::<WorldBounds>();
        let config = world.resource::<GameConfigResource>();
        let physics = world.resource::<PhysicsState>();
        let max_speed = config.0.movement.max_speed;
        let arena_diag = (bounds.width * bounds.width + bounds.height * bounds.height).sqrt();

        let pos = world.get::<Position>(agent).map(|p| p.0).unwrap_or_default();
        let vel = world.get::<Velocity>(agent).map(|v| v.0).unwrap_or_default();
        let health = world.get::<Health>(agent);
        let facing = world.get::<Facing>(agent).map(|f| f.0).unwrap_or(0.0);
        let team = world.get::<Team>(agent).map(|t| t.0).unwrap_or(0);
        let hp = health.map(|h| h.current).unwrap_or(0.0);
        let max_hp = health.map(|h| h.max).unwrap_or(1.0);
        let weapon = world.get::<Weapon>(agent);
        let cooldown_norm = weapon
            .map(|w| {
                if w.fire_rate > 0.0 {
                    w.cooldown_remaining / w.fire_rate
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0);
        let agent_collider = world.get::<PhysicsHandle>(agent).map(|ph| ph.collider);

        let registry = world.resource::<crate::observation::AgentRegistry>();
        let agent_list: Vec<Entity> = registry.agents.clone();
        let max_agents = registry.max_agents;

        let face_dir = Vec2::new(facing.cos(), facing.sin());
        let mut nearest_dist = f32::MAX;
        let mut nearest_bearing = 0.0f32;
        let mut entity_data = Vec::new();

        for &e in &agent_list {
            if e == agent {
                continue;
            }

            let e_pos = world.get::<Position>(e).map(|p| p.0).unwrap_or_default();
            let e_dead = world.get::<Dead>(e).is_some();
            let delta = e_pos - pos;
            let dist = delta.length();

            let visible = if !e_dead && dist > 0.1 {
                let dir = delta / dist;
                let e_collider = world.get::<PhysicsHandle>(e).map(|ph| ph.collider);
                match physics.cast_ray(pos, dir, dist, agent_collider) {
                    Some((hit_col, _)) => e_collider.is_some_and(|ec| hit_col == ec),
                    None => true,
                }
            } else {
                false
            };

            if !visible {
                entity_data.extend_from_slice(&[0.0; 10]);
                continue;
            }

            let e_team = world.get::<Team>(e).map(|t| t.0).unwrap_or(0);
            let to_enemy = delta.normalize_or_zero();
            let cross = face_dir.x * to_enemy.y - face_dir.y * to_enemy.x;
            let bearing = cross.atan2(face_dir.dot(to_enemy));

            if e_team != team && dist < nearest_dist {
                nearest_dist = dist;
                nearest_bearing = bearing;
            }

            let e_vel = world.get::<Velocity>(e).map(|v| v.0).unwrap_or_default();
            let e_health = world.get::<Health>(e);
            let e_facing = world.get::<Facing>(e).map(|f| f.0).unwrap_or(0.0);
            let e_weapon = world.get::<Weapon>(e);
            let e_hp = e_health.map(|h| (h.current / h.max).max(0.0)).unwrap_or(0.0);
            let e_cooldown = e_weapon
                .map(|w| {
                    if w.fire_rate > 0.0 {
                        w.cooldown_remaining / w.fire_rate
                    } else {
                        0.0
                    }
                })
                .unwrap_or(0.0);

            entity_data.extend_from_slice(&[
                delta.x / arena_diag,
                delta.y / arena_diag,
                dist / arena_diag,
                bearing / std::f32::consts::PI,
                e_vel.x / max_speed,
                e_vel.y / max_speed,
                e_hp,
                e_facing.sin(),
                e_facing.cos(),
                e_cooldown,
            ]);
        }

        let nearest_dist_norm = if nearest_dist < f32::MAX {
            nearest_dist / arena_diag
        } else {
            1.0
        };

        writer.write("self_features", &[
            pos.x / bounds.width,
            pos.y / bounds.height,
            vel.x / max_speed,
            vel.y / max_speed,
            hp / max_hp,
            facing.sin(),
            facing.cos(),
            team as f32,
            cooldown_norm,
            nearest_dist_norm,
            nearest_bearing / std::f32::consts::PI,
        ]);

        writer.write_padded("entities", &entity_data, max_agents * 10);

        let shot_buffer = world.resource::<ShotEventBuffer>();
        let mut shot_bearing = 0.0f32;
        let mut shot_proximity = 0.0f32;
        for event in &shot_buffer.events {
            if event.shooter == agent {
                continue;
            }
            let delta = event.origin - pos;
            let d = delta.length();
            if d < arena_diag {
                let to_shot = delta.normalize_or_zero();
                let cross = face_dir.x * to_shot.y - face_dir.y * to_shot.x;
                shot_bearing = cross.atan2(face_dir.dot(to_shot));
                shot_proximity = 1.0 - d / arena_diag;
            }
        }
        writer.write("audio", &[
            shot_bearing / std::f32::consts::PI,
            shot_proximity,
        ]);

        let can_shoot = weapon.map(|w| w.cooldown_remaining <= 0.0).unwrap_or(false);
        let is_dead = world.get::<Dead>(agent).is_some();
        writer.write("action_mask", &[
            if is_dead { 0.0 } else { 1.0 },
            if can_shoot && !is_dead { 1.0 } else { 0.0 },
        ]);
    }

    fn reward(&self, world: &World, agent: Entity) -> f32 {
        let combat_reward = world
            .resource::<crate::observation::RewardBuffer>()
            .get(agent);

        if world.get::<Dead>(agent).is_some() {
            return combat_reward;
        }

        let pos = world.get::<Position>(agent).map(|p| p.0).unwrap_or_default();
        let team = world.get::<Team>(agent).map(|t| t.0).unwrap_or(0);
        let facing = world.get::<Facing>(agent).map(|f| f.0).unwrap_or(0.0);
        let bounds = world.resource::<WorldBounds>();

        let mut nearest_dist = f32::MAX;
        let mut nearest_dir = glam::Vec2::ZERO;
        for &e in &world.resource::<crate::observation::AgentRegistry>().agents {
            if e == agent {
                continue;
            }
            let e_team = world.get::<Team>(e).map(|t| t.0).unwrap_or(0);
            if e_team == team || world.get::<Dead>(e).is_some() {
                continue;
            }
            let e_pos = world.get::<Position>(e).map(|p| p.0).unwrap_or_default();
            let d = pos.distance(e_pos);
            if d < nearest_dist {
                nearest_dist = d;
                nearest_dir = (e_pos - pos).normalize_or_zero();
            }
        }

        let mut shaping = 0.0;

        if nearest_dist < f32::MAX {
            let arena_diag = (bounds.width * bounds.width + bounds.height * bounds.height).sqrt();
            shaping += 0.003 * (1.0 - nearest_dist / arena_diag);

            let face_dir = glam::Vec2::new(facing.cos(), facing.sin());
            let aim_dot = face_dir.dot(nearest_dir);
            shaping += 0.002 * aim_dot.max(0.0);
        }

        // Penalty for hugging walls
        let wall_margin = 30.0;
        let wall_penalty = [
            (wall_margin - pos.x).max(0.0),
            (wall_margin - pos.y).max(0.0),
            (pos.x - (bounds.width - wall_margin)).max(0.0),
            (pos.y - (bounds.height - wall_margin)).max(0.0),
        ]
        .iter()
        .map(|d| d / wall_margin)
        .sum::<f32>();
        shaping -= 0.001 * wall_penalty;

        combat_reward + shaping
    }
}

pub fn setup_world(world: &mut World, config: &GameConfig, physics: &mut PhysicsState) {
    world.insert_resource(WorldBounds {
        width: config.arena.width,
        height: config.arena.height,
    });
    world.insert_resource(TickState::new(config.tick_rate));
    world.insert_resource(TelemetryBuffer::default());
    world.insert_resource(GameConfigResource(config.clone()));
    world.insert_resource(RoundState::default());

    let w = config.arena.width;
    let h = config.arena.height;
    let wall_t = 20.0;
    for (pos, half) in [
        (
            Vec2::new(w / 2.0, -wall_t / 2.0),
            Vec2::new(w / 2.0 + wall_t, wall_t / 2.0),
        ),
        (
            Vec2::new(w / 2.0, h + wall_t / 2.0),
            Vec2::new(w / 2.0 + wall_t, wall_t / 2.0),
        ),
        (
            Vec2::new(-wall_t / 2.0, h / 2.0),
            Vec2::new(wall_t / 2.0, h / 2.0 + wall_t),
        ),
        (
            Vec2::new(w + wall_t / 2.0, h / 2.0),
            Vec2::new(wall_t / 2.0, h / 2.0 + wall_t),
        ),
    ] {
        let (body, collider) = physics.add_static_body(pos, half);
        world.spawn((Position(pos), Obstacle, PhysicsHandle { body, collider }));
    }

    for obs in &config.obstacles {
        let center = Vec2::new(obs.x + obs.width / 2.0, obs.y + obs.height / 2.0);
        let half_extents = Vec2::new(obs.width / 2.0, obs.height / 2.0);
        let (body, collider) = physics.add_static_body(center, half_extents);
        world.spawn((
            Position(center),
            Obstacle,
            PhysicsHandle { body, collider },
        ));
    }

    let mut source_id: u32 = 0;
    for team_idx in 0..config.teams.count {
        for player_idx in 0..config.teams.players_per_team {
            let spawn_x = if team_idx == 0 {
                100.0
            } else {
                config.arena.width - 100.0
            };
            let spacing = config.arena.height / (config.teams.players_per_team as f32 + 1.0);
            let spawn_y = spacing * (player_idx as f32 + 1.0);
            let spawn_pos = Vec2::new(spawn_x, spawn_y);

            let (body, collider) = physics.add_dynamic_body(spawn_pos, 15.0);

            world.spawn((
                Position(spawn_pos),
                Velocity(Vec2::ZERO),
                Facing(if team_idx == 0 {
                    0.0
                } else {
                    std::f32::consts::PI
                }),
                Health {
                    current: 100.0,
                    max: 100.0,
                },
                Team(team_idx),
                Weapon {
                    damage: config.combat.default_weapon.damage,
                    fire_rate: config.combat.default_weapon.fire_rate,
                    range: config.combat.default_weapon.range,
                    cooldown_remaining: 0.0,
                },
                Agent { source_id },
                ScriptedAi(aggressive_ai()),
                PhysicsHandle { body, collider },
            ));

            source_id += 1;
        }
    }
}

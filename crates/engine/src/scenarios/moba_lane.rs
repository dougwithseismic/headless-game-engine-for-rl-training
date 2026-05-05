use bevy_ecs::prelude::*;
use glam::Vec2;

use crate::action_space::{ActionHead, ActionSpaceDef};
use crate::config::GameConfig;
use crate::ecs::components::*;
use crate::ecs::resources::*;
use crate::ecs::systems;
use crate::observation::{ObsFeature, ObsWriter, ObservationSpaceDef};
use crate::physics::PhysicsState;
use crate::scenario::Scenario;
use crate::scripted_ai::{aggressive_ai, creep_ai, ScriptedAi};
use crate::tick::EnginePhase;

#[derive(Component, Debug, Clone)]
pub struct Creep;

#[derive(Resource, Debug, Clone, Default)]
pub struct GoldTracker {
    pub team_gold: [u32; 2],
}

#[derive(Resource, Debug, Clone)]
pub struct CreepSpawner {
    pub interval_ticks: u64,
    pub creeps_per_wave: u8,
    pub last_spawn_tick: u64,
    pub creep_hp: f32,
    pub creep_damage: f32,
    pub creep_speed: f32,
}

pub struct MobaLaneScenario;

impl Scenario for MobaLaneScenario {
    fn name(&self) -> &str {
        "moba-lane"
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

    fn observation_space(&self, _config: &GameConfig) -> ObservationSpaceDef {
        ObservationSpaceDef {
            features: vec![
                ObsFeature {
                    name: "self_features".into(),
                    shape: vec![7],
                },
                ObsFeature {
                    name: "entities".into(),
                    shape: vec![16, 6],
                },
                ObsFeature {
                    name: "action_mask".into(),
                    shape: vec![2],
                },
            ],
        }
    }

    fn register_systems(&self, schedule: &mut Schedule) {
        schedule.add_systems(creep_spawner_system.in_set(EnginePhase::AiDecisions));
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
        let pos = world.get::<Position>(agent).map(|p| p.0).unwrap_or_default();
        let vel = world.get::<Velocity>(agent).map(|v| v.0).unwrap_or_default();
        let health = world.get::<Health>(agent);
        let facing = world.get::<Facing>(agent).map(|f| f.0).unwrap_or(0.0);
        let team = world.get::<Team>(agent).map(|t| t.0).unwrap_or(0);
        let hp = health.map(|h| h.current).unwrap_or(0.0);
        let max_hp = health.map(|h| h.max).unwrap_or(1.0);

        writer.write("self_features", &[
            pos.x, pos.y, vel.x, vel.y, hp / max_hp, facing, team as f32,
        ]);

        let registry = world.resource::<crate::observation::AgentRegistry>();
        let agent_list: Vec<Entity> = registry.agents.clone();
        let mut entity_data = Vec::new();
        for &e in &agent_list {
            if e == agent {
                continue;
            }
            let e_pos = world.get::<Position>(e).map(|p| p.0).unwrap_or_default();
            let e_vel = world.get::<Velocity>(e).map(|v| v.0).unwrap_or_default();
            let e_health = world.get::<Health>(e);
            let e_team = world.get::<Team>(e).map(|t| t.0).unwrap_or(0);
            let e_hp = e_health.map(|h| h.current / h.max).unwrap_or(0.0);
            entity_data.extend_from_slice(&[
                e_pos.x - pos.x,
                e_pos.y - pos.y,
                e_vel.x,
                e_vel.y,
                e_hp,
                e_team as f32,
            ]);
        }
        writer.write_padded("entities", &entity_data, 16 * 6);

        let weapon = world.get::<Weapon>(agent);
        let can_shoot = weapon.map(|w| w.cooldown_remaining <= 0.0).unwrap_or(false);
        let is_dead = world.get::<Dead>(agent).is_some();
        writer.write("action_mask", &[
            if is_dead { 0.0 } else { 1.0 },
            if can_shoot && !is_dead { 1.0 } else { 0.0 },
        ]);
    }

    fn setup(&self, world: &mut World, config: &GameConfig, physics: &mut PhysicsState) {
        let creep_interval = config
            .extra
            .get("creep_interval")
            .and_then(|v| v.as_f64())
            .unwrap_or(5.0);
        let creeps_per_wave = config
            .extra
            .get("creeps_per_wave")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as u8;
        let creep_hp = config
            .extra
            .get("creep_hp")
            .and_then(|v| v.as_f64())
            .unwrap_or(60.0) as f32;

        world.insert_resource(WorldBounds {
            width: config.arena.width,
            height: config.arena.height,
        });
        world.insert_resource(TickState::new(config.tick_rate));
        world.insert_resource(TelemetryBuffer::default());
        world.insert_resource(GameConfigResource(config.clone()));
        world.insert_resource(GoldTracker::default());
        world.insert_resource(CreepSpawner {
            interval_ticks: (creep_interval * config.tick_rate as f64) as u64,
            creeps_per_wave,
            last_spawn_tick: 0,
            creep_hp,
            creep_damage: config.combat.default_weapon.damage * 0.4,
            creep_speed: config.movement.max_speed * 0.4,
        });

        let w = config.arena.width;
        let h = config.arena.height;
        let wall_t = 20.0;
        for (pos, half) in [
            (Vec2::new(w / 2.0, -wall_t / 2.0), Vec2::new(w / 2.0 + wall_t, wall_t / 2.0)),
            (Vec2::new(w / 2.0, h + wall_t / 2.0), Vec2::new(w / 2.0 + wall_t, wall_t / 2.0)),
            (Vec2::new(-wall_t / 2.0, h / 2.0), Vec2::new(wall_t / 2.0, h / 2.0 + wall_t)),
            (Vec2::new(w + wall_t / 2.0, h / 2.0), Vec2::new(wall_t / 2.0, h / 2.0 + wall_t)),
        ] {
            let (body, collider) = physics.add_static_body(pos, half);
            world.spawn((Position(pos), Obstacle, PhysicsHandle { body, collider }));
        }

        for obs in &config.obstacles {
            let center = Vec2::new(obs.x + obs.width / 2.0, obs.y + obs.height / 2.0);
            let half = Vec2::new(obs.width / 2.0, obs.height / 2.0);
            let (body, collider) = physics.add_static_body(center, half);
            world.spawn((Position(center), Obstacle, PhysicsHandle { body, collider }));
        }

        for (source_id, team_idx) in (0_u32..).zip(0..config.teams.count.min(2)) {
            let spawn_x = if team_idx == 0 { 100.0 } else { config.arena.width - 100.0 };
            let spawn_y = config.arena.height / 2.0;
            let spawn_pos = Vec2::new(spawn_x, spawn_y);

            let (body, collider) = physics.add_dynamic_body(spawn_pos, 15.0);

            world.spawn((
                Position(spawn_pos),
                Velocity(Vec2::ZERO),
                Facing(if team_idx == 0 { 0.0 } else { std::f32::consts::PI }),
                Health { current: 200.0, max: 200.0 },
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
        }

        spawn_creep_wave_with_physics(world, physics, config, 0);
        spawn_creep_wave_with_physics(world, physics, config, 1);
    }
}

fn spawn_creep_wave_with_physics(
    world: &mut World,
    physics: &mut PhysicsState,
    config: &GameConfig,
    team: u8,
) {
    let spawner = world.resource::<CreepSpawner>().clone();
    let lane_y = config.arena.height / 2.0;
    let march_angle = if team == 0 { 0.0 } else { std::f32::consts::PI };
    let base_x = if team == 0 { 60.0 } else { config.arena.width - 60.0 };

    for i in 0..spawner.creeps_per_wave {
        let offset_y = (i as f32 - spawner.creeps_per_wave as f32 / 2.0) * 30.0;
        let offset_x = i as f32 * -20.0 * if team == 0 { 1.0 } else { -1.0 };
        let spawn_pos = Vec2::new(base_x + offset_x, lane_y + offset_y);

        let (body, collider) = physics.add_dynamic_body(spawn_pos, 10.0);

        world.spawn((
            Position(spawn_pos),
            Velocity(Vec2::ZERO),
            Facing(march_angle),
            Health { current: spawner.creep_hp, max: spawner.creep_hp },
            Team(team),
            Weapon {
                damage: spawner.creep_damage,
                fire_rate: 0.8,
                range: 150.0,
                cooldown_remaining: 0.0,
            },
            Agent { source_id: 100 + team as u32 * 50 + i as u32 },
            ScriptedAi(creep_ai(march_angle)),
            PhysicsHandle { body, collider },
            Creep,
        ));
    }
}

pub fn creep_spawner_system(world: &mut World) {
    let tick = world.resource::<TickState>().tick;
    let spawner = world.resource::<CreepSpawner>().clone();
    let config = world.resource::<GameConfigResource>().0.clone();

    if tick > 0 && tick - spawner.last_spawn_tick >= spawner.interval_ticks {
        let mut physics = world.remove_resource::<PhysicsState>().unwrap();
        spawn_creep_wave_with_physics(world, &mut physics, &config, 0);
        spawn_creep_wave_with_physics(world, &mut physics, &config, 1);
        world.insert_resource(physics);

        world.resource_mut::<CreepSpawner>().last_spawn_tick = tick;
    }
}

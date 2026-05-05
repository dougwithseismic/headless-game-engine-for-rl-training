use bevy_ecs::prelude::*;
use glam::Vec2;

use crate::action_space::{ActionHead, ActionSpaceDef};
use crate::config::GameConfig;
use crate::ecs::components::*;
use crate::ecs::resources::*;
use crate::ecs::systems;
use crate::observation::{ObsFeature, ObsWriter, ObservationSpaceDef};
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
                    shape: vec![7],
                },
                ObsFeature {
                    name: "entities".into(),
                    shape: vec![max_agents, 6],
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
        let max_agents = registry.max_agents;
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
        writer.write_padded("entities", &entity_data, max_agents * 6);

        let weapon = world.get::<Weapon>(agent);
        let can_shoot = weapon.map(|w| w.cooldown_remaining <= 0.0).unwrap_or(false);
        let is_dead = world.get::<Dead>(agent).is_some();
        writer.write("action_mask", &[
            if is_dead { 0.0 } else { 1.0 },
            if can_shoot && !is_dead { 1.0 } else { 0.0 },
        ]);
    }

    fn reward(&self, world: &World, agent: Entity) -> f32 {
        world
            .resource::<crate::observation::RewardBuffer>()
            .get(agent)
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

use bevy_ecs::prelude::*;
use glam::Vec2;

use rand::Rng;

use crate::action_space::ActionSpaceDef;
use crate::config::GameConfig;
use crate::ecs::components::*;
use crate::ecs::resources::*;
use crate::observation::{ObsWriter, ObservationSpaceDef};
use crate::physics::PhysicsState;
use crate::scripted_ai::{aggressive_ai, ScriptedAi};
use crate::telemetry::TelemetryEvent;

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

    let mut rng = rand::rng();
    let mut obstacle_rects = Vec::new();
    for obs in &config.obstacles {
        let (obs_w, obs_h) = if rng.random_bool(0.5) {
            (obs.height, obs.width)
        } else {
            (obs.width, obs.height)
        };
        let jitter_x = rng.random_range(-40.0f32..40.0);
        let jitter_y = rng.random_range(-40.0f32..40.0);
        let cx = (obs.x + obs_w / 2.0 + jitter_x).clamp(obs_w, w - obs_w);
        let cy = (obs.y + obs_h / 2.0 + jitter_y).clamp(obs_h, h - obs_h);
        let half_extents = Vec2::new(obs_w / 2.0, obs_h / 2.0);
        let (body, collider) = physics.add_static_body(Vec2::new(cx, cy), half_extents);
        world.spawn((
            Position(Vec2::new(cx, cy)),
            Obstacle,
            PhysicsHandle { body, collider },
        ));
        obstacle_rects.push(ObstacleRect {
            x: cx - obs_w / 2.0,
            y: cy - obs_h / 2.0,
            width: obs_w,
            height: obs_h,
        });
    }
    world.insert_resource(ObstacleLayout(obstacle_rects.clone()));

    let margin = 60.0;
    let spawn_points = vec![
        Vec2::new(margin + 40.0, h / 2.0),
        Vec2::new(w - margin - 40.0, h / 2.0),
        Vec2::new(w / 2.0, margin + 40.0),
        Vec2::new(w / 2.0, h - margin - 40.0),
        Vec2::new(margin + 80.0, margin + 80.0),
        Vec2::new(w - margin - 80.0, margin + 80.0),
        Vec2::new(margin + 80.0, h - margin - 80.0),
        Vec2::new(w - margin - 80.0, h - margin - 80.0),
    ];
    world.insert_resource(SpawnPointPool(spawn_points.clone()));

    world.resource_mut::<TelemetryBuffer>().push(TelemetryEvent::RoundStart {
        tick: 0,
        obstacles: obstacle_rects,
        spawn_points: spawn_points.clone(),
    });

    let mut source_id: u32 = 0;
    for team_idx in 0..config.teams.count {
        for _player_idx in 0..config.teams.players_per_team {
            let spawn_pos = spawn_points[rng.random_range(0..spawn_points.len())]
                + Vec2::new(rng.random_range(-15.0f32..15.0), rng.random_range(-15.0f32..15.0));

            let (body, collider) = physics.add_dynamic_body(spawn_pos, 15.0);

            let random_facing = rng.random_range(-std::f32::consts::PI..std::f32::consts::PI);
            world.spawn((
                Position(spawn_pos),
                Velocity(Vec2::ZERO),
                Facing(random_facing),
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

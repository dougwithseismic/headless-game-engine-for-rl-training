use bevy_ecs::prelude::*;
use glam::Vec2;

use crate::action_space::{ActionHead, ActionSpaceDef, RawActionBuffer};
use crate::config::GameConfig;
use crate::ecs::components::*;
use crate::ecs::resources::*;
use crate::observation::{ObsFeature, ObsWriter, ObservationSpaceDef, RewardBuffer};
use crate::physics::PhysicsState;
use crate::scenario::Scenario;
use crate::scripted_ai::ScriptedAi;
use crate::tick::EnginePhase;

#[derive(Component, Debug, Clone)]
pub struct Vehicle {
    pub heading: f32,
    pub speed: f32,
    pub max_speed: f32,
    pub acceleration: f32,
    pub turn_rate: f32,
    pub brake_force: f32,
}

#[derive(Component, Debug, Clone)]
pub struct CheckpointTracker {
    pub next_checkpoint: usize,
    pub lap: u32,
    pub laps_to_win: u32,
}

#[derive(Resource, Debug, Clone)]
pub struct Checkpoints {
    pub positions: Vec<Vec2>,
    pub radius: f32,
}

#[derive(Resource, Debug, Clone, Default)]
pub struct RaceState {
    pub finished: bool,
}

pub struct RacingScenario;

impl Scenario for RacingScenario {
    fn name(&self) -> &str {
        "racing"
    }

    fn action_space(&self, _config: &GameConfig) -> ActionSpaceDef {
        ActionSpaceDef::new(vec![
            ActionHead::Continuous {
                name: "steer".into(),
                size: 1,
                low: vec![-1.0],
                high: vec![1.0],
            },
            ActionHead::Continuous {
                name: "throttle".into(),
                size: 1,
                low: vec![0.0],
                high: vec![1.0],
            },
            ActionHead::Discrete {
                name: "brake".into(),
                n: 2,
            },
        ])
    }

    fn observation_space(&self, config: &GameConfig) -> ObservationSpaceDef {
        let max_cars = config.teams.players_per_team as usize;
        ObservationSpaceDef {
            features: vec![
                ObsFeature {
                    name: "self_features".into(),
                    shape: vec![8],
                },
                ObsFeature {
                    name: "track_waypoints".into(),
                    shape: vec![4, 2],
                },
                ObsFeature {
                    name: "other_cars".into(),
                    shape: vec![max_cars, 4],
                },
                ObsFeature {
                    name: "action_mask".into(),
                    shape: vec![2],
                },
            ],
        }
    }

    fn setup(&self, world: &mut World, config: &GameConfig, physics: &mut PhysicsState) {
        let max_speed = config.extra.get("max_speed")
            .and_then(|v| v.as_f64())
            .unwrap_or(300.0) as f32;
        let acceleration = config.extra.get("acceleration")
            .and_then(|v| v.as_f64())
            .unwrap_or(150.0) as f32;
        let turn_rate = config.extra.get("turn_rate")
            .and_then(|v| v.as_f64())
            .unwrap_or(3.0) as f32;
        let brake_force = config.extra.get("brake_force")
            .and_then(|v| v.as_f64())
            .unwrap_or(400.0) as f32;
        let laps_to_win = config.extra.get("laps_to_win")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as u32;
        let checkpoint_radius = config.extra.get("checkpoint_radius")
            .and_then(|v| v.as_f64())
            .unwrap_or(60.0) as f32;

        let w = config.arena.width;
        let h = config.arena.height;

        world.insert_resource(WorldBounds { width: w, height: h });
        world.insert_resource(TickState::new(config.tick_rate));
        world.insert_resource(TelemetryBuffer::default());
        world.insert_resource(GameConfigResource(config.clone()));
        world.insert_resource(RaceState::default());

        // Checkpoints around the oval (clockwise from bottom-center)
        let checkpoints = Checkpoints {
            positions: vec![
                Vec2::new(w / 2.0, h - 100.0),   // 0: bottom center (start/finish)
                Vec2::new(w - 100.0, h / 2.0),    // 1: right
                Vec2::new(w / 2.0, 100.0),         // 2: top center
                Vec2::new(100.0, h / 2.0),          // 3: left
            ],
            radius: checkpoint_radius,
        };
        let cp_positions = checkpoints.positions.clone();
        world.insert_resource(checkpoints);

        // Outer arena walls
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

        // Inner block (creates the oval track)
        let inner_w = w * 0.45;
        let inner_h = h * 0.45;
        let inner_center = Vec2::new(w / 2.0, h / 2.0);
        let (body, collider) = physics.add_static_body(
            inner_center,
            Vec2::new(inner_w / 2.0, inner_h / 2.0),
        );
        world.spawn((Position(inner_center), Obstacle, PhysicsHandle { body, collider }));

        // Spawn cars at start line (bottom center), staggered
        let num_cars = config.teams.players_per_team;
        let start_x = w / 2.0;
        let start_y = h - 80.0;

        for i in 0..num_cars {
            let offset_x = (i as f32 - num_cars as f32 / 2.0 + 0.5) * 40.0;
            let spawn_pos = Vec2::new(start_x + offset_x, start_y);

            let (body_handle, collider_handle) = physics.add_dynamic_body(spawn_pos, 12.0);

            world.spawn((
                Position(spawn_pos),
                Velocity(Vec2::ZERO),
                Facing(std::f32::consts::FRAC_PI_2), // facing up
                Health { current: 100.0, max: 100.0 },
                Team(0),
                Weapon { damage: 0.0, fire_rate: 1.0, range: 0.0, cooldown_remaining: 0.0 },
                Agent { source_id: i as u32 },
                Vehicle {
                    heading: std::f32::consts::FRAC_PI_2,
                    speed: 0.0,
                    max_speed,
                    acceleration,
                    turn_rate,
                    brake_force,
                },
                CheckpointTracker {
                    next_checkpoint: 0,
                    lap: 0,
                    laps_to_win,
                },
                ScriptedAi(super::super::scripted_ai::racing_ai(cp_positions.clone())),
                PhysicsHandle { body: body_handle, collider: collider_handle },
            ));
        }
    }

    fn register_systems(&self, schedule: &mut Schedule) {
        schedule.add_systems(vehicle_physics_system.in_set(EnginePhase::PrePhysics));
        schedule.add_systems(update_vehicle_state.in_set(EnginePhase::PostPhysics));
        schedule.add_systems(checkpoint_system.in_set(EnginePhase::GameLogic));
    }

    fn observe(&self, world: &World, agent: Entity, writer: &mut ObsWriter) {
        let pos = world.get::<Position>(agent).map(|p| p.0).unwrap_or_default();
        let vel = world.get::<Velocity>(agent).map(|v| v.0).unwrap_or_default();
        let vehicle = world.get::<Vehicle>(agent);
        let tracker = world.get::<CheckpointTracker>(agent);
        let heading = vehicle.map(|v| v.heading).unwrap_or(0.0);
        let speed = vehicle.map(|v| v.speed).unwrap_or(0.0);
        let next_cp = tracker.map(|t| t.next_checkpoint).unwrap_or(0);
        let lap = tracker.map(|t| t.lap).unwrap_or(0);

        writer.write("self_features", &[
            pos.x, pos.y, vel.x, vel.y, speed, heading, next_cp as f32, lap as f32,
        ]);

        // Next 4 checkpoint positions relative to self
        let checkpoints = world.resource::<Checkpoints>();
        let num_cp = checkpoints.positions.len();
        let mut wp_data = Vec::with_capacity(8);
        for i in 0..4 {
            let cp_idx = (next_cp + i) % num_cp;
            let cp_pos = checkpoints.positions[cp_idx];
            wp_data.push(cp_pos.x - pos.x);
            wp_data.push(cp_pos.y - pos.y);
        }
        writer.write("track_waypoints", &wp_data);

        // Other cars
        let registry = world.resource::<crate::observation::AgentRegistry>();
        let max_cars = registry.max_agents;
        let mut car_data = Vec::new();
        for &e in &registry.agents {
            if e == agent {
                continue;
            }
            let e_pos = world.get::<Position>(e).map(|p| p.0).unwrap_or_default();
            let e_vel = world.get::<Velocity>(e).map(|v| v.0).unwrap_or_default();
            let e_vehicle = world.get::<Vehicle>(e);
            car_data.extend_from_slice(&[
                e_pos.x - pos.x,
                e_pos.y - pos.y,
                e_vel.length(),
                e_vehicle.map(|v| v.heading).unwrap_or(0.0),
            ]);
        }
        writer.write_padded("other_cars", &car_data, max_cars * 4);

        writer.write("action_mask", &[1.0, 1.0]);
    }

    fn reward(&self, world: &World, agent: Entity) -> f32 {
        let mut reward = world.resource::<RewardBuffer>().get(agent);

        let vehicle = world.get::<Vehicle>(agent);
        let speed = vehicle.map(|v| v.speed).unwrap_or(0.0);
        if speed < 1.0 {
            reward -= 0.5;
        } else {
            reward += 0.01 * speed.min(100.0);
        }

        reward
    }

    fn is_done(&self, world: &World, agent: Entity) -> bool {
        if world.resource::<RaceState>().finished {
            return true;
        }
        world
            .get::<CheckpointTracker>(agent)
            .map(|t| t.lap >= t.laps_to_win)
            .unwrap_or(false)
    }
}

// --- Racing-specific systems ---

fn vehicle_physics_system(
    mut vehicles: Query<(Entity, &mut Vehicle, &PhysicsHandle), Without<Dead>>,
    raw_buffer: Res<RawActionBuffer>,
    tick: Res<TickState>,
    mut physics: ResMut<PhysicsState>,
) {
    let dt = tick.delta;

    for (entity, mut vehicle, ph) in &mut vehicles {
        let (steer, throttle, brake) = if let Some(raw) = raw_buffer.get(entity)
            && raw.len() >= 3
        {
            (
                raw[0].clamp(-1.0, 1.0),
                raw[1].clamp(0.0, 1.0),
                raw[2] > 0.5,
            )
        } else {
            (0.0, 0.0, false)
        };

        // Steering: turn_rate scales with speed (can't turn much when stopped)
        let speed_factor = (vehicle.speed / 50.0).clamp(0.1, 1.0);
        vehicle.heading += steer * vehicle.turn_rate * speed_factor * dt;

        let forward = Vec2::new(vehicle.heading.cos(), vehicle.heading.sin());

        if brake {
            let brake_decel = vehicle.brake_force * dt;
            vehicle.speed = (vehicle.speed - brake_decel).max(0.0);
        } else if throttle > 0.01 {
            vehicle.speed += vehicle.acceleration * throttle * dt;
            vehicle.speed = vehicle.speed.min(vehicle.max_speed);
        } else {
            // Coast: slow deceleration
            vehicle.speed = (vehicle.speed - 30.0 * dt).max(0.0);
        }

        let vel = forward * vehicle.speed;
        physics.set_body_linvel(ph.body, vel);
    }
}

fn update_vehicle_state(
    mut vehicles: Query<(&mut Vehicle, &mut Facing, &PhysicsHandle)>,
    physics: Res<PhysicsState>,
) {
    for (mut vehicle, mut facing, ph) in &mut vehicles {
        if let Some(vel) = physics.body_velocity(ph.body) {
            vehicle.speed = vel.length();
            if vehicle.speed > 1.0 {
                vehicle.heading = vel.y.atan2(vel.x);
            }
        }
        facing.0 = vehicle.heading;
    }
}

fn checkpoint_system(
    mut trackers: Query<(Entity, &Position, &mut CheckpointTracker)>,
    checkpoints: Res<Checkpoints>,
    mut race_state: ResMut<RaceState>,
    mut rewards: ResMut<RewardBuffer>,
) {
    for (entity, pos, mut tracker) in &mut trackers {
        if tracker.lap >= tracker.laps_to_win {
            continue;
        }

        let target = checkpoints.positions[tracker.next_checkpoint];
        let dist = pos.0.distance(target);

        if dist <= checkpoints.radius {
            tracker.next_checkpoint += 1;
            rewards.add(entity, 1.0);

            if tracker.next_checkpoint >= checkpoints.positions.len() {
                tracker.next_checkpoint = 0;
                tracker.lap += 1;
                rewards.add(entity, 10.0);

                if tracker.lap >= tracker.laps_to_win {
                    race_state.finished = true;
                }
            }
        }
    }
}

use bevy_ecs::prelude::*;
use glam::Vec2;

use crate::action_space::{ActionHead, ActionSpaceDef, RawActionBuffer};
use crate::config::GameConfig;
use crate::ecs::components::*;
use crate::ecs::resources::*;
use crate::ecs::systems;
use crate::navigation::NavGrid;
use crate::observation::{
    AgentRegistry, ObsFeature, ObsWriter, ObservationSpaceDef, RewardBuffer, ShotEventBuffer,
};
use crate::physics::PhysicsState;
use crate::scenario::{setup_world, Scenario};
use crate::scripted_ai::{tactical_aggressive_ai, ScriptedAi};
use crate::sensors;
use crate::telemetry::{RewardBreakdown, TelemetryEvent};
use crate::tick::EnginePhase;
use crate::weapons;

pub struct TacticalDeathmatchScenario;

#[derive(Resource, Debug, Clone)]
pub struct TacticalConfig {
    pub reward_mode: RewardMode,
    pub candidate_distance: f32,
    pub sensor_rays: usize,
    pub sensor_range: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RewardMode {
    Base,
    Cover,
}

impl Scenario for TacticalDeathmatchScenario {
    fn name(&self) -> &str {
        "tactical-deathmatch"
    }

    fn action_space(&self, _config: &GameConfig) -> ActionSpaceDef {
        ActionSpaceDef::new(vec![
            ActionHead::Discrete {
                name: "move_target".into(),
                n: 12,
            },
            ActionHead::Continuous {
                name: "aim_delta".into(),
                size: 1,
                low: vec![-1.0],
                high: vec![1.0],
            },
            ActionHead::Discrete {
                name: "shoot".into(),
                n: 2,
            },
            ActionHead::Discrete {
                name: "weapon_select".into(),
                n: 2,
            },
        ])
    }

    fn observation_space(&self, config: &GameConfig) -> ObservationSpaceDef {
        let max_agents =
            (config.teams.count as usize) * (config.teams.players_per_team as usize);
        let sensor_rays = config.extra_usize("sensor_rays", 64);

        ObservationSpaceDef {
            features: vec![
                ObsFeature {
                    name: "self_state".into(),
                    shape: vec![8],
                },
                ObsFeature {
                    name: "weapon_state".into(),
                    shape: vec![8],
                },
                ObsFeature {
                    name: "enemy_state".into(),
                    shape: vec![max_agents, 10],
                },
                ObsFeature {
                    name: "raycasts".into(),
                    shape: vec![sensor_rays, 2],
                },
                ObsFeature {
                    name: "candidates".into(),
                    shape: vec![12, 5],
                },
                ObsFeature {
                    name: "context".into(),
                    shape: vec![10],
                },
                ObsFeature {
                    name: "audio".into(),
                    shape: vec![2],
                },
                ObsFeature {
                    name: "action_mask".into(),
                    shape: vec![16],
                },
            ],
        }
    }

    fn setup(&self, world: &mut World, config: &GameConfig, physics: &mut PhysicsState) {
        let reward_mode = match config.extra_str("reward_mode", "base") {
            "cover" => RewardMode::Cover,
            _ => RewardMode::Base,
        };
        let tactical_config = TacticalConfig {
            reward_mode,
            candidate_distance: config.extra_f32("candidate_distance", 45.0),
            sensor_rays: config.extra_usize("sensor_rays", 64),
            sensor_range: config.extra_f32("sensor_range", 500.0),
        };

        setup_world(world, config, physics);

        world.insert_resource(tactical_config);
        world.insert_resource(CandidatePositionBuffer::default());
        world.insert_resource(ObstacleColliders::default());

        let obstacle_layout = world.resource::<ObstacleLayout>().0.clone();
        let obstacles: Vec<(Vec2, Vec2)> = obstacle_layout
            .iter()
            .map(|r| {
                (
                    Vec2::new(r.x + r.width / 2.0, r.y + r.height / 2.0),
                    Vec2::new(r.width / 2.0, r.height / 2.0),
                )
            })
            .collect();

        let nav_grid = NavGrid::from_obstacles(
            config.arena.width,
            config.arena.height,
            &obstacles,
            15.0,
            15.0,
        );
        world.insert_resource(nav_grid);

        // Collect obstacle collider handles for sensor ray classification
        {
            let mut obs_query = world.query_filtered::<&PhysicsHandle, With<Obstacle>>();
            let colliders: Vec<_> = obs_query.iter(world).map(|ph| ph.collider).collect();
            world.insert_resource(ObstacleColliders(colliders));
        }

        let mut agents_query = world.query_filtered::<Entity, With<Agent>>();
        let agent_entities: Vec<Entity> = agents_query.iter(world).collect();
        for entity in agent_entities {
            world.entity_mut(entity).insert((
                EnemyMemory::default(),
                PathState::default(),
                LosTracker::default(),
                ScriptedAi(tactical_aggressive_ai()),
                Inventory {
                    weapons: vec![
                        WeaponSlot {
                            weapon_type: WeaponType::Rifle,
                            damage: 34.0,
                            fire_rate: 0.3,
                            range: 400.0,
                            cooldown_remaining: 0.0,
                            ammo: 30,
                            max_ammo: 30,
                            reload_time: 2.0,
                            reload_remaining: 0.0,
                            is_reloading: false,
                        },
                        WeaponSlot {
                            weapon_type: WeaponType::Shotgun,
                            damage: 15.0,
                            fire_rate: 0.8,
                            range: 200.0,
                            cooldown_remaining: 0.0,
                            ammo: 8,
                            max_ammo: 8,
                            reload_time: 2.5,
                            reload_remaining: 0.0,
                            is_reloading: false,
                        },
                    ],
                    active: 0,
                },
            ));
        }
    }

    fn register_systems(&self, schedule: &mut Schedule) {
        schedule.add_systems(
            (crate::candidates::compute_candidates_system, tactical_movement_system)
                .chain()
                .in_set(EnginePhase::PrePhysics),
        );
        schedule.add_systems(
            (
                systems::facing_system,
                weapons::inventory_cooldown_system,
                weapons::weapon_switch_system,
                weapons::inventory_combat_system,
                update_enemy_memory_system,
                update_los_tracker_system,
            )
                .chain()
                .in_set(EnginePhase::GameLogic),
        );
        schedule.add_systems(
            (systems::death_system, systems::respawn_system)
                .chain()
                .in_set(EnginePhase::StateTransitions),
        );
        schedule.add_systems(
            tactical_telemetry_system.in_set(EnginePhase::Telemetry),
        );
    }

    fn observe(&self, world: &World, agent: Entity, writer: &mut ObsWriter) {
        let bounds = world.resource::<WorldBounds>();
        let config = world.resource::<GameConfigResource>();
        let physics = world.resource::<PhysicsState>();
        let tactical = world.resource::<TacticalConfig>();
        let max_speed = config.0.movement.max_speed;
        let arena_diag = bounds.diagonal();

        let pos = world.get::<Position>(agent).map(|p| p.0).unwrap_or_default();
        let vel = world.get::<Velocity>(agent).map(|v| v.0).unwrap_or_default();
        let health = world.get::<Health>(agent);
        let facing = world.get::<Facing>(agent).map(|f| f.0).unwrap_or(0.0);
        let hp = health.map(|h| h.current).unwrap_or(0.0);
        let max_hp = health.map(|h| h.max).unwrap_or(1.0);
        let weapon = world.get::<Weapon>(agent);
        let inv = world.get::<Inventory>(agent);

        // Use Inventory active weapon for cooldown if available, fall back to Weapon
        let cooldown_norm = inv
            .and_then(|i| i.active_weapon())
            .map(|w| w.cooldown_fraction())
            .or_else(|| {
                weapon.map(|w| {
                    if w.fire_rate > 0.0 {
                        w.cooldown_remaining / w.fire_rate
                    } else {
                        0.0
                    }
                })
            })
            .unwrap_or(0.0);

        // Self state [8]
        writer.write("self_state", &[
            pos.x / bounds.width,
            pos.y / bounds.height,
            vel.x / max_speed,
            vel.y / max_speed,
            hp / max_hp,
            facing.sin(),
            facing.cos(),
            cooldown_norm,
        ]);

        // Weapon state [8]: per weapon (x2): ammo_fraction, is_reloading, cooldown_fraction = 6
        //                    + current weapon one-hot (2) = 8
        if let Some(inventory) = inv {
            let mut weapon_obs = Vec::with_capacity(8);
            for slot in &inventory.weapons {
                weapon_obs.push(slot.ammo_fraction());
                weapon_obs.push(if slot.is_reloading { 1.0 } else { 0.0 });
                weapon_obs.push(slot.cooldown_fraction());
            }
            // Pad to 6 if fewer than 2 weapons
            while weapon_obs.len() < 6 {
                weapon_obs.push(0.0);
            }
            // One-hot current weapon
            for i in 0..2 {
                weapon_obs.push(if inventory.active == i { 1.0 } else { 0.0 });
            }
            writer.write("weapon_state", &weapon_obs);
        } else {
            writer.write("weapon_state", &[0.0; 8]);
        }

        // Enemy state [max_agents, 10]
        let registry = world.resource::<AgentRegistry>();
        let agent_list = &registry.agents;
        let max_agents = registry.max_agents;
        let team = world.get::<Team>(agent).map(|t| t.0).unwrap_or(0);
        let agent_collider = world.get::<PhysicsHandle>(agent).map(|ph| ph.collider);
        let memory = world.get::<EnemyMemory>(agent);
        let tick = world.resource::<TickState>().tick;

        let face_dir = Vec2::new(facing.cos(), facing.sin());
        let mut entity_data = Vec::new();

        for &e in agent_list {
            if e == agent {
                continue;
            }

            let e_pos = world.get::<Position>(e).map(|p| p.0).unwrap_or_default();
            let e_dead = world.get::<Dead>(e).is_some();
            let e_team = world.get::<Team>(e).map(|t| t.0).unwrap_or(0);
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

            if e_team != team {
                // God-mode teacher: always provide real enemy state.
                // LOS flag tells the agent whether it can currently see them.
                let e_vel = world.get::<Velocity>(e).map(|v| v.0).unwrap_or_default();
                let e_hp = world
                    .get::<Health>(e)
                    .map(|h| (h.current / h.max).max(0.0))
                    .unwrap_or(0.0);
                let los = if visible { 1.0 } else { 0.0 };
                let time_since = if visible {
                    0.0
                } else {
                    memory
                        .and_then(|m| m.entries.get(&e))
                        .map(|mem| ((tick - mem.last_seen_tick) as f32 / 300.0).min(1.0))
                        .unwrap_or(1.0)
                };
                let last_known_dx = if visible {
                    0.0
                } else {
                    memory
                        .and_then(|m| m.entries.get(&e))
                        .map(|mem| (mem.last_known_pos.x - pos.x) / arena_diag)
                        .unwrap_or(0.0)
                };
                let last_known_dy = if visible {
                    0.0
                } else {
                    memory
                        .and_then(|m| m.entries.get(&e))
                        .map(|mem| (mem.last_known_pos.y - pos.y) / arena_diag)
                        .unwrap_or(0.0)
                };
                let threat = if dist > 0.1 {
                    (1.0 / (dist / arena_diag)) * los * e_hp
                } else {
                    e_hp
                };

                entity_data.extend_from_slice(&[
                    delta.x / arena_diag,
                    delta.y / arena_diag,
                    (e_vel.x - vel.x) / max_speed,
                    (e_vel.y - vel.y) / max_speed,
                    e_hp,
                    los,
                    time_since,
                    last_known_dx,
                    last_known_dy,
                    threat.min(10.0) / 10.0,
                ]);
            }
        }
        writer.write_padded("enemy_state", &entity_data, max_agents * 10);

        // Raycasts [sensor_rays, 2]
        let obstacle_colliders = world.resource::<ObstacleColliders>();
        let mut agent_collider_data = Vec::new();
        for &e in agent_list {
            if e == agent {
                continue;
            }
            if world.get::<Dead>(e).is_some() {
                continue;
            }
            if let Some(ph) = world.get::<PhysicsHandle>(e) {
                let t = world.get::<Team>(e).map(|t| t.0).unwrap_or(0);
                agent_collider_data.push((ph.collider, t));
            }
        }
        let collider_types =
            sensors::build_collider_type_map(&obstacle_colliders.0, &agent_collider_data, team);

        let rays = sensors::cast_sensor_rays(
            physics,
            pos,
            agent_collider.unwrap_or(rapier2d::prelude::ColliderHandle::from_raw_parts(0, 0)),
            tactical.sensor_rays,
            tactical.sensor_range,
            &collider_types,
        );
        let mut ray_data = Vec::with_capacity(rays.len() * 2);
        for r in &rays {
            ray_data.push(r.distance);
            ray_data.push(r.hit_type);
        }
        writer.write("raycasts", &ray_data);

        // Candidate features [12, 5]
        let candidate_buffer = world.resource::<CandidatePositionBuffer>();
        if let Some(cand_set) = candidate_buffer.get(agent) {
            let features = cand_set.as_obs_features();
            writer.write("candidates", &features);
        } else {
            writer.write("candidates", &[0.0; 60]);
        }

        // Context [10]
        let round = world.resource::<RoundState>();
        let round_time_limit = config.0.spawning.round_time_limit;
        let round_timer = if round_time_limit > 0.0 {
            (round.round_clock / round_time_limit).min(1.0)
        } else {
            0.0
        };
        // score_diff placeholder (would need kill tracking per team)
        let score_diff = 0.0f32;
        // strategic_goal one-hot — 8 zeros (placeholder for future strategic model)
        writer.write("context", &[
            round_timer,
            score_diff,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]);

        // Audio [2]
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

        // Action mask [16]: 12 move + 1 shoot + 1 alive + 2 weapon_select
        let is_dead = world.get::<Dead>(agent).is_some();
        let can_shoot = inv
            .and_then(|i| i.active_weapon())
            .map(|w| w.cooldown_remaining <= 0.0 && w.ammo > 0 && !w.is_reloading)
            .or_else(|| weapon.map(|w| w.cooldown_remaining <= 0.0))
            .unwrap_or(false);
        let alive_f = if is_dead { 0.0 } else { 1.0 };
        let shoot_f = if can_shoot && !is_dead { 1.0 } else { 0.0 };

        let mut mask = Vec::with_capacity(16);
        // 12 movement candidate masks
        if let Some(cand_set) = candidate_buffer.get(agent) {
            let wmask = cand_set.walkable_mask();
            for &m in &wmask {
                mask.push(if is_dead { 0.0 } else { m });
            }
        } else {
            for _ in 0..12 {
                mask.push(alive_f);
            }
        }
        mask.push(shoot_f);
        mask.push(alive_f);
        // 2 weapon_select masks (always valid when alive)
        mask.push(alive_f);
        mask.push(alive_f);
        writer.write("action_mask", &mask);
    }

    fn reward(&self, world: &World, agent: Entity) -> f32 {
        let combat_reward = world.resource::<RewardBuffer>().get(agent);

        if world.get::<Dead>(agent).is_some() {
            return combat_reward;
        }

        let pos = world.get::<Position>(agent).map(|p| p.0).unwrap_or_default();
        let team = world.get::<Team>(agent).map(|t| t.0).unwrap_or(0);
        let facing = world.get::<Facing>(agent).map(|f| f.0).unwrap_or(0.0);
        let tactical_config = world.resource::<TacticalConfig>();

        let mut nearest_dist = f32::MAX;
        let mut nearest_dir = Vec2::ZERO;
        for &e in &world.resource::<AgentRegistry>().agents {
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
        let physics = world.resource::<PhysicsState>();
        let agent_collider = world.get::<PhysicsHandle>(agent).map(|ph| ph.collider);

        if nearest_dist < f32::MAX {
            let nearest_enemy_pos = pos + nearest_dir * nearest_dist;
            let has_los = physics.has_line_of_sight(pos, nearest_enemy_pos, agent_collider);

            let engage_range = 500.0;
            let dist_frac = (nearest_dist / engage_range).min(1.0);
            shaping += 0.005 * (1.0 - dist_frac);

            if has_los {
                let face_dir = Vec2::new(facing.cos(), facing.sin());
                let aim_dot = face_dir.dot(nearest_dir);
                shaping += 0.003 * aim_dot.max(0.0);
            }

            if tactical_config.reward_mode == RewardMode::Cover && !has_los {
                let vel = world.get::<Velocity>(agent).map(|v| v.0).unwrap_or_default();
                let speed = vel.length();
                // Hiding without LOS: mild penalty even when moving, harsher when stationary
                if speed < 10.0 {
                    shaping -= 0.008;
                } else {
                    shaping -= 0.002;
                }
            }
        }

        shaping -= 0.0005;

        combat_reward + shaping
    }
}

// --- ECS Systems ---

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn tactical_movement_system(
    mut query: Query<(Entity, &mut PathState, &PhysicsHandle), Without<Dead>>,
    raw_buffer: Res<crate::action_space::RawActionBuffer>,
    action_space: Res<ActionSpaceDef>,
    candidate_buffer: Res<CandidatePositionBuffer>,
    nav_grid: Res<NavGrid>,
    config: Res<GameConfigResource>,
    mut physics: ResMut<PhysicsState>,
) {
    let max_speed = config.0.movement.max_speed;

    for (entity, mut path_state, ph) in &mut query {
        let Some(raw) = raw_buffer.get(entity) else {
            continue;
        };
        if raw.len() < action_space.total_size {
            continue;
        }

        let move_slice = action_space.extract_head(raw, 0);
        let target_idx = (move_slice[0].round() as usize).min(11);

        // If target changed, recompute path
        if path_state.target_candidate != Some(target_idx) || path_state.is_complete() {
            path_state.target_candidate = Some(target_idx);

            if target_idx == 8 {
                // Stay
                path_state.clear();
                path_state.target_candidate = Some(8);
                physics.set_body_linvel(ph.body, Vec2::ZERO);
                continue;
            }

            if let Some(candidates) = candidate_buffer.get(entity) {
                let target_pos = candidates.positions[target_idx].world_pos;
                let current_pos = physics.body_position(ph.body).unwrap_or_default();

                if let Some(waypoints) = nav_grid.astar(current_pos, target_pos) {
                    path_state.waypoints = waypoints;
                    path_state.current_index = 0;
                } else {
                    // No path found — move directly
                    let dir = (target_pos - current_pos).normalize_or_zero();
                    physics.set_body_linvel(ph.body, dir * max_speed);
                    path_state.clear();
                    path_state.target_candidate = Some(target_idx);
                    continue;
                }
            } else {
                physics.set_body_linvel(ph.body, Vec2::ZERO);
                continue;
            }
        }

        // Follow current path
        if let Some(waypoint) = path_state.current_waypoint() {
            let current_pos = physics.body_position(ph.body).unwrap_or_default();
            let to_wp = waypoint - current_pos;
            let dist = to_wp.length();

            if dist < 20.0 {
                path_state.advance();
                if let Some(next_wp) = path_state.current_waypoint() {
                    let dir = (next_wp - current_pos).normalize_or_zero();
                    physics.set_body_linvel(ph.body, dir * max_speed);
                } else {
                    physics.set_body_linvel(ph.body, Vec2::ZERO);
                }
            } else {
                let dir = to_wp / dist;
                physics.set_body_linvel(ph.body, dir * max_speed);
            }
        } else {
            physics.set_body_linvel(ph.body, Vec2::ZERO);
        }
    }
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn tactical_combat_system(
    mut commands: Commands,
    mut shooters: Query<
        (Entity, &Position, &Facing, &mut Weapon, &Team, &PhysicsHandle),
        Without<Dead>,
    >,
    mut targets: Query<(Entity, &Position, &mut Health, &Team, &PhysicsHandle), Without<Dead>>,
    raw_buffer: Res<crate::action_space::RawActionBuffer>,
    action_space: Res<ActionSpaceDef>,
    physics: Res<PhysicsState>,
    tick: Res<TickState>,
    mut telemetry: ResMut<TelemetryBuffer>,
    mut rewards: ResMut<RewardBuffer>,
    mut shot_events: ResMut<ShotEventBuffer>,
    tactical_config: Res<TacticalConfig>,
    candidate_buffer: Res<CandidatePositionBuffer>,
) {
    let mut hits: Vec<(Entity, f32, Entity, bool)> = Vec::new();

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

        telemetry.push(crate::telemetry::TelemetryEvent::ShotFired {
            tick: tick.tick,
            shooter: shooter_entity.to_bits(),
            origin: shooter_pos.0,
            direction: dir,
            hit_target: best_hit.map(|(e, _)| e.to_bits()),
        });
        shot_events.push(shooter_entity, shooter_pos.0);

        if let Some((hit_entity, _)) = best_hit {
            // Check if shooter is in cover
            let in_cover = candidate_buffer
                .get(shooter_entity)
                .and_then(|cs| {
                    cs.positions.get(8).map(|stay| !stay.has_los_to_enemy)
                })
                .unwrap_or(false);
            hits.push((hit_entity, damage, shooter_entity, in_cover));
        }
    }

    for &(hit_entity, damage, shooter_entity, shooter_in_cover) in &hits {
        if let Ok((_entity, _pos, mut health, _team, _ph)) = targets.get_mut(hit_entity) {
            let max_hp = health.max;
            health.current -= damage;
            telemetry.push(crate::telemetry::TelemetryEvent::Damage {
                tick: tick.tick,
                source: shooter_entity.to_bits(),
                target: hit_entity.to_bits(),
                amount: damage,
            });
            let base_dmg_reward = 0.5 * damage / max_hp;
            let cover_bonus = if shooter_in_cover
                && tactical_config.reward_mode == RewardMode::Cover
            {
                0.5 * damage / max_hp
            } else {
                0.0
            };
            rewards.add(shooter_entity, base_dmg_reward + cover_bonus);
            rewards.add(hit_entity, -0.3 * damage / max_hp);
        }
        commands
            .entity(hit_entity)
            .insert(LastDamageSource(shooter_entity));
    }
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn tactical_telemetry_system(
    agents: Query<
        (Entity, &Position, &Velocity, &Team, &Facing, &PathState, &PhysicsHandle, &LosTracker),
        (With<Agent>, Without<Dead>),
    >,
    all_agents: Query<(&PhysicsHandle, &Team), (With<Agent>, Without<Dead>)>,
    enemy_positions: Query<(Entity, &Position, &Team), (With<Agent>, Without<Dead>)>,
    obstacles_res: Res<ObstacleColliders>,
    tick: Res<TickState>,
    candidate_buffer: Res<CandidatePositionBuffer>,
    raw_buffer: Res<RawActionBuffer>,
    action_space: Res<ActionSpaceDef>,
    tactical_config: Res<TacticalConfig>,
    physics: Res<PhysicsState>,
    bounds: Res<WorldBounds>,
    reward_buffer: Res<RewardBuffer>,
    mut telemetry: ResMut<TelemetryBuffer>,
) {
    if !tick.tick.is_multiple_of(2) {
        return;
    }

    for (entity, pos, vel, team, facing, path_state, ph, los_tracker) in &agents {
        let mut move_target: u8 = 0;
        let mut shooting = false;

        if let Some(raw) = raw_buffer.get(entity)
            && raw.len() >= action_space.total_size
        {
            let move_slice = action_space.extract_head(raw, 0);
            move_target = (move_slice[0].round() as u8).min(11);

            if action_space.heads.len() > 2 {
                let shoot_slice = action_space.extract_head(raw, 2);
                if !shoot_slice.is_empty() && shoot_slice[0] > 0.5 {
                    shooting = true;
                }
            }
        }

        let mut candidates: Vec<[f32; 2]> = Vec::with_capacity(12);
        let mut candidate_los: Vec<bool> = Vec::with_capacity(12);

        if let Some(cand_set) = candidate_buffer.get(entity) {
            for c in &cand_set.positions {
                candidates.push([c.world_pos.x, c.world_pos.y]);
                candidate_los.push(c.has_los_to_enemy);
            }
        }

        let path: Vec<[f32; 2]> = path_state
            .waypoints
            .iter()
            .map(|wp| [wp.x, wp.y])
            .collect();

        // Cast sensor rays for vision polygon
        let agent_collider_data: Vec<_> = all_agents
            .iter()
            .filter(|(other_ph, _)| other_ph.collider != ph.collider)
            .map(|(other_ph, t)| (other_ph.collider, t.0))
            .collect();
        let collider_types =
            sensors::build_collider_type_map(&obstacles_res.0, &agent_collider_data, team.0);
        let rays = sensors::cast_sensor_rays(
            &physics,
            pos.0,
            ph.collider,
            tactical_config.sensor_rays,
            tactical_config.sensor_range,
            &collider_types,
        );
        let ray_distances: Vec<f32> = rays.iter().map(|r| r.distance).collect();

        // Compute reward breakdown
        let arena_diag = bounds.diagonal();
        let mut proximity = 0.0f32;
        let mut aim = 0.0f32;
        let mut idle_penalty = 0.0f32;
        let mut los_gain = 0.0f32;
        let mut nearest_dist = f32::MAX;
        let mut nearest_dir = Vec2::ZERO;

        for (e, e_pos, e_team) in &enemy_positions {
            if e == entity || e_team.0 == team.0 {
                continue;
            }
            let d = pos.0.distance(e_pos.0);
            if d < nearest_dist {
                nearest_dist = d;
                nearest_dir = (e_pos.0 - pos.0).normalize_or_zero();
            }
        }

        if nearest_dist < f32::MAX {
            proximity = 0.003 * (1.0 - nearest_dist / arena_diag);
            let face_dir = Vec2::new(facing.0.cos(), facing.0.sin());
            aim = 0.002 * face_dir.dot(nearest_dir).max(0.0);
        }

        if tactical_config.reward_mode == RewardMode::Cover {
            if vel.0.length() < 10.0 && nearest_dist < f32::MAX {
                let nearest_enemy_pos = pos.0 + nearest_dir * nearest_dist;
                if !physics.has_line_of_sight(pos.0, nearest_enemy_pos, Some(ph.collider)) {
                    idle_penalty = -0.02;
                }
            }
            if los_tracker.had_los {
                los_gain = 0.1;
            }
        }

        let combat = reward_buffer.get(entity);
        let breakdown = RewardBreakdown {
            proximity,
            aim,
            time_penalty: -0.0005,
            cover_bonus: 0.0, // tracked in combat system, already in `combat`
            idle_penalty,
            los_gain,
            combat,
        };

        telemetry.push(TelemetryEvent::TacticalState {
            tick: tick.tick,
            entity: entity.to_bits(),
            move_target,
            candidates,
            candidate_los,
            path,
            aim_angle: facing.0,
            shooting,
            ray_distances,
            rewards: Some(breakdown),
        });
    }
}

pub fn update_enemy_memory_system(
    mut agents: Query<(Entity, &Position, &Team, &mut EnemyMemory, &PhysicsHandle), Without<Dead>>,
    targets: Query<(Entity, &Position, &Team, &PhysicsHandle), Without<Dead>>,
    physics: Res<PhysicsState>,
    tick: Res<TickState>,
) {
    let all_targets: Vec<_> = targets
        .iter()
        .map(|(e, p, t, ph)| (e, p.0, t.0, ph.collider))
        .collect();

    for (entity, pos, team, mut memory, ph) in &mut agents {
        for &(target_e, target_pos, target_team, target_collider) in &all_targets {
            if target_e == entity || target_team == team.0 {
                continue;
            }
            let delta = target_pos - pos.0;
            let dist = delta.length();
            if dist < 0.1 {
                continue;
            }
            let dir = delta / dist;
            let visible = match physics.cast_ray(pos.0, dir, dist, Some(ph.collider)) {
                Some((hit_col, _)) => hit_col == target_collider,
                None => true,
            };
            if visible {
                memory.entries.insert(target_e, MemoryEntry {
                    last_seen_tick: tick.tick,
                    last_known_pos: target_pos,
                });
            }
        }

        // Prune old entries
        memory
            .entries
            .retain(|_, entry| tick.tick - entry.last_seen_tick < 600);
    }
}

pub fn update_los_tracker_system(
    mut agents: Query<(Entity, &Position, &Team, &mut LosTracker, &PhysicsHandle), Without<Dead>>,
    targets: Query<(Entity, &Position, &Team), Without<Dead>>,
    physics: Res<PhysicsState>,
    mut rewards: ResMut<RewardBuffer>,
    tactical_config: Res<TacticalConfig>,
) {
    if tactical_config.reward_mode != RewardMode::Cover {
        return;
    }
    for (entity, pos, team, mut tracker, ph) in &mut agents {
        let mut has_los_now = false;
        for (target_e, target_pos, target_team) in &targets {
            if target_e == entity || target_team.0 == team.0 {
                continue;
            }
            if physics.has_line_of_sight(pos.0, target_pos.0, Some(ph.collider)) {
                has_los_now = true;
                break;
            }
        }
        if has_los_now && !tracker.had_los {
            rewards.add(entity, 0.1);
        }
        tracker.had_los = has_los_now;
    }
}


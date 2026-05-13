use bevy_ecs::prelude::*;
use glam::{Vec2, Vec3};
use rand::Rng;

use crate::action_space::{ActionHead, ActionSpaceDef, RawActionBuffer};
use crate::config::GameConfig;
use crate::ecs::components::*;
use crate::ecs::resources::*;
use crate::navigation::NavGrid;
use crate::observation::{
    AgentRegistry, ObsFeature, ObsWriter, ObservationSpaceDef, RewardBreakdownBuffer, RewardBuffer, ShotEventBuffer,
};
use crate::physics::PhysicsState;
use crate::physics3d::Physics3DState;
use crate::scenario::Scenario;
use crate::telemetry::TelemetryEvent;
use crate::tick::EnginePhase;

// ---------------------------------------------------------------------------
// 3D components (previously in arena3d/drone modules)
// ---------------------------------------------------------------------------

#[derive(Component, Debug, Clone)]
pub struct Position3D(pub Vec3);

#[derive(Component, Debug, Clone)]
pub struct Velocity3D(pub Vec3);

#[derive(Component, Debug, Clone)]
pub struct Facing3D {
    pub yaw: f32,
    pub pitch: f32,
}

impl Default for Facing3D {
    fn default() -> Self {
        Self { yaw: 0.0, pitch: 0.0 }
    }
}

impl Facing3D {
    pub fn direction(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
    }

    pub fn forward_xz(&self) -> Vec3 {
        Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin())
    }

    pub fn right_xz(&self) -> Vec3 {
        Vec3::new(self.yaw.sin(), 0.0, -self.yaw.cos())
    }
}

#[derive(Component, Debug, Clone)]
pub struct PhysicsHandle3D {
    pub body: rapier3d::prelude::RigidBodyHandle,
    pub collider: rapier3d::prelude::ColliderHandle,
}

#[derive(Resource, Debug, Clone, Default)]
pub struct SpawnPoints3D(pub Vec<Vec3>);

#[derive(Resource, Debug, Clone, Default)]
pub struct ObstacleColliders3D(pub Vec<rapier3d::prelude::ColliderHandle>);

// ---------------------------------------------------------------------------
// CS-Lite specific components
// ---------------------------------------------------------------------------

#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsSide {
    Terrorist,
    CounterTerrorist,
}

#[derive(Component, Debug, Clone)]
pub struct Armor {
    pub has_armor: bool,
}

#[derive(Component, Debug, Clone)]
pub struct BombCarrier;

#[derive(Component, Debug, Clone)]
pub struct PlantDefuseProgress {
    pub is_planting: bool,
    pub is_defusing: bool,
    pub progress: f32,
}

// ---------------------------------------------------------------------------
// Goal conditioning (Phase 3)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveType {
    PlantBomb,
    DefuseBomb,
    HoldPosition,
    Eliminate,
    Rotate,
}

impl Default for ObjectiveType {
    fn default() -> Self { Self::Eliminate }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Posture {
    Aggressive,
    Default,
    Passive,
}

impl Default for Posture {
    fn default() -> Self { Self::Default }
}

#[derive(Component, Debug, Clone)]
pub struct AgentGoal {
    pub objective: ObjectiveType,
    pub target_position: Vec3,
    pub posture: Posture,
}

impl Default for AgentGoal {
    fn default() -> Self {
        Self {
            objective: ObjectiveType::Eliminate,
            target_position: Vec3::ZERO,
            posture: Posture::Default,
        }
    }
}

// ---------------------------------------------------------------------------
// Round phase state machine
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundPhase {
    BuyFreeze,
    Active,
    RoundEnd,
}

#[derive(Resource, Debug, Clone)]
pub struct CsRoundState {
    pub phase: RoundPhase,
    pub phase_timer: f32,
    pub round_number: u32,
    pub t_score: u32,
    pub ct_score: u32,
    pub max_rounds: u32,
    pub round_time_limit: f32,
    pub buy_time: f32,
    pub end_time: f32,
    pub t_alive: u32,
    pub ct_alive: u32,
    pub round_winner: Option<CsSide>,
}

impl CsRoundState {
    pub fn new(config: &CsLiteConfig) -> Self {
        Self {
            phase: RoundPhase::BuyFreeze,
            phase_timer: config.buy_time,
            round_number: 1,
            t_score: 0,
            ct_score: 0,
            max_rounds: config.max_rounds,
            round_time_limit: config.round_time_limit,
            buy_time: config.buy_time,
            end_time: config.end_time,
            t_alive: config.players_per_team,
            ct_alive: config.players_per_team,
            round_winner: None,
        }
    }

    pub fn match_over(&self) -> bool {
        let rounds_to_win = self.max_rounds / 2 + 1;
        self.t_score >= rounds_to_win || self.ct_score >= rounds_to_win
    }
}

// ---------------------------------------------------------------------------
// Bomb state (Phase 2 — stubbed for now)
// ---------------------------------------------------------------------------

#[derive(Resource, Debug, Clone, Default)]
pub struct BombState {
    pub carrier: Option<Entity>,
    pub planted: bool,
    pub plant_site: Option<u8>,
    pub plant_position: Option<Vec3>,
    pub plant_tick: Option<u64>,
    pub detonated: bool,
    pub defused: bool,
    pub dropped_position: Option<Vec3>,
}

// ---------------------------------------------------------------------------
// Bomb site zones
// ---------------------------------------------------------------------------

#[derive(Resource, Debug, Clone)]
pub struct BombSites {
    pub site_a_center: Vec3,
    pub site_a_radius: f32,
    pub site_b_center: Vec3,
    pub site_b_radius: f32,
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Resource, Debug, Clone)]
pub struct CsLiteConfig {
    pub arena_width: f32,
    pub arena_height: f32,
    pub arena_depth: f32,
    pub gravity: f32,
    pub max_speed: f32,
    pub agent_half_height: f32,
    pub agent_radius: f32,
    pub eye_height: f32,
    pub yaw_bins: usize,
    pub pitch_bins: usize,
    pub yaw_rate: f32,
    pub pitch_rate: f32,
    pub pitch_limit: f32,
    pub ray_h_count: usize,
    pub ray_v_count: usize,
    pub ray_h_fov: f32,
    pub ray_v_fov: f32,
    pub ray_max_range: f32,
    pub max_rounds: u32,
    pub round_time_limit: f32,
    pub buy_time: f32,
    pub end_time: f32,
    pub hitbox_radius: f32,
    pub players_per_team: u32,
    pub reward_kill: f32,
    pub reward_death: f32,
    pub reward_damage_dealt: f32,
    pub reward_damage_taken: f32,
    pub reward_round_win: f32,
    pub reward_round_loss: f32,
    pub reward_near_miss: f32,
    pub reward_friendly_fire: f32,
    pub reward_bomb_pickup: f32,
    pub reward_bomb_plant: f32,
    pub reward_bomb_defuse: f32,
    pub reward_moving_shot: f32,
}

impl CsLiteConfig {
    pub fn from_game_config(config: &GameConfig) -> Self {
        Self {
            arena_width: config.extra_f32("arena_width", 80.0),
            arena_height: config.extra_f32("arena_height_3d", 10.0),
            arena_depth: config.extra_f32("arena_depth", 60.0),
            gravity: config.extra_f32("gravity", 9.81),
            max_speed: config.movement.max_speed,
            agent_half_height: config.extra_f32("agent_half_height", 0.9),
            agent_radius: config.extra_f32("agent_radius", 0.4),
            eye_height: config.extra_f32("eye_height", 1.6),
            yaw_bins: config.extra_usize("yaw_bins", 11),
            pitch_bins: config.extra_usize("pitch_bins", 11),
            yaw_rate: config.extra_f32("yaw_rate", 0.15),
            pitch_rate: config.extra_f32("pitch_rate", 0.10),
            pitch_limit: config.extra_f32("pitch_limit", 1.047),
            ray_h_count: config.extra_usize("ray_h_count", 9),
            ray_v_count: config.extra_usize("ray_v_count", 5),
            ray_h_fov: config.extra_f32("ray_h_fov", 2.094),
            ray_v_fov: config.extra_f32("ray_v_fov", 1.396),
            ray_max_range: config.extra_f32("ray_max_range", 60.0),
            max_rounds: config.extra_usize("max_rounds", 24) as u32,
            round_time_limit: config.extra_f32("round_time_limit", 115.0),
            buy_time: config.extra_f32("buy_time", 3.0),
            end_time: config.extra_f32("end_time", 3.0),
            hitbox_radius: config.extra_f32("hitbox_radius", 0.5),
            players_per_team: config.teams.players_per_team as u32,
            reward_kill: config.extra_f32("reward_kill", 3.0),
            reward_death: config.extra_f32("reward_death", 0.0),
            reward_damage_dealt: config.extra_f32("reward_damage_dealt", 1.0),
            reward_damage_taken: config.extra_f32("reward_damage_taken", 0.0),
            reward_round_win: config.extra_f32("reward_round_win", 5.0),
            reward_round_loss: config.extra_f32("reward_round_loss", 0.0),
            reward_near_miss: config.extra_f32("reward_near_miss", 0.03),
            reward_friendly_fire: config.extra_f32("reward_friendly_fire", -2.0),
            reward_bomb_pickup: config.extra_f32("reward_bomb_pickup", 0.1),
            reward_bomb_plant: config.extra_f32("reward_bomb_plant", 0.5),
            reward_bomb_defuse: config.extra_f32("reward_bomb_defuse", 1.0),
            reward_moving_shot: config.extra_f32("reward_moving_shot", 0.0),
        }
    }

    pub fn extra_candidate_distance(&self) -> f32 {
        15.0
    }

    pub fn arena_diagonal(&self) -> f32 {
        (self.arena_width * self.arena_width
            + self.arena_height * self.arena_height
            + self.arena_depth * self.arena_depth)
            .sqrt()
    }
}

// ---------------------------------------------------------------------------
// Spawn point sets per side
// ---------------------------------------------------------------------------

#[derive(Resource, Debug, Clone)]
pub struct CsSpawnPoints {
    pub t_spawns: Vec<Vec3>,
    pub ct_spawns: Vec<Vec3>,
}

// ---------------------------------------------------------------------------
// Telemetry event for CS-Lite
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Scenario
// ---------------------------------------------------------------------------

pub struct CsLiteScenario {
    pub dummy_ai: bool,
}

impl Default for CsLiteScenario {
    fn default() -> Self {
        Self { dummy_ai: false }
    }
}

impl Scenario for CsLiteScenario {
    fn name(&self) -> &str {
        "cs_lite"
    }

    fn action_space(&self, _config: &GameConfig) -> ActionSpaceDef {
        ActionSpaceDef::new(vec![
            ActionHead::Discrete { name: "move_target".into(), n: 12 },
            ActionHead::Discrete { name: "shoot".into(), n: 2 },
            ActionHead::Discrete { name: "reload".into(), n: 2 },
            ActionHead::Discrete { name: "use_action".into(), n: 3 },
        ])
    }

    fn observation_space(&self, config: &GameConfig) -> ObservationSpaceDef {
        let ppt = config.teams.players_per_team as usize;
        let ray_h = config.extra_usize("ray_h_count", 9);
        let ray_v = config.extra_usize("ray_v_count", 5);
        let total_rays = ray_h * ray_v;

        ObservationSpaceDef {
            features: vec![
                ObsFeature { name: "self_state".into(), shape: vec![12] },
                ObsFeature { name: "weapon_state".into(), shape: vec![20] },
                ObsFeature { name: "teammate_state".into(), shape: vec![ppt - 1, 5] },
                ObsFeature { name: "enemy_state".into(), shape: vec![ppt, 10] },
                ObsFeature { name: "round_info".into(), shape: vec![9] },
                ObsFeature { name: "bomb_state".into(), shape: vec![8] },
                // 12 candidates x 5 features (path_dist, has_los, no_los, dist_to_enemy, enemies_with_los)
                ObsFeature { name: "candidates".into(), shape: vec![12, 5] },
                ObsFeature { name: "raycasts_3d".into(), shape: vec![total_rays, 2] },
                ObsFeature { name: "audio_3d".into(), shape: vec![6] },
                ObsFeature { name: "aim_state".into(), shape: vec![5] },
                ObsFeature { name: "goal_state".into(), shape: vec![11] },
                ObsFeature { name: "action_mask".into(), shape: vec![19] },
            ],
        }
    }

    fn setup(&self, world: &mut World, config: &GameConfig, physics: &mut PhysicsState) {
        let cs_config = CsLiteConfig::from_game_config(config);
        let ppt = cs_config.players_per_team;

        // Standard 2D resources for engine compatibility
        world.insert_resource(WorldBounds {
            width: config.arena.width,
            height: config.arena.height,
        });
        world.insert_resource(TickState::new(config.tick_rate));
        world.insert_resource(TelemetryBuffer::default());
        world.insert_resource(GameConfigResource(config.clone()));
        world.insert_resource(RoundState::default());
        let obstacle_rects: Vec<ObstacleRect> = config.obstacles.iter().map(|o| ObstacleRect {
            x: o.x, y: o.y, width: o.width, height: o.height,
        }).collect();
        world.insert_resource(ObstacleLayout(obstacle_rects));

        // 3D physics world
        let dt = 1.0 / config.tick_rate as f32;
        let mut physics3d = Physics3DState::new(
            (cs_config.arena_width, cs_config.arena_height, cs_config.arena_depth),
            cs_config.gravity,
            dt,
        );

        // Build map geometry
        let mut obstacle_colliders_3d = Vec::new();
        let wall_height = cs_config.arena_height * 0.6;

        // Add obstacles from config
        for obs in &config.obstacles {
            let cx = obs.x + obs.width / 2.0;
            let cz = obs.y + obs.height / 2.0;
            let pos = Vec3::new(cx, wall_height / 2.0, cz);
            let half = Vec3::new(obs.width / 2.0, wall_height / 2.0, obs.height / 2.0);
            let (_bh, ch) = physics3d.add_static_box(pos, half);
            obstacle_colliders_3d.push(ch);
        }
        world.insert_resource(ObstacleColliders3D(obstacle_colliders_3d));

        // NavGrid for A* pathfinding (operates on XZ plane via Vec2)
        let nav_obstacles: Vec<(Vec2, Vec2)> = config.obstacles.iter().map(|o| {
            let center = Vec2::new(o.x + o.width / 2.0, o.y + o.height / 2.0);
            let half = Vec2::new(o.width / 2.0, o.height / 2.0);
            (center, half)
        }).collect();
        let nav_grid = NavGrid::from_obstacles(
            cs_config.arena_width,
            cs_config.arena_depth,
            &nav_obstacles,
            2.0,
            1.0,
        );
        world.insert_resource(nav_grid);
        world.insert_resource(CandidatePositionBuffer::default());

        use crate::scenarios::tactical_deathmatch::TacticalConfig;
        let tactical_config = TacticalConfig {
            reward_mode: crate::scenarios::tactical_deathmatch::RewardMode::Base,
            candidate_distance: config.extra_f32("candidate_distance", 15.0),
            sensor_rays: config.extra_usize("sensor_rays", 64),
            sensor_range: config.extra_f32("sensor_range", 60.0),
        };
        world.insert_resource(tactical_config);

        // Spawn points: T-side at low Z, CT-side at high Z
        let ground_y = cs_config.agent_half_height + cs_config.agent_radius + 0.1;
        let aw = cs_config.arena_width;
        let ad = cs_config.arena_depth;

        let mut t_spawns: Vec<Vec3> = (0..ppt)
            .map(|i| {
                let spacing = aw / (ppt as f32 + 1.0);
                Vec3::new(spacing * (i as f32 + 1.0), ground_y, 4.0)
            })
            .collect();
        t_spawns.push(Vec3::new(aw * 0.2, ground_y, ad * 0.2));
        t_spawns.push(Vec3::new(aw * 0.5, ground_y, ad * 0.15));
        t_spawns.push(Vec3::new(aw * 0.8, ground_y, ad * 0.2));

        let mut ct_spawns: Vec<Vec3> = (0..ppt)
            .map(|i| {
                let spacing = aw / (ppt as f32 + 1.0);
                Vec3::new(spacing * (i as f32 + 1.0), ground_y, ad - 4.0)
            })
            .collect();
        ct_spawns.push(Vec3::new(aw * 0.2, ground_y, ad * 0.8));
        ct_spawns.push(Vec3::new(aw * 0.5, ground_y, ad * 0.85));
        ct_spawns.push(Vec3::new(aw * 0.8, ground_y, ad * 0.8));

        // Bomb sites
        let bomb_sites = BombSites {
            site_a_center: Vec3::new(aw * 0.25, ground_y, ad * 0.75),
            site_a_radius: 6.0,
            site_b_center: Vec3::new(aw * 0.75, ground_y, ad * 0.75),
            site_b_radius: 6.0,
        };

        // All spawn points for 2D compat
        let all_spawns_2d: Vec<Vec2> = t_spawns.iter().chain(ct_spawns.iter())
            .map(|p| Vec2::new(p.x, p.z))
            .collect();
        world.insert_resource(SpawnPointPool(all_spawns_2d.clone()));
        world.insert_resource(SpawnPoints3D(t_spawns.iter().chain(ct_spawns.iter()).cloned().collect()));
        world.insert_resource(CsSpawnPoints {
            t_spawns: t_spawns.clone(),
            ct_spawns: ct_spawns.clone(),
        });
        world.insert_resource(bomb_sites);

        // Emit RoundStart for viewer
        world.resource_mut::<TelemetryBuffer>().push(TelemetryEvent::RoundStart {
            tick: 0,
            obstacles: config.obstacles.iter().map(|o| ObstacleRect {
                x: o.x, y: o.y, width: o.width, height: o.height,
            }).collect(),
            spawn_points: all_spawns_2d,
        });

        // Spawn agents
        let mut source_id: u32 = 0;
        let mut rng = rand::rng();

        // T-side (team 0)
        let mut first_t_entity = None;
        for i in 0..ppt {
            let spawn_pos = t_spawns[i as usize];
            let jitter = Vec3::new(rng.random_range(-0.5f32..0.5), 0.0, rng.random_range(-0.5f32..0.5));
            let pos_3d = spawn_pos + jitter;
            let dummy_2d = Vec2::new(pos_3d.x, pos_3d.z);

            let (body_2d, collider_2d) = physics.add_dynamic_body(dummy_2d, 0.5);
            let (body_3d, collider_3d) = physics3d.add_capsule_agent(
                pos_3d, cs_config.agent_half_height, cs_config.agent_radius, 0,
            );

            let eid = world.spawn((
                Position(dummy_2d),
                Velocity(Vec2::ZERO),
                Facing(std::f32::consts::FRAC_PI_2),
                Health { current: 100.0, max: 100.0 },
                Team(0),
                Weapon { damage: 25.0, fire_rate: 0.3, range: 50.0, cooldown_remaining: 0.0 },
                Agent { source_id },
                PhysicsHandle { body: body_2d, collider: collider_2d },
                Position3D(pos_3d),
                Velocity3D(Vec3::ZERO),
                Facing3D { yaw: std::f32::consts::FRAC_PI_2, pitch: 0.0 },
                PhysicsHandle3D { body: body_3d, collider: collider_3d },
                CsSide::Terrorist,
                Armor { has_armor: false },
            )).id();
            world.entity_mut(eid).insert((
                Inventory { weapons: vec![make_pistol()], active: 0 },
                EnemyMemory::default(),
                PathState::default(),
                LosTracker::default(),
                AgentGoal::default(),
            ));
            if i == 0 { first_t_entity = Some(eid); }
            source_id += 1;
        }
        if let Some(e) = first_t_entity {
            world.entity_mut(e).insert(BombCarrier);
        }

        // CT-side (team 1)
        for i in 0..ppt {
            let spawn_pos = ct_spawns[i as usize];
            let jitter = Vec3::new(rng.random_range(-0.5f32..0.5), 0.0, rng.random_range(-0.5f32..0.5));
            let pos_3d = spawn_pos + jitter;
            let dummy_2d = Vec2::new(pos_3d.x, pos_3d.z);

            let (body_2d, collider_2d) = physics.add_dynamic_body(dummy_2d, 0.5);
            let (body_3d, collider_3d) = physics3d.add_capsule_agent(
                pos_3d, cs_config.agent_half_height, cs_config.agent_radius, 1,
            );

            let ct_eid = world.spawn((
                Position(dummy_2d),
                Velocity(Vec2::ZERO),
                Facing(-std::f32::consts::FRAC_PI_2),
                Health { current: 100.0, max: 100.0 },
                Team(1),
                Weapon { damage: 25.0, fire_rate: 0.3, range: 50.0, cooldown_remaining: 0.0 },
                Agent { source_id },
                PhysicsHandle { body: body_2d, collider: collider_2d },
                Position3D(pos_3d),
                Velocity3D(Vec3::ZERO),
                Facing3D { yaw: -std::f32::consts::FRAC_PI_2, pitch: 0.0 },
                PhysicsHandle3D { body: body_3d, collider: collider_3d },
                CsSide::CounterTerrorist,
                Armor { has_armor: false },
            )).id();
            world.entity_mut(ct_eid).insert((
                Inventory { weapons: vec![make_pistol()], active: 0 },
                EnemyMemory::default(),
                PathState::default(),
                LosTracker::default(),
                AgentGoal::default(),
            ));
            source_id += 1;
        }

        // Assign bomb to first T-side agent
        {
            let mut role_query = world.query::<(Entity, &Team, &Agent)>();
            let mut entities_sorted: Vec<(Entity, u8, u32)> = role_query
                .iter(world)
                .map(|(e, t, a)| (e, t.0, a.source_id))
                .collect();
            entities_sorted.sort_by_key(|(_, _, sid)| *sid);

            for (entity, team, _) in &entities_sorted {
                if *team == 0 {
                    world.entity_mut(*entity).insert(BombCarrier);
                    break;
                }
            }
        }

        let round_state = CsRoundState::new(&cs_config);
        world.insert_resource(round_state);
        world.insert_resource(BombState::default());
        world.insert_resource(cs_config);
        world.insert_resource(physics3d);
    }

    fn register_systems(&self, schedule: &mut Schedule) {
        if self.dummy_ai {
            schedule.add_systems(
                (cs_goal_assignment_system, super::cs_lite_dummy_ai::cs_dummy_ai_system)
                    .chain()
                    .in_set(EnginePhase::AiDecisions),
            );
        } else {
            schedule.add_systems(
                (cs_goal_assignment_system, cs_scripted_ai_system)
                    .chain()
                    .in_set(EnginePhase::AiDecisions),
            );
        }
        schedule.add_systems(
            (cs_compute_candidates_system, cs_facing_system, cs_tactical_movement_system)
                .chain()
                .in_set(EnginePhase::PrePhysics),
        );
        schedule.add_systems(
            (cs_physics_step, cs_sync_system)
                .chain()
                .in_set(EnginePhase::PhysicsStep),
        );
        schedule.add_systems(
            (
                crate::weapons::inventory_cooldown_system,
                cs_weapon_switch_system,
                cs_reload_system,
                cs_combat_system,
                cs_bomb_system,
            )
                .chain()
                .in_set(EnginePhase::GameLogic),
        );
        schedule.add_systems(
            (cs_death_system, cs_round_state_system)
                .chain()
                .in_set(EnginePhase::StateTransitions),
        );
        schedule.add_systems(
            cs_telemetry_system.in_set(EnginePhase::Telemetry),
        );
    }

    fn observe(&self, world: &World, agent: Entity, writer: &mut ObsWriter) {
        let config = world.resource::<CsLiteConfig>();
        let physics3d = world.resource::<Physics3DState>();
        let round = world.resource::<CsRoundState>();
        let _bomb = world.resource::<BombState>();
        let arena_diag = config.arena_diagonal();
        let max_speed = config.max_speed;

        let pos3 = world.get::<Position3D>(agent).map(|p| p.0).unwrap_or_default();
        let vel3 = world.get::<Velocity3D>(agent).map(|v| v.0).unwrap_or_default();
        let facing3d = world.get::<Facing3D>(agent).cloned().unwrap_or_default();
        let health = world.get::<Health>(agent);
        let hp = health.map(|h| h.current).unwrap_or(0.0);
        let max_hp = health.map(|h| h.max).unwrap_or(1.0);
        let inv = world.get::<Inventory>(agent);
        let self_collider = world.get::<PhysicsHandle3D>(agent).map(|ph| ph.collider);
        let has_armor = world.get::<Armor>(agent).is_some_and(|a| a.has_armor);
        let self_team = world.get::<Team>(agent).map(|t| t.0).unwrap_or(0);
        let is_dead = world.get::<Dead>(agent).is_some();

        let cooldown_norm = inv
            .and_then(|i| i.active_weapon())
            .map(|w| w.cooldown_fraction())
            .unwrap_or(0.0);

        // Self state [12]
        writer.write("self_state", &[
            pos3.x / config.arena_width,
            pos3.y / config.arena_height,
            pos3.z / config.arena_depth,
            vel3.x / max_speed,
            vel3.y / max_speed.max(10.0),
            vel3.z / max_speed,
            facing3d.yaw.sin(),
            facing3d.yaw.cos(),
            facing3d.pitch / config.pitch_limit,
            hp / max_hp,
            if has_armor { 1.0 } else { 0.0 },
            cooldown_norm,
        ]);

        // Weapon state [20]: 4 weapon slots x 4 features + 4 active one-hot
        let mut weapon_obs = vec![0.0f32; 20];
        if let Some(inventory) = inv {
            for (i, slot) in inventory.weapons.iter().enumerate() {
                if i >= 4 { break; }
                weapon_obs[i * 4] = 1.0; // owned
                weapon_obs[i * 4 + 1] = slot.ammo_fraction();
                weapon_obs[i * 4 + 2] = slot.cooldown_fraction();
                weapon_obs[i * 4 + 3] = if slot.is_reloading { 1.0 } else { 0.0 };
            }
            if inventory.active < 4 {
                weapon_obs[16 + inventory.active] = 1.0;
            }
        }
        writer.write("weapon_state", &weapon_obs);

        // Teammate state [(ppt-1), 5]
        let registry = world.resource::<AgentRegistry>();
        let ppt = config.players_per_team as usize;
        let mut teammate_data = Vec::new();
        for &e in &registry.agents {
            if e == agent { continue; }
            let e_team = world.get::<Team>(e).map(|t| t.0).unwrap_or(0);
            if e_team != self_team { continue; }
            let e_pos = world.get::<Position3D>(e).map(|p| p.0).unwrap_or_default();
            let e_hp = world.get::<Health>(e).map(|h| h.current / h.max).unwrap_or(0.0);
            let e_alive = if world.get::<Dead>(e).is_some() { 0.0 } else { 1.0 };
            let delta = e_pos - pos3;
            teammate_data.extend_from_slice(&[
                delta.x / arena_diag,
                delta.y / arena_diag,
                delta.z / arena_diag,
                e_hp,
                e_alive,
            ]);
        }
        writer.write_padded("teammate_state", &teammate_data, (ppt - 1) * 5);

        // Enemy state [ppt, 10]
        let facing_dir = facing3d.direction();
        let mut enemy_data = Vec::new();
        for &e in &registry.agents {
            if e == agent { continue; }
            let e_team = world.get::<Team>(e).map(|t| t.0).unwrap_or(0);
            if e_team == self_team { continue; }

            let e_pos = world.get::<Position3D>(e).map(|p| p.0).unwrap_or_default();
            let e_vel = world.get::<Velocity3D>(e).map(|v| v.0).unwrap_or_default();
            let e_dead = world.get::<Dead>(e).is_some();
            let delta = e_pos - pos3;
            let dist = delta.length();

            let e_hp = world.get::<Health>(e)
                .map(|h| (h.current / h.max).max(0.0))
                .unwrap_or(0.0);

            let los = if !e_dead && !is_dead && dist > 0.1 {
                let e_collider = world.get::<PhysicsHandle3D>(e).map(|ph| ph.collider);
                let dir = delta / dist;
                match physics3d.cast_ray(
                    pos3 + Vec3::Y * config.eye_height,
                    dir, dist, self_collider,
                ) {
                    Some((hit_col, _)) => e_collider.is_some_and(|ec| hit_col == ec),
                    None => true,
                }
            } else {
                false
            };

            let angle_to = if dist > 0.1 {
                facing_dir.dot(delta.normalize())
            } else {
                1.0
            };

            // Only reveal position if we have line of sight
            if los {
                enemy_data.extend_from_slice(&[
                    delta.x / arena_diag,
                    delta.y / arena_diag,
                    delta.z / arena_diag,
                    (e_vel.x - vel3.x) / max_speed,
                    (e_vel.y - vel3.y) / max_speed.max(10.0),
                    (e_vel.z - vel3.z) / max_speed,
                    e_hp,
                    1.0,
                    angle_to,
                    (dist / arena_diag).min(1.0),
                ]);
            } else {
                enemy_data.extend_from_slice(&[0.0; 10]);
            }
        }
        writer.write_padded("enemy_state", &enemy_data, ppt * 10);

        // Round info [9]
        let rounds_to_win = round.max_rounds / 2 + 1;
        writer.write("round_info", &[
            if round.phase == RoundPhase::BuyFreeze { 1.0 } else { 0.0 },
            if round.phase == RoundPhase::Active { 1.0 } else { 0.0 },
            if round.phase == RoundPhase::RoundEnd { 1.0 } else { 0.0 },
            (round.phase_timer / round.round_time_limit).clamp(0.0, 1.0),
            round.t_score as f32 / rounds_to_win as f32,
            round.ct_score as f32 / rounds_to_win as f32,
            round.t_alive as f32 / config.players_per_team as f32,
            round.ct_alive as f32 / config.players_per_team as f32,
            if self_team == 0 { 1.0 } else { 0.0 },
        ]);

        // Bomb state [8]
        let bomb = world.resource::<BombState>();
        let has_bomb_carrier = world.get::<BombCarrier>(agent).is_some();
        let bomb_pos_delta = if let Some(bp) = bomb.plant_position {
            let d = bp - pos3;
            [d.x / config.arena_width, d.y / config.arena_height, d.z / config.arena_depth]
        } else {
            [0.0, 0.0, 0.0]
        };
        let bomb_timer_frac = if bomb.planted {
            if let Some(pt) = bomb.plant_tick {
                let elapsed = world.resource::<TickState>().tick.saturating_sub(pt) as f32;
                let bomb_time = 40.0 * config.arena_width.max(1.0) / config.arena_width; // 40s * tick_rate
                (elapsed / (40.0 * 64.0)).min(1.0)
            } else { 0.0 }
        } else { 0.0 };
        writer.write("bomb_state", &[
            if bomb.planted { 1.0 } else { 0.0 },
            bomb_pos_delta[0],
            bomb_pos_delta[1],
            bomb_pos_delta[2],
            bomb_timer_frac,
            if has_bomb_carrier { 1.0 } else { 0.0 },
            if bomb.plant_site == Some(0) { 1.0 } else { 0.0 },
            if bomb.plant_site == Some(1) { 1.0 } else { 0.0 },
        ]);

        // Candidates [12, 5]
        let candidate_buffer = world.resource::<CandidatePositionBuffer>();
        if let Some(cand_set) = candidate_buffer.get(agent) {
            let features = cand_set.as_obs_features();
            writer.write("candidates", &features);
        } else {
            writer.write("candidates", &[0.0; 60]);
        }

        // Raycasts 3D
        let total_rays = config.ray_h_count * config.ray_v_count;
        let mut ray_data = Vec::with_capacity(total_rays * 2);
        let eye_pos = pos3 + Vec3::Y * config.eye_height;
        let h_center = (config.ray_h_count as f32 - 1.0) / 2.0;
        let v_center = (config.ray_v_count as f32 - 1.0) / 2.0;

        for vi in 0..config.ray_v_count {
            let pitch_offset = if config.ray_v_count > 1 {
                config.ray_v_fov * (vi as f32 - v_center) / (config.ray_v_count as f32 - 1.0)
            } else {
                0.0
            };
            for hi in 0..config.ray_h_count {
                let yaw_offset = if config.ray_h_count > 1 {
                    config.ray_h_fov * (hi as f32 - h_center) / (config.ray_h_count as f32 - 1.0)
                } else {
                    0.0
                };
                let ray_yaw = facing3d.yaw + yaw_offset;
                let ray_pitch = (facing3d.pitch + pitch_offset).clamp(-config.pitch_limit, config.pitch_limit);
                let ray_dir = Vec3::new(
                    ray_yaw.cos() * ray_pitch.cos(),
                    ray_pitch.sin(),
                    ray_yaw.sin() * ray_pitch.cos(),
                );
                let (dist_norm, hit_type) = physics3d.cast_ray_classified(
                    eye_pos, ray_dir, config.ray_max_range, self_collider, self_team,
                );
                ray_data.push(dist_norm);
                ray_data.push(hit_type);
            }
        }
        writer.write("raycasts_3d", &ray_data);

        // Audio 3D [6]: gunshot (bearing, proximity, fresh) + footstep (bearing, loudness) + own noise
        let shot_buffer = world.resource::<ShotEventBuffer>();
        let mut shot_yaw_bearing = 0.0f32;
        let mut shot_proximity = 0.0f32;
        let mut shot_fresh = 0.0f32;
        for event in &shot_buffer.events {
            if event.shooter == agent { continue; }
            let shot_pos = Vec3::new(event.origin.x, config.eye_height, event.origin.y);
            let delta = shot_pos - pos3;
            let d = delta.length();
            if d < arena_diag && d > 0.1 {
                let to_shot = delta / d;
                let forward_xz = facing3d.forward_xz();
                let right_xz = facing3d.right_xz();
                let dot_f = forward_xz.dot(Vec3::new(to_shot.x, 0.0, to_shot.z));
                let dot_r = right_xz.dot(Vec3::new(to_shot.x, 0.0, to_shot.z));
                shot_yaw_bearing = dot_r.atan2(dot_f);
                shot_proximity = 1.0 - d / arena_diag;
                shot_fresh = 1.0;
            }
        }

        let mut footstep_bearing = 0.0f32;
        let mut footstep_loudness = 0.0f32;
        let mut nearest_moving_dist = f32::MAX;
        for &e in &registry.agents {
            if e == agent { continue; }
            let e_team = world.get::<Team>(e).map(|t| t.0).unwrap_or(0);
            if e_team == self_team || world.get::<Dead>(e).is_some() { continue; }
            let e_pos = world.get::<Position3D>(e).map(|p| p.0).unwrap_or_default();
            let e_vel = world.get::<Velocity3D>(e).map(|v| v.0).unwrap_or_default();
            let e_speed = (e_vel.x * e_vel.x + e_vel.z * e_vel.z).sqrt();
            if e_speed < 0.1 { continue; }
            let delta = e_pos - pos3;
            let d = delta.length();
            if d < nearest_moving_dist && d > 0.1 {
                nearest_moving_dist = d;
                let to_enemy = delta / d;
                let forward_xz = facing3d.forward_xz();
                let right_xz = facing3d.right_xz();
                let dot_f = forward_xz.dot(Vec3::new(to_enemy.x, 0.0, to_enemy.z));
                let dot_r = right_xz.dot(Vec3::new(to_enemy.x, 0.0, to_enemy.z));
                footstep_bearing = dot_r.atan2(dot_f);
                footstep_loudness = (e_speed / max_speed).min(1.0);
            }
        }
        let own_speed = (vel3.x * vel3.x + vel3.z * vel3.z).sqrt();
        let own_noise = (own_speed / max_speed).min(1.0);

        writer.write("audio_3d", &[
            shot_yaw_bearing / std::f32::consts::PI,
            shot_proximity,
            shot_fresh,
            footstep_bearing / std::f32::consts::PI,
            footstep_loudness,
            own_noise,
        ]);

        // Aim state [5]: what the auto-aim is doing, so the agent can time shots
        let mut aim_yaw_error = 0.0f32;
        let mut aim_dist = 0.0f32;
        let mut aim_on_target = 0.0f32;
        let mut aim_ticks_to_lock = 0.0f32;
        let mut aim_has_target = 0.0f32;
        {
            let mut best_d = f32::MAX;
            for &e in &registry.agents {
                if e == agent { continue; }
                let e_team = world.get::<Team>(e).map(|t| t.0).unwrap_or(0);
                if e_team == self_team || world.get::<Dead>(e).is_some() { continue; }
                let e_pos = world.get::<Position3D>(e).map(|p| p.0).unwrap_or_default();
                let e_eye = e_pos + Vec3::Y * config.eye_height;
                let d = pos3.distance(e_pos);
                if d < best_d && d > 0.1 && !physics3d.obstacles_block_los(eye_pos, e_eye) {
                    best_d = d;
                    let delta = e_pos - pos3;
                    let target_yaw = delta.z.atan2(delta.x);
                    let mut yd = target_yaw - facing3d.yaw;
                    while yd > std::f32::consts::PI { yd -= std::f32::consts::TAU; }
                    while yd < -std::f32::consts::PI { yd += std::f32::consts::TAU; }
                    let max_yaw_tick = 4.0 / 60.0;
                    aim_yaw_error = (yd / max_yaw_tick).clamp(-1.0, 1.0);
                    aim_dist = (d / arena_diag).min(1.0);
                    aim_on_target = if yd.abs() < config.hitbox_radius / d.max(1.0) { 1.0 } else { 0.0 };
                    aim_ticks_to_lock = (yd.abs() / max_yaw_tick).min(1.0);
                    aim_has_target = 1.0;
                }
            }
        }
        writer.write("aim_state", &[aim_yaw_error, aim_dist, aim_on_target, aim_ticks_to_lock, aim_has_target]);

        // Goal state [11]: objective one-hot [5] + relative target [3] + posture one-hot [3]
        let goal_obs = if let Some(goal) = world.get::<AgentGoal>(agent) {
            let obj = match goal.objective {
                ObjectiveType::PlantBomb => [1.0, 0.0, 0.0, 0.0, 0.0],
                ObjectiveType::DefuseBomb => [0.0, 1.0, 0.0, 0.0, 0.0],
                ObjectiveType::HoldPosition => [0.0, 0.0, 1.0, 0.0, 0.0],
                ObjectiveType::Eliminate => [0.0, 0.0, 0.0, 1.0, 0.0],
                ObjectiveType::Rotate => [0.0, 0.0, 0.0, 0.0, 1.0],
            };
            let delta = goal.target_position - pos3;
            let dx = (delta.x / config.arena_width).clamp(-1.0, 1.0);
            let dy = (delta.y / config.arena_height).clamp(-1.0, 1.0);
            let dz = (delta.z / config.arena_depth).clamp(-1.0, 1.0);
            let posture = match goal.posture {
                Posture::Aggressive => [1.0, 0.0, 0.0],
                Posture::Default => [0.0, 1.0, 0.0],
                Posture::Passive => [0.0, 0.0, 1.0],
            };
            [obj[0], obj[1], obj[2], obj[3], obj[4], dx, dy, dz, posture[0], posture[1], posture[2]]
        } else {
            [0.0; 11]
        };
        writer.write("goal_state", &goal_obs);

        let frozen = round.phase != RoundPhase::Active;
        let active_weapon = inv.and_then(|i| i.active_weapon());
        let can_shoot = !frozen && !is_dead && active_weapon
            .map(|w| w.cooldown_remaining <= 0.0 && w.ammo > 0 && !w.is_reloading)
            .unwrap_or(false);
        let can_reload = !frozen && !is_dead && active_weapon
            .map(|w| w.ammo < w.max_ammo && !w.is_reloading)
            .unwrap_or(false);
        let alive_and_active = if !is_dead && !frozen { 1.0 } else { 0.0 };

        let bomb = world.resource::<BombState>();
        let bomb_sites = world.resource::<BombSites>();
        let has_bomb_carrier_mask = world.get::<BombCarrier>(agent).is_some();
        let can_plant = !frozen && !is_dead && self_team == 0 && has_bomb_carrier_mask && !bomb.planted && {
            let dist_a = (pos3 - bomb_sites.site_a_center).length();
            let dist_b = (pos3 - bomb_sites.site_b_center).length();
            dist_a < bomb_sites.site_a_radius || dist_b < bomb_sites.site_b_radius
        };
        let can_defuse = !frozen && !is_dead && self_team == 1 && bomb.planted
            && bomb.plant_position.map(|bp| (pos3 - bp).length() < bomb_sites.site_a_radius).unwrap_or(false);

        let mut mask = Vec::with_capacity(19);
        if let Some(cand_set) = candidate_buffer.get(agent) {
            let wmask = cand_set.walkable_mask();
            for &m in &wmask {
                mask.push(if is_dead || frozen { 0.0 } else { m });
            }
        } else {
            for _ in 0..12 { mask.push(alive_and_active); }
        }
        mask.push(1.0);
        mask.push(if can_shoot { 1.0 } else { 0.0 });
        mask.push(1.0);
        mask.push(if can_reload { 1.0 } else { 0.0 });
        mask.push(1.0);
        mask.push(if can_plant { 1.0 } else { 0.0 });
        mask.push(if can_defuse { 1.0 } else { 0.0 });

        writer.write("action_mask", &mask);
    }

    fn reward(&self, world: &World, agent: Entity) -> f32 {
        world.resource::<RewardBuffer>().get(agent)
    }

    fn is_done(&self, world: &World, _agent: Entity) -> bool {
        let round = world.resource::<CsRoundState>();
        round.match_over()
    }
}

// ---------------------------------------------------------------------------
// Weapon factory helpers
// ---------------------------------------------------------------------------

fn make_pistol() -> WeaponSlot {
    WeaponSlot {
        weapon_type: WeaponType::Rifle,
        damage: 25.0,
        fire_rate: 0.3,
        range: 50.0,
        cooldown_remaining: 0.0,
        ammo: 12,
        max_ammo: 12,
        reload_time: 1.5,
        reload_remaining: 0.0,
        is_reloading: false,
    }
}

// ---------------------------------------------------------------------------
// ECS Systems
// ---------------------------------------------------------------------------

#[allow(clippy::type_complexity)]
pub fn cs_goal_assignment_system(
    mut commands: Commands,
    agents: Query<(Entity, &Position3D, &Team, Option<&BombCarrier>, Option<&super::cs_lite_bridge::StrategyControlled>), Without<Dead>>,
    round: Res<CsRoundState>,
    bomb: Res<BombState>,
    bomb_sites: Res<BombSites>,
    config: Res<CsLiteConfig>,
    game_config: Res<GameConfigResource>,
    tick: Res<TickState>,
) {
    let randomize = game_config.0.extra_bool("randomize_goals", false);

    if randomize && tick.tick % 256 == 0 {
        let mut rng = rand::rng();
        for (entity, _, _, _, strategy) in &agents {
            if strategy.is_some() { continue; }
            let objective = match rng.random_range(0u8..5) {
                0 => ObjectiveType::PlantBomb,
                1 => ObjectiveType::DefuseBomb,
                2 => ObjectiveType::HoldPosition,
                3 => ObjectiveType::Eliminate,
                _ => ObjectiveType::Rotate,
            };
            let target_position = Vec3::new(
                rng.random_range(2.0..config.arena_width - 2.0),
                0.0,
                rng.random_range(2.0..config.arena_depth - 2.0),
            );
            let posture = match rng.random_range(0u8..3) {
                0 => Posture::Aggressive,
                1 => Posture::Default,
                _ => Posture::Passive,
            };
            commands.entity(entity).insert(AgentGoal { objective, target_position, posture });
        }
        return;
    }

    if round.phase == RoundPhase::BuyFreeze {
        let mid = (bomb_sites.site_a_center + bomb_sites.site_b_center) * 0.5;
        for (entity, pos3, team, carrier, strategy) in &agents {
            if strategy.is_some() { continue; }
            let goal = if team.0 == 0 {
                if carrier.is_some() {
                    AgentGoal { objective: ObjectiveType::HoldPosition, target_position: pos3.0, posture: Posture::Default }
                } else {
                    AgentGoal { objective: ObjectiveType::HoldPosition, target_position: pos3.0, posture: Posture::Default }
                }
            } else {
                AgentGoal { objective: ObjectiveType::HoldPosition, target_position: mid, posture: Posture::Default }
            };
            commands.entity(entity).insert(goal);
        }
        return;
    }

    if round.phase != RoundPhase::Active { return; }

    for (entity, pos3, team, carrier, strategy) in &agents {
        if strategy.is_some() { continue; }
        let goal = if team.0 == 0 {
            if carrier.is_some() {
                let dist_a = (pos3.0 - bomb_sites.site_a_center).length();
                let dist_b = (pos3.0 - bomb_sites.site_b_center).length();
                let nearest_site = if dist_a < dist_b { bomb_sites.site_a_center } else { bomb_sites.site_b_center };
                AgentGoal { objective: ObjectiveType::PlantBomb, target_position: nearest_site, posture: Posture::Default }
            } else {
                let dist_a = (pos3.0 - bomb_sites.site_a_center).length();
                let dist_b = (pos3.0 - bomb_sites.site_b_center).length();
                let nearest_site = if dist_a < dist_b { bomb_sites.site_a_center } else { bomb_sites.site_b_center };
                AgentGoal { objective: ObjectiveType::Eliminate, target_position: nearest_site, posture: Posture::Aggressive }
            }
        } else if bomb.planted {
            let target = bomb.plant_position.unwrap_or(bomb_sites.site_a_center);
            AgentGoal { objective: ObjectiveType::DefuseBomb, target_position: target, posture: Posture::Aggressive }
        } else {
            let mid = (bomb_sites.site_a_center + bomb_sites.site_b_center) * 0.5;
            AgentGoal { objective: ObjectiveType::HoldPosition, target_position: mid, posture: Posture::Default }
        };
        commands.entity(entity).insert(goal);
    }
}

#[allow(clippy::type_complexity)]
pub fn cs_facing_system(
    mut agents: Query<(Entity, &mut Facing3D, &mut Facing, &Position3D, &Team), Without<Dead>>,
    all_agents: Query<(Entity, &Position3D, &Team, &PhysicsHandle3D), Without<Dead>>,
    physics3d: Res<Physics3DState>,
    config: Res<CsLiteConfig>,
    round: Res<CsRoundState>,
    tick: Res<TickState>,
) {
    if round.phase != RoundPhase::Active { return; }

    let max_yaw_per_tick = 4.0 * tick.delta;
    let max_pitch_per_tick = 3.0 * tick.delta;

    for (entity, mut facing3d, mut facing2d, pos, team) in &mut agents {
        let eye_pos = pos.0 + Vec3::Y * config.eye_height;

        let mut best_dist = f32::MAX;
        let mut target_pos = None;

        for (other_e, other_pos, other_team, _) in &all_agents {
            if other_e == entity || other_team.0 == team.0 { continue; }
            let d = pos.0.distance(other_pos.0);
            if d < 0.1 || d >= best_dist { continue; }
            let enemy_eye = other_pos.0 + Vec3::Y * config.eye_height;
            if !physics3d.obstacles_block_los(eye_pos, enemy_eye) {
                best_dist = d;
                target_pos = Some(other_pos.0);
            }
        }

        if let Some(target) = target_pos {
            let delta = target - pos.0;
            let target_yaw = delta.z.atan2(delta.x);
            let mut yaw_diff = target_yaw - facing3d.yaw;
            while yaw_diff > std::f32::consts::PI { yaw_diff -= std::f32::consts::TAU; }
            while yaw_diff < -std::f32::consts::PI { yaw_diff += std::f32::consts::TAU; }
            facing3d.yaw += yaw_diff.clamp(-max_yaw_per_tick, max_yaw_per_tick);

            let horiz = (delta.x * delta.x + delta.z * delta.z).sqrt();
            let target_pitch = if horiz > 0.01 { delta.y.atan2(horiz) } else { 0.0 };
            let pitch_diff = target_pitch - facing3d.pitch;
            facing3d.pitch = (facing3d.pitch + pitch_diff.clamp(-max_pitch_per_tick, max_pitch_per_tick))
                .clamp(-config.pitch_limit, config.pitch_limit);
        }

        facing2d.0 = facing3d.yaw;
    }
}

pub const COMPASS_DIRS: [(f32, f32); 8] = [
    (0.0, 1.0), (0.707, 0.707), (1.0, 0.0), (0.707, -0.707),
    (0.0, -1.0), (-0.707, -0.707), (-1.0, 0.0), (-0.707, 0.707),
];

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn cs_compute_candidates_system(
    agents: Query<(Entity, &Position3D, &Team, &PhysicsHandle3D), (With<Agent>, Without<Dead>)>,
    mut candidate_buffer: ResMut<CandidatePositionBuffer>,
    nav: Res<NavGrid>,
    physics3d: Res<Physics3DState>,
    config: Res<CsLiteConfig>,
    registry: Res<AgentRegistry>,
    all_agents: Query<(Entity, &Position3D, &Team, &PhysicsHandle3D), Without<Dead>>,
    round: Res<CsRoundState>,
    world_tick: Res<TickState>,
) {
    if round.phase != RoundPhase::Active { return; }

    if world_tick.tick % 4 != 0 { return; }

    let arena_diag = (config.arena_width * config.arena_width + config.arena_depth * config.arena_depth).sqrt();
    let dist = config.extra_candidate_distance();
    candidate_buffer.clear();

    for (entity, pos3, team, ph) in &agents {
        let agent_xz = Vec2::new(pos3.0.x, pos3.0.z);
        let eye_y = pos3.0.y + config.eye_height;

        let mut primary_enemy_xz: Option<Vec2> = None;
        let mut primary_enemy_3d: Option<Vec3> = None;
        let mut all_enemies_xz: Vec<(Entity, Vec2)> = Vec::new();
        let mut best_dist = f32::MAX;

        for &e in &registry.agents {
            if e == entity { continue; }
            if let Ok((_, e_pos, e_team, _)) = all_agents.get(e) {
                if e_team.0 != team.0 {
                    let exz = Vec2::new(e_pos.0.x, e_pos.0.z);
                    all_enemies_xz.push((e, exz));
                    let d = agent_xz.distance(exz);
                    if d < best_dist {
                        best_dist = d;
                        primary_enemy_xz = Some(exz);
                        primary_enemy_3d = Some(e_pos.0);
                    }
                }
            }
        }

        let enemy_xz = primary_enemy_xz.unwrap_or(agent_xz);
        let enemy_3d = primary_enemy_3d.unwrap_or(pos3.0);
        let mut set = CandidateSet::default();

        for (i, &(dx, dz)) in COMPASS_DIRS.iter().enumerate() {
            let dir = Vec2::new(dx, dz);
            let raw_pos = agent_xz + dir * dist;
            let candidate_xz = if nav.is_walkable(raw_pos) { raw_pos } else { nav.snap_to_walkable(raw_pos, dir) };
            set.positions[i] = compute_candidate_features_3d(
                candidate_xz, agent_xz, enemy_xz, &all_enemies_xz,
                &nav, &physics3d, eye_y, enemy_3d.y + config.eye_height, arena_diag, Some(ph.collider),
            );
        }

        // Stay (8)
        set.positions[8] = compute_candidate_features_3d(
            agent_xz, agent_xz, enemy_xz, &all_enemies_xz,
            &nav, &physics3d, eye_y, enemy_3d.y + config.eye_height, arena_diag, Some(ph.collider),
        );
        set.positions[8].path_distance = 0.0;

        // Cover (9)
        let cover_xz = find_nearest_cover_3d(agent_xz, enemy_xz, enemy_3d, &nav, &physics3d, eye_y, dist * 3.0);
        set.positions[9] = compute_candidate_features_3d(
            cover_xz, agent_xz, enemy_xz, &all_enemies_xz,
            &nav, &physics3d, eye_y, enemy_3d.y + config.eye_height, arena_diag, Some(ph.collider),
        );

        // Advance (10): A* toward enemy, pick a point partway along the path
        let advance_xz = if let Some(path) = nav.astar(agent_xz, enemy_xz) {
            let target_dist = dist.min(5.0 * nav.cell_size);
            let mut walked = 0.0f32;
            let mut advance = agent_xz;
            for i in 1..path.len() {
                let seg = path[i].distance(path[i - 1]);
                walked += seg;
                advance = path[i];
                if walked >= target_dist { break; }
            }
            advance
        } else {
            let to_enemy = (enemy_xz - agent_xz).normalize_or_zero();
            let mut a = agent_xz;
            for step in 1..=5 {
                let c = agent_xz + to_enemy * (step as f32 * nav.cell_size);
                if nav.is_walkable(c) { a = c; } else { break; }
            }
            a
        };
        set.positions[10] = compute_candidate_features_3d(
            advance_xz, agent_xz, enemy_xz, &all_enemies_xz,
            &nav, &physics3d, eye_y, enemy_3d.y + config.eye_height, arena_diag, Some(ph.collider),
        );

        // Retreat (11)
        let retreat_dir = (agent_xz - enemy_xz).normalize_or_zero();
        let raw_retreat = agent_xz + retreat_dir * dist * 2.0;
        let clamped = raw_retreat.clamp(
            Vec2::new(3.0, 3.0),
            Vec2::new(config.arena_width - 3.0, config.arena_depth - 3.0),
        );
        let retreat_xz = if nav.is_walkable(clamped) { clamped } else { nav.snap_to_walkable(clamped, retreat_dir) };
        set.positions[11] = compute_candidate_features_3d(
            retreat_xz, agent_xz, enemy_xz, &all_enemies_xz,
            &nav, &physics3d, eye_y, enemy_3d.y + config.eye_height, arena_diag, Some(ph.collider),
        );

        candidate_buffer.insert(entity, set);
    }
}

fn find_nearest_cover_3d(
    agent_xz: Vec2, enemy_xz: Vec2, enemy_3d: Vec3,
    nav: &NavGrid, physics3d: &Physics3DState, eye_y: f32, search_radius: f32,
) -> Vec2 {
    let steps = (search_radius / nav.cell_size) as usize;
    let mut best_pos = agent_xz;
    let mut best_dist = f32::MAX;

    for ring in 1..=steps {
        for i in 0..(ring * 8) {
            let angle = i as f32 * std::f32::consts::TAU / (ring * 8) as f32;
            let offset = Vec2::new(angle.cos(), angle.sin()) * (ring as f32 * nav.cell_size);
            let candidate = agent_xz + offset;
            if !nav.is_walkable(candidate) { continue; }

            let from = Vec3::new(candidate.x, eye_y, candidate.y);
            let to = Vec3::new(enemy_xz.x, enemy_3d.y + 1.6, enemy_xz.y);
            if !physics3d.has_line_of_sight(from, to, None) {
                let d = agent_xz.distance(candidate);
                if d < best_dist { best_dist = d; best_pos = candidate; }
            }
        }
    }
    best_pos
}

fn compute_candidate_features_3d(
    candidate_xz: Vec2, agent_xz: Vec2, enemy_xz: Vec2,
    all_enemies: &[(Entity, Vec2)],
    nav: &NavGrid, physics3d: &Physics3DState,
    eye_y: f32, enemy_eye_y: f32, arena_diag: f32,
    _exclude: Option<rapier3d::prelude::ColliderHandle>,
) -> CandidatePosition {
    let path_dist = nav.path_distance(agent_xz, candidate_xz)
        .map(|d| (d / arena_diag).min(1.0))
        .unwrap_or(1.0);

    let from_3d = Vec3::new(candidate_xz.x, eye_y, candidate_xz.y);
    let to_3d = Vec3::new(enemy_xz.x, enemy_eye_y, enemy_xz.y);
    let has_los = physics3d.has_line_of_sight(from_3d, to_3d, None);

    let dist_to_enemy = (candidate_xz.distance(enemy_xz) / arena_diag).min(1.0);
    let enemies_with_los = all_enemies.iter()
        .filter(|(_, ep)| {
            let e3d = Vec3::new(ep.x, enemy_eye_y, ep.y);
            physics3d.has_line_of_sight(from_3d, e3d, None)
        })
        .count() as f32 / all_enemies.len().max(1) as f32;

    CandidatePosition {
        world_pos: candidate_xz,
        path_distance: path_dist,
        has_los_to_enemy: has_los,
        dist_to_enemy,
        enemies_with_los,
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn cs_tactical_movement_system(
    mut query: Query<(Entity, &mut PathState, &Position3D, &PhysicsHandle3D), Without<Dead>>,
    raw_buffer: Res<RawActionBuffer>,
    action_space: Res<ActionSpaceDef>,
    candidate_buffer: Res<CandidatePositionBuffer>,
    nav: Res<NavGrid>,
    config: Res<CsLiteConfig>,
    mut physics3d: ResMut<Physics3DState>,
    round: Res<CsRoundState>,
) {
    for (entity, mut path_state, pos3, ph3d) in &mut query {
        if round.phase != RoundPhase::Active {
            let cv = physics3d.body_velocity(ph3d.body).unwrap_or_default();
            physics3d.set_body_linvel(ph3d.body, Vec3::new(0.0, cv.y, 0.0));
            continue;
        }

        let Some(raw) = raw_buffer.get(entity) else {
            physics3d.set_body_linvel(ph3d.body, Vec3::ZERO);
            continue;
        };
        if raw.len() < action_space.total_size {
            physics3d.set_body_linvel(ph3d.body, Vec3::ZERO);
            continue;
        }

        let move_slice = action_space.extract_head(raw, 0);
        let target_idx = (move_slice[0].round() as usize).min(11);

        let target_changed = path_state.target_candidate != Some(target_idx);
        let path_done = path_state.is_complete();

        if target_changed || path_done {
            // Clear old path so we always recompute from current position
            path_state.waypoints.clear();
            path_state.current_index = 0;
            path_state.target_candidate = Some(target_idx);

            if target_idx == 8 {
                path_state.clear();
                path_state.target_candidate = Some(8);
                let cv = physics3d.body_velocity(ph3d.body).unwrap_or_default();
                physics3d.set_body_linvel(ph3d.body, Vec3::new(0.0, cv.y, 0.0));
                continue;
            }

            if let Some(candidates) = candidate_buffer.get(entity) {
                let target_xz = candidates.positions[target_idx].world_pos;
                let current_xz = Vec2::new(pos3.0.x, pos3.0.z);

                if let Some(waypoints) = nav.astar(current_xz, target_xz) {
                    path_state.waypoints = waypoints;
                    path_state.current_index = 0;
                } else {
                    let dir_xz = (target_xz - current_xz).normalize_or_zero();
                    let cv = physics3d.body_velocity(ph3d.body).unwrap_or_default();
                    physics3d.set_body_linvel(ph3d.body, Vec3::new(
                        dir_xz.x * config.max_speed, cv.y, dir_xz.y * config.max_speed,
                    ));
                    path_state.clear();
                    path_state.target_candidate = Some(target_idx);
                    continue;
                }
            } else {
                let cv = physics3d.body_velocity(ph3d.body).unwrap_or_default();
                physics3d.set_body_linvel(ph3d.body, Vec3::new(0.0, cv.y, 0.0));
                continue;
            }
        }

        if let Some(waypoint) = path_state.current_waypoint() {
            let current_xz = Vec2::new(pos3.0.x, pos3.0.z);
            let to_wp = waypoint - current_xz;
            let dist = to_wp.length();

            if dist < 2.0 {
                path_state.advance();
                if let Some(next_wp) = path_state.current_waypoint() {
                    let dir = (next_wp - current_xz).normalize_or_zero();
                    let cv = physics3d.body_velocity(ph3d.body).unwrap_or_default();
                    physics3d.set_body_linvel(ph3d.body, Vec3::new(
                        dir.x * config.max_speed, cv.y, dir.y * config.max_speed,
                    ));
                } else {
                    let cv = physics3d.body_velocity(ph3d.body).unwrap_or_default();
                    physics3d.set_body_linvel(ph3d.body, Vec3::new(0.0, cv.y, 0.0));
                }
            } else {
                let dir = to_wp / dist;
                let cv = physics3d.body_velocity(ph3d.body).unwrap_or_default();
                physics3d.set_body_linvel(ph3d.body, Vec3::new(
                    dir.x * config.max_speed, cv.y, dir.y * config.max_speed,
                ));
            }
        } else {
            let cv = physics3d.body_velocity(ph3d.body).unwrap_or_default();
            physics3d.set_body_linvel(ph3d.body, Vec3::new(0.0, cv.y, 0.0));
        }
    }
}

pub fn cs_physics_step(mut physics3d: ResMut<Physics3DState>) {
    physics3d.step();
}

pub fn cs_sync_system(
    physics3d: Res<Physics3DState>,
    mut query: Query<(&PhysicsHandle3D, &mut Position3D, &mut Velocity3D, &mut Position)>,
) {
    for (ph3d, mut pos3, mut vel3, mut pos2d) in &mut query {
        if let Some(p) = physics3d.body_position(ph3d.body) {
            pos3.0 = p;
            pos2d.0 = Vec2::new(p.x, p.z);
        }
        if let Some(v) = physics3d.body_velocity(ph3d.body) {
            vel3.0 = v;
        }
    }
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn cs_scripted_ai_system(
    agents: Query<
        (Entity, &Position3D, &Facing3D, &Team, &PhysicsHandle3D, Option<&Inventory>, Option<&BombCarrier>, Option<&AgentGoal>),
        (With<Agent>, Without<Dead>),
    >,
    mut raw_buffer: ResMut<RawActionBuffer>,
    config: Res<CsLiteConfig>,
    round: Res<CsRoundState>,
    tick: Res<TickState>,
    physics3d: Res<Physics3DState>,
    bomb: Res<BombState>,
    bomb_sites: Res<BombSites>,
) {
    if round.phase != RoundPhase::Active { return; }

    let all: Vec<_> = agents.iter()
        .map(|(e, p, _f, t, ph, _, _, _)| (e, p.0, t.0, ph.collider))
        .collect();

    for (entity, pos3, _facing, team, _ph, inv, carrier, goal) in &agents {
        let pos = pos3.0;
        let team_id = team.0;
        if raw_buffer.get(entity).is_some() { continue; }

        let eye_pos = pos + Vec3::Y * config.eye_height;

        let mut nearest_enemy_pos = None;
        let mut nearest_enemy_dist = f32::MAX;
        let mut visible_enemy_pos = None;
        let mut visible_enemy_dist = f32::MAX;

        for &(other_e, other_pos, other_team, _) in &all {
            if other_e == entity || other_team == team_id { continue; }
            let d = pos.distance(other_pos);
            if d < 0.1 { continue; }

            if d < nearest_enemy_dist {
                nearest_enemy_dist = d;
                nearest_enemy_pos = Some(other_pos);
            }

            let enemy_eye = other_pos + Vec3::Y * config.eye_height;
            if !physics3d.obstacles_block_los(eye_pos, enemy_eye) && d < visible_enemy_dist {
                visible_enemy_dist = d;
                visible_enemy_pos = Some(other_pos);
            }
        }

        let mut shoot = 0.0f32;
        let move_target: f32;

        let agent_hash = entity.to_bits();

        if visible_enemy_pos.is_some() {
            shoot = 1.0;
        }

        let has_objective_goal = matches!(
            goal,
            Some(g) if matches!(g.objective, ObjectiveType::PlantBomb | ObjectiveType::DefuseBomb | ObjectiveType::Rotate)
                && g.target_position != Vec3::ZERO
        );

        if has_objective_goal {
            let target = goal.unwrap().target_position;
            let to_target = (target - pos).normalize_or_zero();
            if to_target.length_squared() < 0.01 {
                move_target = 8.0;
            } else {
                let mut best_compass = 0usize;
                let mut best_dot = f32::MIN;
                for (i, &(cx, cz)) in COMPASS_DIRS.iter().enumerate() {
                    let dot = to_target.x * cx + to_target.z * cz;
                    if dot > best_dot { best_dot = dot; best_compass = i; }
                }
                move_target = best_compass as f32;
            }
        } else if visible_enemy_pos.is_some() {
            if visible_enemy_dist > 12.0 {
                move_target = 10.0;
            } else if visible_enemy_dist < 5.0 {
                let phase = (tick.tick / 15 + agent_hash) % 4;
                move_target = match phase { 0 => 1.0, 1 => 3.0, 2 => 5.0, _ => 7.0 };
            } else {
                let phase = (tick.tick / 30 + agent_hash) % 3;
                move_target = match phase { 0 => 10.0, 1 => 9.0, _ => 10.0 };
            }
        } else {
            let nav_target = nearest_enemy_pos.unwrap_or(pos);
            let to_target = (nav_target - pos).normalize_or_zero();
            if to_target.length_squared() < 0.01 {
                move_target = 8.0;
            } else {
                let mut best_compass = 0usize;
                let mut best_dot = f32::MIN;
                for (i, &(cx, cz)) in COMPASS_DIRS.iter().enumerate() {
                    let dot = to_target.x * cx + to_target.z * cz;
                    if dot > best_dot { best_dot = dot; best_compass = i; }
                }
                move_target = best_compass as f32;
            }
        }

        let reload = if let Some(inventory) = inv {
            if let Some(weapon) = inventory.active_weapon() {
                if weapon.ammo == 0 && !weapon.is_reloading { 1.0 } else { 0.0 }
            } else { 0.0 }
        } else { 0.0 };

        let use_action = if team_id == 0 && carrier.is_some() && !bomb.planted {
            let dist_a = (pos - bomb_sites.site_a_center).length();
            let dist_b = (pos - bomb_sites.site_b_center).length();
            if dist_a < bomb_sites.site_a_radius || dist_b < bomb_sites.site_b_radius {
                1.0
            } else { 0.0 }
        } else if team_id == 1 && bomb.planted {
            if let Some(bp) = bomb.plant_position {
                if (pos - bp).length() < 6.0 { 2.0 } else { 0.0 }
            } else { 0.0 }
        } else { 0.0 };

        let action = vec![move_target, shoot, reload, use_action];
        raw_buffer.insert(entity, action);
    }
}

pub fn cs_weapon_switch_system(
    mut query: Query<(Entity, &mut Inventory), Without<Dead>>,
    raw_buffer: Res<RawActionBuffer>,
    action_space: Res<ActionSpaceDef>,
    round: Res<CsRoundState>,
) {
    if round.phase != RoundPhase::Active { return; }
    if action_space.heads.len() <= 4 { return; }

    for (entity, mut inv) in &mut query {
        let Some(raw) = raw_buffer.get(entity) else { continue };
        if raw.len() < action_space.total_size { continue; }
        let select_slice = action_space.extract_head(raw, 4);
        if select_slice.is_empty() { continue; }
        let new_index = select_slice[0].round() as usize;
        if new_index < inv.weapons.len() && new_index != inv.active {
            inv.active = new_index;
            inv.weapons[new_index].cooldown_remaining = 0.3;
        }
    }
}

pub fn cs_reload_system(
    mut query: Query<(Entity, &mut Inventory), Without<Dead>>,
    raw_buffer: Res<RawActionBuffer>,
    action_space: Res<ActionSpaceDef>,
    round: Res<CsRoundState>,
) {
    if round.phase != RoundPhase::Active { return; }

    for (entity, mut inv) in &mut query {
        let Some(raw) = raw_buffer.get(entity) else { continue };
        if raw.len() < action_space.total_size { continue; }

        let reload = action_space.extract_head(raw, 2)[0].round() as u8;
        if reload != 1 { continue; }

        let Some(weapon) = inv.active_weapon_mut() else { continue };
        if weapon.ammo < weapon.max_ammo && !weapon.is_reloading {
            weapon.is_reloading = true;
            weapon.reload_remaining = if weapon.reload_time > 0.0 { weapon.reload_time } else { 2.0 };
        }
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn cs_combat_system(
    mut commands: Commands,
    mut shooters: Query<
        (Entity, &Position3D, &mut Facing3D, &Velocity3D, &Team, &PhysicsHandle3D, &mut Inventory),
        Without<Dead>,
    >,
    mut targets: Query<(Entity, &Position3D, &mut Health, &Team, &PhysicsHandle3D, Option<&Armor>), Without<Dead>>,
    raw_buffer: Res<RawActionBuffer>,
    action_space: Res<ActionSpaceDef>,
    physics3d: Res<Physics3DState>,
    config: Res<CsLiteConfig>,
    tick: Res<TickState>,
    mut telemetry: ResMut<TelemetryBuffer>,
    mut rewards: ResMut<RewardBuffer>,
    mut breakdown: ResMut<RewardBreakdownBuffer>,
    mut shot_events: ResMut<ShotEventBuffer>,
    round: Res<CsRoundState>,
) {
    if round.phase != RoundPhase::Active { return; }

    let mut hits: Vec<(Entity, f32, Entity, bool)> = Vec::new();
    let mut fired: Vec<(Entity, Vec3, Vec3, f32)> = Vec::new();

    for (shooter_entity, shooter_pos, facing3d, shooter_vel, shooter_team, _shooter_ph, inventory) in &shooters {
        let Some(raw) = raw_buffer.get(shooter_entity) else { continue };
        if raw.len() < action_space.total_size { continue; }

        let shoot_val = action_space.extract_head(raw, 1)[0];
        if shoot_val < 0.5 { continue; }

        let active_weapon = match inventory.active_weapon() {
            Some(w) => w,
            None => continue,
        };
        if active_weapon.cooldown_remaining > 0.0 || active_weapon.ammo == 0 || active_weapon.is_reloading {
            continue;
        }

        let eye_pos = shooter_pos.0 + Vec3::Y * config.eye_height;
        let aim_dir = facing3d.direction();
        let range = active_weapon.range;
        let damage = active_weapon.damage;

        fired.push((shooter_entity, eye_pos, aim_dir, range));

        // Movement inaccuracy: speed shrinks effective hitbox (CS-style)
        let speed_xz = (shooter_vel.0.x * shooter_vel.0.x + shooter_vel.0.z * shooter_vel.0.z).sqrt();
        let speed_frac = (speed_xz / config.max_speed).min(1.0);
        let accuracy_mult = 1.0 - speed_frac * 0.6;

        // Penalty for shooting while moving, scaled by speed fraction
        if config.reward_moving_shot != 0.0 && speed_frac > 0.1 {
            let penalty = config.reward_moving_shot * speed_frac;
            rewards.add(shooter_entity, penalty);
            breakdown.add(shooter_entity, "moving_shot", penalty);
        }
        let effective_hitbox = config.hitbox_radius * accuracy_mult;

        let mut best_hit: Option<(Entity, f32, bool)> = None;

        for (target_entity, target_pos, _, target_team, _target_ph, _) in &targets {
            if target_entity == shooter_entity { continue; }
            let is_friendly = target_team.0 == shooter_team.0;
            let to_target = target_pos.0 + Vec3::Y * config.eye_height * 0.5 - eye_pos;
            let dist = to_target.length();
            if dist > range { continue; }
            let proj = to_target.dot(aim_dir);
            if proj < 0.0 { continue; }
            let perp = (to_target - aim_dir * proj).length();
            if perp > effective_hitbox { continue; }

            let target_center = target_pos.0 + Vec3::Y * config.eye_height * 0.5;
            let occluded = physics3d.obstacles_block_los(eye_pos, target_center);

            if !occluded && (best_hit.is_none() || proj < best_hit.as_ref().unwrap().1) {
                best_hit = Some((target_entity, proj, is_friendly));
            }
        }

        let origin_2d = Vec2::new(shooter_pos.0.x, shooter_pos.0.z);
        let dir_2d = Vec2::new(aim_dir.x, aim_dir.z).normalize_or_zero();
        telemetry.push(TelemetryEvent::ShotFired {
            tick: tick.tick,
            shooter: shooter_entity.to_bits(),
            origin: origin_2d,
            direction: dir_2d,
            hit_target: best_hit.as_ref().map(|(e, _, _)| e.to_bits()),
        });
        shot_events.push(shooter_entity, origin_2d);

        if let Some((hit_entity, _, friendly)) = best_hit {
            hits.push((hit_entity, damage, shooter_entity, friendly));
        }
    }

    let fired_entities: Vec<Entity> = fired.iter().map(|(e, _, _, _)| *e).collect();
    for (entity, _, mut facing3d, _, _, _, mut inventory) in &mut shooters {
        if !fired_entities.contains(&entity) { continue; }
        facing3d.pitch = (facing3d.pitch + 0.04).min(config.pitch_limit);
        if let Some(slot) = inventory.active_weapon_mut() {
            slot.cooldown_remaining = slot.fire_rate;
            if slot.ammo > 0 { slot.ammo -= 1; }
            if slot.ammo == 0 {
                slot.is_reloading = true;
                slot.reload_remaining = slot.reload_time;
            }
        }
    }

    for &(hit_entity, damage, shooter_entity, is_friendly) in &hits {
        if let Ok((_, _, mut health, _, _, armor)) = targets.get_mut(hit_entity) {
            let armor_mult = if armor.is_some_and(|a| a.has_armor) { 0.5 } else { 1.0 };
            let ff_mult = if is_friendly { 0.33 } else { 1.0 };
            let actual_damage = damage * armor_mult * ff_mult;
            health.current -= actual_damage;
            telemetry.push(TelemetryEvent::Damage {
                tick: tick.tick,
                source: shooter_entity.to_bits(),
                target: hit_entity.to_bits(),
                amount: actual_damage,
            });
            if is_friendly {
                rewards.add(shooter_entity, config.reward_friendly_fire);
                breakdown.add(shooter_entity, "friendly_fire", config.reward_friendly_fire);
            } else {
                rewards.add(shooter_entity, config.reward_damage_dealt * actual_damage / health.max);
                breakdown.add(shooter_entity, "damage_dealt", config.reward_damage_dealt * actual_damage / health.max);
            }
            // damage_taken penalty (defaults to 0.0, configurable via JSON)
            rewards.add(hit_entity, config.reward_damage_taken * actual_damage / health.max);
            breakdown.add(hit_entity, "damage_taken", config.reward_damage_taken * actual_damage / health.max);
        }
        commands.entity(hit_entity).insert(LastDamageSource(shooter_entity));
    }

    let near_miss_radius = config.hitbox_radius * 3.0;
    for &(shooter_entity, eye_pos, aim_dir, range) in &fired {
        let hit_anyone = hits.iter().any(|&(_, _, s, _)| s == shooter_entity);
        if hit_anyone { continue; }

        for (target_entity, target_pos, _, _, _, _) in &targets {
            if target_entity == shooter_entity { continue; }
            let to_target = target_pos.0 + Vec3::Y * config.eye_height * 0.5 - eye_pos;
            let dist = to_target.length();
            if dist > range { continue; }
            let proj = to_target.dot(aim_dir);
            if proj < 0.0 { continue; }
            let perp = (to_target - aim_dir * proj).length();
            if perp <= near_miss_radius {
                rewards.add(shooter_entity, config.reward_near_miss);
                breakdown.add(shooter_entity, "near_miss", config.reward_near_miss);
                break;
            }
        }
    }
}

#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn cs_bomb_system(
    mut commands: Commands,
    agents: Query<(Entity, &Position3D, &Team, Option<&BombCarrier>, Option<&mut PlantDefuseProgress>), Without<Dead>>,
    raw_buffer: Res<RawActionBuffer>,
    action_space: Res<ActionSpaceDef>,
    mut bomb: ResMut<BombState>,
    bomb_sites: Res<BombSites>,
    tick: Res<TickState>,
    round: Res<CsRoundState>,
    config: Res<CsLiteConfig>,
    mut rewards: ResMut<RewardBuffer>,
    mut breakdown: ResMut<RewardBreakdownBuffer>,
    mut telemetry: ResMut<TelemetryBuffer>,
) {
    if round.phase != RoundPhase::Active { return; }
    if bomb.detonated || bomb.defused { return; }

    let plant_time = 3.0;
    let defuse_time = 5.0;
    let bomb_timer_secs = 40.0;

    // Check detonation
    if bomb.planted {
        if let Some(pt) = bomb.plant_tick {
            let elapsed = (tick.tick - pt) as f32 * tick.delta;
            if elapsed >= bomb_timer_secs {
                bomb.detonated = true;
                return;
            }
        }
    }

    // Bomb pickup: T-side agent near dropped bomb gets it
    if !bomb.planted && bomb.dropped_position.is_some() {
        let dropped = bomb.dropped_position.unwrap();
        for (entity, pos3, team, carrier, _progress) in &agents {
            if team.0 != 0 || carrier.is_some() { continue; }
            if (pos3.0 - dropped).length() < 3.0 {
                commands.entity(entity).insert(BombCarrier);
                bomb.carrier = Some(entity);
                bomb.dropped_position = None;
                rewards.add(entity, config.reward_bomb_pickup);
                breakdown.add(entity, "bomb_pickup", config.reward_bomb_pickup);
                break;
            }
        }
    }

    for (entity, pos3, team, carrier, _progress) in &agents {
        let Some(raw) = raw_buffer.get(entity) else { continue };
        if raw.len() < action_space.total_size { continue; }
        if action_space.heads.len() <= 3 { continue; }

        let bomb_action = action_space.extract_head(raw, 3)[0].round() as u8;

        // Plant (T-side, carrying bomb, in site, action=1)
        if bomb_action == 1 && team.0 == 0 && carrier.is_some() && !bomb.planted {
            let dist_a = (pos3.0 - bomb_sites.site_a_center).length();
            let dist_b = (pos3.0 - bomb_sites.site_b_center).length();
            let in_site_a = dist_a < bomb_sites.site_a_radius;
            let in_site_b = dist_b < bomb_sites.site_b_radius;

            if in_site_a || in_site_b {
                bomb.planted = true;
                bomb.plant_position = Some(pos3.0);
                bomb.plant_tick = Some(tick.tick);
                bomb.plant_site = if in_site_a { Some(0) } else { Some(1) };
                bomb.carrier = None;
                bomb.dropped_position = None;
                commands.entity(entity).remove::<BombCarrier>();
                rewards.add(entity, config.reward_bomb_plant);
                breakdown.add(entity, "bomb_plant", config.reward_bomb_plant);
                telemetry.push(TelemetryEvent::Spawn {
                    tick: tick.tick,
                    entity: entity.to_bits(),
                    position: Vec2::new(pos3.0.x, pos3.0.z),
                    team: team.0,
                });
            }
        }

        // Defuse (CT-side, bomb planted, near bomb, action=2)
        if bomb_action == 2 && team.0 == 1 && bomb.planted {
            if let Some(bp) = bomb.plant_position {
                if (pos3.0 - bp).length() < bomb_sites.site_a_radius {
                    bomb.defused = true;
                    rewards.add(entity, config.reward_bomb_defuse);
                    breakdown.add(entity, "bomb_defuse", config.reward_bomb_defuse);
                }
            }
        }
    }
}

#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub fn cs_death_system(
    mut commands: Commands,
    query: Query<(Entity, &Health, &Team, &Position3D, Option<&LastDamageSource>, Option<&BombCarrier>), (With<Agent>, Without<Dead>)>,
    tick: Res<TickState>,
    mut telemetry: ResMut<TelemetryBuffer>,
    mut rewards: ResMut<RewardBuffer>,
    mut breakdown: ResMut<RewardBreakdownBuffer>,
    mut round: ResMut<CsRoundState>,
    mut bomb: ResMut<BombState>,
    config: Res<CsLiteConfig>,
) {
    if round.phase != RoundPhase::Active { return; }

    for (entity, health, team, pos3, last_damage, carrier) in &query {
        if health.current > 0.0 { continue; }

        commands.entity(entity).insert(Dead);

        let killer = last_damage.map(|lds| lds.0.to_bits()).unwrap_or(0);
        telemetry.push(TelemetryEvent::Kill {
            tick: tick.tick,
            killer,
            victim: entity.to_bits(),
        });

        if let Some(lds) = last_damage {
            rewards.add(lds.0, config.reward_kill);
            breakdown.add(lds.0, "kill", config.reward_kill);
        }
        // death penalty (defaults to 0.0, configurable via JSON)
        rewards.add(entity, config.reward_death);
        breakdown.add(entity, "death", config.reward_death);

        // Drop bomb if carrier dies
        if carrier.is_some() && !bomb.planted {
            bomb.dropped_position = Some(pos3.0);
            bomb.carrier = None;
            commands.entity(entity).remove::<BombCarrier>();
        }

        match team.0 {
            0 => round.t_alive = round.t_alive.saturating_sub(1),
            1 => round.ct_alive = round.ct_alive.saturating_sub(1),
            _ => {}
        }
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn cs_round_state_system(
    mut commands: Commands,
    mut agents: Query<
        (Entity, &mut Health, &mut Position3D, &mut Velocity3D, &mut Facing3D,
         &PhysicsHandle3D, &Team, &mut Inventory, &Agent),
        With<Agent>,
    >,
    tick: Res<TickState>,
    mut round: ResMut<CsRoundState>,
    config: Res<CsLiteConfig>,
    spawns: Res<CsSpawnPoints>,
    mut physics3d: ResMut<Physics3DState>,
    mut rewards: ResMut<RewardBuffer>,
    mut breakdown: ResMut<RewardBreakdownBuffer>,
    mut telemetry: ResMut<TelemetryBuffer>,
    mut bomb: ResMut<BombState>,
) {
    let dt = tick.delta;
    round.phase_timer -= dt;

    match round.phase {
        RoundPhase::BuyFreeze => {
            if round.phase_timer <= 0.0 {
                round.phase = RoundPhase::Active;
                round.phase_timer = round.round_time_limit;
            }
        }
        RoundPhase::Active => {
            let bomb_planted = bomb.planted;
            let bomb_detonated = bomb.detonated;
            let bomb_defused = bomb.defused;

            let winner = if bomb_detonated {
                Some(CsSide::Terrorist)
            } else if bomb_defused {
                Some(CsSide::CounterTerrorist)
            } else if round.t_alive == 0 && !bomb_planted {
                Some(CsSide::CounterTerrorist)
            } else if round.t_alive == 0 && bomb_planted {
                // Post-plant: CTs must defuse, don't auto-win
                None
            } else if round.ct_alive == 0 {
                Some(CsSide::Terrorist)
            } else if round.phase_timer <= 0.0 && !bomb_planted {
                Some(CsSide::CounterTerrorist)
            } else {
                None
            };

            if let Some(side) = winner {
                round.phase = RoundPhase::RoundEnd;
                round.phase_timer = round.end_time;
                round.round_winner = Some(side);

                match side {
                    CsSide::Terrorist => round.t_score += 1,
                    CsSide::CounterTerrorist => round.ct_score += 1,
                }

                let registry_agents: Vec<(Entity, u8)> = agents.iter()
                    .map(|(e, _, _, _, _, _, t, _, _)| (e, t.0))
                    .collect();
                let winning_team = match side {
                    CsSide::Terrorist => 0u8,
                    CsSide::CounterTerrorist => 1u8,
                };
                for (entity, team) in &registry_agents {
                    if *team == winning_team {
                        rewards.add(*entity, config.reward_round_win);
                        breakdown.add(*entity, "round_win", config.reward_round_win);
                    } else {
                        // round_loss penalty (defaults to 0.0, configurable via JSON)
                        rewards.add(*entity, config.reward_round_loss);
                        breakdown.add(*entity, "round_loss", config.reward_round_loss);
                    }
                }
            }
        }
        RoundPhase::RoundEnd => {
            if round.phase_timer <= 0.0 {
                // Reset for next round (or new match if match is over)
                if round.match_over() {
                    round.t_score = 0;
                    round.ct_score = 0;
                    round.round_number = 1;
                } else {
                    round.round_number += 1;
                }
                round.phase = RoundPhase::BuyFreeze;
                round.phase_timer = round.buy_time;
                round.t_alive = config.players_per_team;
                round.ct_alive = config.players_per_team;
                round.round_winner = None;

                // Collect agents sorted by source_id for deterministic bomb assignment
                let mut sorted_agents: Vec<(Entity, u8, u32)> = agents.iter()
                    .map(|(e, _, _, _, _, _, t, _, a)| (e, t.0, a.source_id))
                    .collect();
                sorted_agents.sort_by_key(|(_, _, sid)| *sid);

                // Assign bomb to first T-side agent
                let mut bomb_assigned = false;
                for (entity, team_id, _) in &sorted_agents {
                    if *team_id == 0 && !bomb_assigned {
                        commands.entity(*entity).insert(BombCarrier);
                        bomb_assigned = true;
                    }
                }

                // Respawn all agents
                let mut rng = rand::rng();
                for (entity, mut health, mut pos3, mut vel3, mut facing3d, ph3d, team, mut inv, _) in &mut agents {
                    commands.entity(entity).remove::<Dead>();
                    commands.entity(entity).remove::<LastDamageSource>();
                    commands.entity(entity).remove::<Respawning>();

                    health.current = health.max;
                    vel3.0 = Vec3::ZERO;

                    let spawn_list = if team.0 == 0 { &spawns.t_spawns } else { &spawns.ct_spawns };
                    let idx = rng.random_range(0..spawn_list.len());
                    let spawn = spawn_list[idx];
                    let jitter = Vec3::new(
                        rng.random_range(-0.5f32..0.5),
                        0.0,
                        rng.random_range(-0.5f32..0.5),
                    );
                    let new_pos = spawn + jitter;
                    pos3.0 = new_pos;
                    physics3d.teleport_body(ph3d.body, new_pos);

                    facing3d.yaw = if team.0 == 0 {
                        std::f32::consts::FRAC_PI_2
                    } else {
                        -std::f32::consts::FRAC_PI_2
                    };
                    facing3d.pitch = 0.0;

                    inv.weapons = vec![make_pistol()];
                    inv.active = 0;

                    telemetry.push(TelemetryEvent::Spawn {
                        tick: tick.tick,
                        entity: entity.to_bits(),
                        position: Vec2::new(new_pos.x, new_pos.z),
                        team: team.0,
                    });

                    commands.entity(entity).remove::<BombCarrier>();
                    commands.entity(entity).remove::<PlantDefuseProgress>();
                    commands.entity(entity).remove::<PathState>();
                    world_entity_reset_path_state(&mut commands, entity);
                }

                // Reset bomb state
                *bomb = BombState::default();
            }
        }
    }
}

fn world_entity_reset_path_state(commands: &mut Commands, entity: Entity) {
    commands.entity(entity).insert(PathState::default());
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn cs_telemetry_system(
    agents: Query<
        (Entity, &Position3D, &Velocity3D, &Facing3D, &Team, &Inventory, &PhysicsHandle3D, &Health, Option<&Dead>),
        With<Agent>,
    >,
    tick: Res<TickState>,
    config: Res<CsLiteConfig>,
    physics3d: Res<Physics3DState>,
    raw_buffer: Res<RawActionBuffer>,
    action_space: Res<ActionSpaceDef>,
    round: Res<CsRoundState>,
    mut telemetry: ResMut<TelemetryBuffer>,
) {
    if !tick.tick.is_multiple_of(4) { return; }

    telemetry.push(TelemetryEvent::CsLiteRoundState {
        tick: tick.tick,
        phase: match round.phase {
            RoundPhase::BuyFreeze => "buy_freeze".into(),
            RoundPhase::Active => "active".into(),
            RoundPhase::RoundEnd => "round_end".into(),
        },
        round_number: round.round_number,
        t_score: round.t_score,
        ct_score: round.ct_score,
        phase_timer: round.phase_timer,
        t_alive: round.t_alive,
        ct_alive: round.ct_alive,
    });

    for (entity, pos3, vel3, facing3d, team, inventory, ph3d, health, dead) in &agents {
        let is_dead = dead.is_some();
        let mut shooting = false;
        let mut move_dir: u8 = 8;

        if !is_dead {
            if let Some(raw) = raw_buffer.get(entity)
                && raw.len() >= action_space.total_size
            {
                move_dir = (action_space.extract_head(raw, 0)[0].round() as u8).min(11);
                let shoot_val = action_space.extract_head(raw, 1)[0];
                shooting = shoot_val > 0.5;
            }
        }

        let total_rays = config.ray_h_count * config.ray_v_count;
        let mut ray_distances;
        let mut ray_hit_types;

        if is_dead {
            ray_distances = vec![config.ray_max_range; total_rays];
            ray_hit_types = vec![0.0; total_rays];
        } else {
            ray_distances = Vec::with_capacity(total_rays);
            ray_hit_types = Vec::with_capacity(total_rays);
            let eye_pos = pos3.0 + Vec3::Y * config.eye_height;
            let h_center = (config.ray_h_count as f32 - 1.0) / 2.0;
            let v_center = (config.ray_v_count as f32 - 1.0) / 2.0;

            for vi in 0..config.ray_v_count {
                let pitch_offset = if config.ray_v_count > 1 {
                    config.ray_v_fov * (vi as f32 - v_center) / (config.ray_v_count as f32 - 1.0)
                } else {
                    0.0
                };
                for hi in 0..config.ray_h_count {
                    let yaw_offset = if config.ray_h_count > 1 {
                        config.ray_h_fov * (hi as f32 - h_center) / (config.ray_h_count as f32 - 1.0)
                    } else {
                        0.0
                    };
                    let ray_yaw = facing3d.yaw + yaw_offset;
                    let ray_pitch = (facing3d.pitch + pitch_offset).clamp(-config.pitch_limit, config.pitch_limit);
                    let ray_dir = Vec3::new(
                        ray_yaw.cos() * ray_pitch.cos(),
                        ray_pitch.sin(),
                        ray_yaw.sin() * ray_pitch.cos(),
                    );
                    let (dist_norm, hit_type) = physics3d.cast_ray_classified(
                        eye_pos, ray_dir, config.ray_max_range, Some(ph3d.collider), team.0,
                    );
                    ray_distances.push(dist_norm * config.ray_max_range);
                    ray_hit_types.push(hit_type);
                }
            }
        }

        telemetry.push(TelemetryEvent::Arena3DState {
            tick: tick.tick,
            entity: entity.to_bits(),
            position: [pos3.0.x, pos3.0.y, pos3.0.z],
            velocity: [vel3.0.x, vel3.0.y, vel3.0.z],
            yaw: facing3d.yaw,
            pitch: facing3d.pitch,
            health: health.current,
            max_health: health.max,
            team: team.0,
            is_dead,
            active_weapon: inventory.active as u8,
            shooting,
            move_direction: move_dir,
            ray_distances,
            ray_hit_types,
        });
    }
}

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

// --- Compass constants for tactical AI ---

const D: f32 = std::f32::consts::FRAC_1_SQRT_2;

const COMPASS_DIRS: [Vec2; 8] = [
    Vec2::new(0.0, 1.0),  // 0: N
    Vec2::new(D, D),       // 1: NE
    Vec2::new(1.0, 0.0),  // 2: E
    Vec2::new(D, -D),     // 3: SE
    Vec2::new(0.0, -1.0), // 4: S
    Vec2::new(-D, -D),    // 5: SW
    Vec2::new(-1.0, 0.0), // 6: W
    Vec2::new(-D, D),     // 7: NW
];

/// Maps compass index (N=0, NE=1, E=2, ...) to wall_ray index.
/// Wall rays are cast at angle `i * PI/4` starting from east (0 rad),
/// so wall_rays[0]=E, [1]=NE, [2]=N, [3]=NW, [4]=W, [5]=SW, [6]=S, [7]=SE.
const COMPASS_TO_WALL_RAY: [usize; 8] = [
    2, // N  -> wall_ray[2] (PI/2)
    1, // NE -> wall_ray[1] (PI/4)
    0, // E  -> wall_ray[0] (0)
    7, // SE -> wall_ray[7] (7PI/4)
    6, // S  -> wall_ray[6] (3PI/2)
    5, // SW -> wall_ray[5] (5PI/4)
    4, // W  -> wall_ray[4] (PI)
    3, // NW -> wall_ray[3] (3PI/4)
];

/// Pick the compass direction index (0-7) whose vector best aligns with `desired_dir`.
/// Returns the index as `f32` for direct use in the action vector.
pub fn best_compass_index(desired_dir: Vec2) -> f32 {
    let mut best_idx = 0usize;
    let mut best_dot = f32::MIN;
    for (i, dir) in COMPASS_DIRS.iter().enumerate() {
        let dot = dir.dot(desired_dir);
        if dot > best_dot {
            best_dot = dot;
            best_idx = i;
        }
    }
    best_idx as f32
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

// --- Tactical AI ---
// Action layout (Tactical): [move_target, aim_delta, shoot]
// move_target: 0-7 compass, 8=stay, 9=cover, 10=advance, 11=retreat

/// Score each compass direction (0-7) with wall avoidance and return the best index as f32.
/// Uses context steering: `interest - wall_danger * 3.0`.
/// Correctly maps compass indices to their corresponding wall_ray indices.
fn best_compass_with_walls(desired_dir: Vec2, wall_rays: &[f32; 8]) -> f32 {
    let mut best_idx = 0usize;
    let mut best_score = f32::MIN;
    for i in 0..8 {
        let interest = COMPASS_DIRS[i].dot(desired_dir);
        let ray_idx = COMPASS_TO_WALL_RAY[i];
        let danger = (1.0 - wall_rays[ray_idx]).max(0.0);
        let score = interest - danger * 3.0;
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    best_idx as f32
}

/// Tactical aggressive AI that outputs the 4-float tactical action format:
/// `[move_target, aim_delta, shoot, weapon_select]`.
///
/// Behavior:
/// - **Enemies visible**: advance/retreat/strafe based on distance to nearest enemy,
///   aim toward enemy, shoot when in range and weapon ready.
///   Uses rifle (0.0) when enemy > 200 units, shotgun (1.0) when < 200 units.
/// - **No enemies visible**: wander using deterministic compass cycling, aim toward
///   movement direction. Defaults to rifle (0.0).
/// - **Wall avoidance**: context-steered compass selection using `wall_rays`.
pub fn tactical_aggressive_ai() -> AiFn {
    Box::new(|ctx: &AiContext| {
        let nearest = ctx
            .enemies
            .iter()
            .min_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        if let Some(enemy) = nearest {
            let dist = enemy.distance;
            let to_enemy = (enemy.position - ctx.my_position).normalize_or_zero();
            let angle = to_enemy.y.atan2(to_enemy.x);

            // Pick movement strategy based on distance
            let move_target = if dist > ctx.weapon_range * 0.6 {
                // Too far -- advance toward enemy
                best_compass_with_walls(to_enemy, &ctx.wall_rays)
            } else if dist < ctx.weapon_range * 0.2 {
                // Too close -- retreat away from enemy
                best_compass_with_walls(-to_enemy, &ctx.wall_rays)
            } else {
                // In engagement range -- strafe perpendicular
                let perp = Vec2::new(-to_enemy.y, to_enemy.x);
                let phase = (ctx.tick / 90 + ctx.entity_bits) % 3;
                let strafe_sign = if phase == 0 {
                    1.0
                } else if phase == 1 {
                    -1.0
                } else {
                    0.3
                };
                let strafe_dir = (perp * strafe_sign + to_enemy * 0.2).normalize_or_zero();
                best_compass_with_walls(strafe_dir, &ctx.wall_rays)
            };

            let aim_delta = angle_to_turn_delta(ctx.my_facing, angle);

            let shoot = if dist <= ctx.weapon_range && ctx.weapon_cooldown <= 0.0 {
                1.0
            } else {
                0.0
            };

            // Weapon select: shotgun (1.0) when close, rifle (0.0) when far
            let weapon_select = if dist < 200.0 { 1.0 } else { 0.0 };

            vec![move_target, aim_delta, shoot, weapon_select]
        } else {
            // No enemies visible -- wander deterministically
            let epoch = ctx.tick / 128;
            let hash = (epoch.wrapping_mul(2654435761) ^ ctx.entity_bits) as f32;
            let wander_idx = (hash.abs() as usize) % 8;
            let wander_dir = COMPASS_DIRS[wander_idx];

            let move_target = best_compass_with_walls(wander_dir, &ctx.wall_rays);

            // Aim toward movement direction
            let move_dir = COMPASS_DIRS[move_target as usize];
            let desired_angle = move_dir.y.atan2(move_dir.x);
            let aim_delta = angle_to_turn_delta(ctx.my_facing, desired_angle);

            // Default to rifle when no enemies
            vec![move_target, aim_delta, 0.0, 0.0]
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_context() -> AiContext {
        AiContext {
            my_position: Vec2::new(300.0, 300.0),
            my_velocity: Vec2::ZERO,
            my_team: 0,
            my_facing: 0.0, // facing east
            my_health: 100.0,
            weapon_range: 400.0,
            weapon_cooldown: 0.0,
            tick: 100,
            entity_bits: 42,
            arena_width: 600.0,
            arena_height: 600.0,
            enemies: Vec::new(),
            allies: Vec::new(),
            wall_rays: [1.0; 8], // no walls nearby
        }
    }

    // -------------------------------------------------------------------
    // best_compass_index tests
    // -------------------------------------------------------------------
    #[test]
    fn best_compass_index_north() {
        assert_eq!(best_compass_index(Vec2::new(0.0, 1.0)), 0.0);
    }

    #[test]
    fn best_compass_index_east() {
        assert_eq!(best_compass_index(Vec2::new(1.0, 0.0)), 2.0);
    }

    #[test]
    fn best_compass_index_south() {
        assert_eq!(best_compass_index(Vec2::new(0.0, -1.0)), 4.0);
    }

    #[test]
    fn best_compass_index_west() {
        assert_eq!(best_compass_index(Vec2::new(-1.0, 0.0)), 6.0);
    }

    #[test]
    fn best_compass_index_northeast() {
        assert_eq!(best_compass_index(Vec2::new(0.7, 0.7)), 1.0);
    }

    #[test]
    fn best_compass_index_southwest() {
        assert_eq!(best_compass_index(Vec2::new(-0.7, -0.7)), 5.0);
    }

    // -------------------------------------------------------------------
    // tactical_aggressive_ai output format
    // -------------------------------------------------------------------
    #[test]
    fn tactical_ai_returns_4_element_vec() {
        let ai = tactical_aggressive_ai();
        let ctx = default_context();
        let actions = ai(&ctx);
        assert_eq!(
            actions.len(),
            4,
            "tactical AI must return exactly 4 values, got {}",
            actions.len()
        );
    }

    #[test]
    fn tactical_ai_no_enemies_shoot_is_zero() {
        let ai = tactical_aggressive_ai();
        let ctx = default_context();
        let actions = ai(&ctx);
        assert_eq!(
            actions[2], 0.0,
            "shoot should be 0.0 when no enemies visible"
        );
    }

    #[test]
    fn tactical_ai_no_enemies_move_target_is_compass() {
        let ai = tactical_aggressive_ai();
        let ctx = default_context();
        let actions = ai(&ctx);
        let mt = actions[0];
        assert!(
            mt >= 0.0 && mt <= 7.0 && mt == mt.floor(),
            "move_target with no enemies should be an integer 0-7, got {}",
            mt
        );
    }

    #[test]
    fn tactical_ai_enemy_in_range_shoots() {
        let ai = tactical_aggressive_ai();
        let mut ctx = default_context();
        // Place enemy to the east at distance 200 (within weapon_range=400)
        ctx.enemies.push(TargetInfo {
            position: Vec2::new(500.0, 300.0),
            distance: 200.0,
            health: 80.0,
        });
        ctx.weapon_cooldown = 0.0;

        let actions = ai(&ctx);
        assert_eq!(
            actions[2], 1.0,
            "shoot should be 1.0 when enemy in range and weapon ready"
        );
    }

    #[test]
    fn tactical_ai_enemy_in_range_but_weapon_cooling_no_shoot() {
        let ai = tactical_aggressive_ai();
        let mut ctx = default_context();
        ctx.enemies.push(TargetInfo {
            position: Vec2::new(500.0, 300.0),
            distance: 200.0,
            health: 80.0,
        });
        ctx.weapon_cooldown = 0.5; // weapon still cooling

        let actions = ai(&ctx);
        assert_eq!(
            actions[2], 0.0,
            "shoot should be 0.0 when weapon is cooling down"
        );
    }

    #[test]
    fn tactical_ai_enemy_out_of_range_no_shoot() {
        let ai = tactical_aggressive_ai();
        let mut ctx = default_context();
        // Enemy at 500, weapon_range=400 => out of range
        ctx.enemies.push(TargetInfo {
            position: Vec2::new(800.0, 300.0),
            distance: 500.0,
            health: 100.0,
        });
        ctx.weapon_cooldown = 0.0;

        let actions = ai(&ctx);
        assert_eq!(
            actions[2], 0.0,
            "shoot should be 0.0 when enemy is beyond weapon range"
        );
    }

    #[test]
    fn tactical_ai_move_target_is_valid_range() {
        let ai = tactical_aggressive_ai();
        let mut ctx = default_context();
        ctx.enemies.push(TargetInfo {
            position: Vec2::new(500.0, 300.0),
            distance: 200.0,
            health: 80.0,
        });

        let actions = ai(&ctx);
        let mt = actions[0];
        assert!(
            mt >= 0.0 && mt <= 11.0,
            "move_target should be in [0, 11], got {}",
            mt
        );
    }

    #[test]
    fn tactical_ai_aim_delta_in_range() {
        let ai = tactical_aggressive_ai();
        let mut ctx = default_context();
        ctx.enemies.push(TargetInfo {
            position: Vec2::new(300.0, 500.0),
            distance: 200.0,
            health: 80.0,
        });
        ctx.my_facing = 0.5; // facing slightly north-east

        let actions = ai(&ctx);
        let aim = actions[1];
        assert!(
            aim >= -1.0 && aim <= 1.0,
            "aim_delta should be in [-1, 1], got {}",
            aim
        );
    }

    #[test]
    fn tactical_ai_advances_when_far() {
        let ai = tactical_aggressive_ai();
        let mut ctx = default_context();
        ctx.my_position = Vec2::new(100.0, 300.0);
        // Enemy far to the east at distance > weapon_range*0.6 = 240
        ctx.enemies.push(TargetInfo {
            position: Vec2::new(500.0, 300.0),
            distance: 400.0,
            health: 100.0,
        });

        let actions = ai(&ctx);
        let mt = actions[0] as usize;
        // Should pick compass direction aligned with east (index 2)
        // or close to it (1 or 3)
        assert!(
            mt <= 7,
            "move_target when advancing should be a compass direction 0-7, got {}",
            mt
        );
        // The direction to enemy is east, so compass index 2 (E) should be picked
        assert_eq!(mt, 2, "should advance east toward enemy at (500, 300)");
    }

    #[test]
    fn tactical_ai_retreats_when_close() {
        let ai = tactical_aggressive_ai();
        let mut ctx = default_context();
        ctx.my_position = Vec2::new(300.0, 300.0);
        // Enemy very close, within weapon_range*0.2 = 80
        ctx.enemies.push(TargetInfo {
            position: Vec2::new(350.0, 300.0),
            distance: 50.0,
            health: 100.0,
        });

        let actions = ai(&ctx);
        let mt = actions[0] as usize;
        // Should retreat west (index 6) -- opposite of east
        assert!(mt <= 7, "move_target when retreating should be compass 0-7");
        assert_eq!(mt, 6, "should retreat west away from nearby enemy");
    }

    #[test]
    fn tactical_ai_wall_avoidance_changes_direction() {
        let ai = tactical_aggressive_ai();
        let mut ctx = default_context();
        ctx.my_position = Vec2::new(100.0, 300.0);
        // Enemy to the east (far)
        ctx.enemies.push(TargetInfo {
            position: Vec2::new(500.0, 300.0),
            distance: 400.0,
            health: 100.0,
        });

        // Without wall, should advance east (compass 2)
        let actions_no_wall = ai(&ctx);
        assert_eq!(
            actions_no_wall[0] as usize, 2,
            "without wall should advance east"
        );

        // wall_rays[0] is the east ray (angle = 0 * PI/4 = 0 rad = east)
        // COMPASS_TO_WALL_RAY[2] (compass E) maps to wall_rays[0]
        ctx.wall_rays[0] = 0.05; // wall very close to east
        let actions_with_wall = ai(&ctx);

        assert_ne!(
            actions_with_wall[0] as usize, 2,
            "with wall east, should avoid compass east (index 2)"
        );
        assert!(
            actions_with_wall[0] >= 0.0 && actions_with_wall[0] <= 7.0,
            "move_target should still be valid compass direction"
        );
    }

    // -------------------------------------------------------------------
    // best_compass_with_walls tests
    // -------------------------------------------------------------------
    #[test]
    fn best_compass_with_walls_no_walls_matches_simple() {
        let no_walls = [1.0f32; 8];
        let dir = Vec2::new(1.0, 0.0); // east
        let result = best_compass_with_walls(dir, &no_walls);
        assert_eq!(result, best_compass_index(dir));
    }

    #[test]
    fn best_compass_with_walls_avoids_walled_direction() {
        let mut wall_rays = [1.0f32; 8];
        // Block the north direction: compass N = index 0
        // COMPASS_TO_WALL_RAY[0] = wall_rays[2] (PI/2 = north)
        wall_rays[2] = 0.05; // very close wall to the north

        // Desired direction is north
        let dir = Vec2::new(0.0, 1.0);
        let result = best_compass_with_walls(dir, &wall_rays);

        // Should NOT pick 0 (north) because wall is too close there
        assert_ne!(
            result, 0.0,
            "should avoid north (compass index 0) due to close wall"
        );
    }
}


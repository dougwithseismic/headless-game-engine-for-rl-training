use bevy_ecs::prelude::*;
use glam::Vec2;

use crate::ecs::components::*;
use crate::ecs::resources::*;
use crate::navigation::NavGrid;
use crate::observation::AgentRegistry;
use crate::physics::PhysicsState;
use crate::scenarios::tactical_deathmatch::TacticalConfig;

const COMPASS_DIRS: [(f32, f32); 8] = [
    (0.0, 1.0),
    (0.707, 0.707),
    (1.0, 0.0),
    (0.707, -0.707),
    (0.0, -1.0),
    (-0.707, -0.707),
    (-1.0, 0.0),
    (-0.707, 0.707),
];

fn find_nearest_cover(
    agent_pos: Vec2,
    enemy_pos: Vec2,
    nav: &NavGrid,
    physics: &PhysicsState,
    search_radius: f32,
) -> Vec2 {
    let steps = (search_radius / nav.cell_size) as usize;
    let mut best_pos = agent_pos;
    let mut best_dist = f32::MAX;

    for ring in 1..=steps {
        for i in 0..(ring * 8) {
            let angle = i as f32 * std::f32::consts::TAU / (ring * 8) as f32;
            let offset = Vec2::new(angle.cos(), angle.sin()) * (ring as f32 * nav.cell_size);
            let candidate = agent_pos + offset;

            if !nav.is_walkable(candidate) {
                continue;
            }

            let has_wall_between = !physics.has_line_of_sight(candidate, enemy_pos, None);

            if has_wall_between {
                let d = agent_pos.distance(candidate);
                if d < best_dist {
                    best_dist = d;
                    best_pos = candidate;
                }
            }
        }
    }

    best_pos
}

fn find_advance_position(
    agent_pos: Vec2,
    enemy_pos: Vec2,
    nav: &NavGrid,
) -> Vec2 {
    let to_enemy = (enemy_pos - agent_pos).normalize_or_zero();
    let steps = 5;
    for i in 1..=steps {
        let candidate = agent_pos + to_enemy * (i as f32 * nav.cell_size);
        if nav.is_walkable(candidate) {
            return candidate;
        }
    }
    agent_pos
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn compute_candidates_system(
    agents: Query<(Entity, &Position, &Team, &PhysicsHandle), (With<Agent>, Without<Dead>)>,
    mut candidate_buffer: ResMut<CandidatePositionBuffer>,
    nav: Res<NavGrid>,
    physics: Res<PhysicsState>,
    bounds: Res<WorldBounds>,
    config: Res<TacticalConfig>,
    registry: Res<AgentRegistry>,
    all_agents: Query<(Entity, &Position, &Team, &PhysicsHandle), Without<Dead>>,
) {
    let arena_diag = bounds.diagonal();
    candidate_buffer.clear();

    for (entity, pos, team, _ph) in &agents {
        let agent_pos = pos.0;
        let dist = config.candidate_distance;

        // Find primary enemy
        let mut primary_enemy: Option<(Entity, Vec2)> = None;
        let mut all_enemies: Vec<(Entity, Vec2)> = Vec::new();
        for &e in &registry.agents {
            if e == entity {
                continue;
            }
            if let Ok((_, e_pos, e_team, _)) = all_agents.get(e)
                && e_team.0 != team.0
            {
                all_enemies.push((e, e_pos.0));
                let d = agent_pos.distance(e_pos.0);
                if primary_enemy.is_none() || d < agent_pos.distance(primary_enemy.unwrap().1) {
                    primary_enemy = Some((e, e_pos.0));
                }
            }
        }

        let enemy_pos = primary_enemy.map(|(_, p)| p).unwrap_or(agent_pos);
        let mut set = CandidateSet::default();

        // Indices 0-7: compass directions
        for (i, &(dx, dy)) in COMPASS_DIRS.iter().enumerate() {
            let dir = Vec2::new(dx, dy);
            let raw_pos = agent_pos + dir * dist;
            let candidate_pos = if nav.is_walkable(raw_pos) {
                raw_pos
            } else {
                nav.snap_to_walkable(raw_pos, dir)
            };

            set.positions[i] = compute_features(
                candidate_pos,
                agent_pos,
                enemy_pos,
                &all_enemies,
                &nav,
                &physics,
                arena_diag,
            );
        }

        // Index 8: Stay
        set.positions[8] = compute_features(
            agent_pos,
            agent_pos,
            enemy_pos,
            &all_enemies,
            &nav,
            &physics,

            arena_diag,
        );
        set.positions[8].path_distance = 0.0;

        // Index 9: Nearest cover from primary enemy
        let cover_pos = find_nearest_cover(agent_pos, enemy_pos, &nav, &physics, dist * 3.0);
        set.positions[9] = compute_features(
            cover_pos,
            agent_pos,
            enemy_pos,
            &all_enemies,
            &nav,
            &physics,

            arena_diag,
        );

        // Index 10: Advance toward primary enemy
        let advance_pos = find_advance_position(agent_pos, enemy_pos, &nav);
        set.positions[10] = compute_features(
            advance_pos,
            agent_pos,
            enemy_pos,
            &all_enemies,
            &nav,
            &physics,

            arena_diag,
        );

        // Index 11: Retreat (move away from enemy toward map center or spawn)
        let retreat_dir = (agent_pos - enemy_pos).normalize_or_zero();
        let retreat_pos = if retreat_dir.length_squared() > 0.01 {
            let raw = agent_pos + retreat_dir * dist * 2.0;
            let clamped = raw.clamp(
                Vec2::new(30.0, 30.0),
                Vec2::new(bounds.width - 30.0, bounds.height - 30.0),
            );
            if nav.is_walkable(clamped) {
                clamped
            } else {
                nav.snap_to_walkable(clamped, retreat_dir)
            }
        } else {
            // No enemy — retreat to center
            Vec2::new(bounds.width / 2.0, bounds.height / 2.0)
        };
        set.positions[11] = compute_features(
            retreat_pos,
            agent_pos,
            enemy_pos,
            &all_enemies,
            &nav,
            &physics,

            arena_diag,
        );

        candidate_buffer.insert(entity, set);
    }
}

fn compute_features(
    candidate_pos: Vec2,
    agent_pos: Vec2,
    enemy_pos: Vec2,
    all_enemies: &[(Entity, Vec2)],
    nav: &NavGrid,
    physics: &PhysicsState,
    arena_diag: f32,
) -> CandidatePosition {
    let path_dist = nav
        .path_distance(agent_pos, candidate_pos)
        .map(|d| (d / arena_diag).min(1.0))
        .unwrap_or(1.0);

    let has_los = physics.has_line_of_sight(candidate_pos, enemy_pos, None);
    let dist_to_enemy = candidate_pos.distance(enemy_pos) / arena_diag;
    let enemies_with_los = all_enemies
        .iter()
        .filter(|(_, ep)| physics.has_line_of_sight(candidate_pos, *ep, None))
        .count() as f32
        / all_enemies.len().max(1) as f32;

    CandidatePosition {
        world_pos: candidate_pos,
        path_distance: path_dist,
        has_los_to_enemy: has_los,
        dist_to_enemy: dist_to_enemy.min(1.0),
        enemies_with_los,
    }
}

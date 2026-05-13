use bevy_ecs::prelude::*;
use glam::Vec3;

use crate::action_space::RawActionBuffer;
use crate::ecs::components::{Agent, Dead, Team};
use crate::ecs::resources::TickState;

use crate::physics3d::Physics3DState;

use super::cs_lite::{
    AgentGoal, BombCarrier, BombSites, BombState,
    CsLiteConfig, CsRoundState, Facing3D, ObjectiveType, PhysicsHandle3D,
    Position3D, RoundPhase, COMPASS_DIRS,
};

/// A deliberately bad opponent for Phase 1 curriculum training.
///
/// Behavior:
/// - Wanders randomly, changing direction every 30-60 ticks
/// - Turns slowly toward the enemy (50% of max turn rate)
/// - Only shoots when very roughly aligned (wide threshold)
/// - Misses shots by adding random aim jitter
///
/// The point is to give the learning agent a target it can actually
/// kill while still providing some challenge (it shoots back).
#[allow(clippy::too_many_arguments)]
pub fn cs_dummy_ai_system(
    agents: Query<
        (Entity, &Position3D, &Facing3D, &Team, &PhysicsHandle3D, Option<&BombCarrier>, Option<&AgentGoal>),
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
    if round.phase != RoundPhase::Active {
        return;
    }

    let all: Vec<_> = agents
        .iter()
        .map(|(e, p, _f, t, _ph, _, _)| (e, p.0, t.0))
        .collect();

    for (entity, pos3, _facing, team, _ph, carrier, goal) in &agents {
        let pos = pos3.0;
        let team_id = team.0;
        if raw_buffer.get(entity).is_some() {
            continue;
        }

        let eye_pos = pos + Vec3::Y * config.eye_height;
        let agent_hash = entity.to_bits();

        let mut nearest_enemy_pos = None;
        let mut nearest_enemy_dist = f32::MAX;
        let mut visible_enemy_pos = None;
        let mut visible_enemy_dist = f32::MAX;

        for &(other_e, other_pos, other_team) in &all {
            if other_e == entity || other_team == team_id {
                continue;
            }
            let d = pos.distance(other_pos);
            if d < 0.1 {
                continue;
            }

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

        // Shoot at visible enemies
        if visible_enemy_pos.is_some() && (tick.tick + agent_hash) % 3 == 0 {
            shoot = 1.0;
        }

        // Navigate toward goal target for objective-driven goals,
        // otherwise wander/chase enemies
        let has_objective_goal = matches!(
            goal,
            Some(g) if matches!(g.objective, ObjectiveType::PlantBomb | ObjectiveType::DefuseBomb | ObjectiveType::HoldPosition | ObjectiveType::Rotate)
                && g.target_position != Vec3::ZERO
        );

        if has_objective_goal {
            let target = goal.unwrap().target_position;
            let to_target = (target - pos).normalize_or_zero();
            if to_target.length_squared() < 0.01 {
                move_target = 8.0; // stay
            } else {
                let mut best_compass = 0usize;
                let mut best_dot = f32::MIN;
                for (i, &(cx, cz)) in COMPASS_DIRS.iter().enumerate() {
                    let dot = to_target.x * cx + to_target.z * cz;
                    if dot > best_dot { best_dot = dot; best_compass = i; }
                }
                move_target = best_compass as f32;
            }
        } else if let Some(enemy_pos) = nearest_enemy_pos {
            let to_enemy = (enemy_pos - pos).normalize_or_zero();
            let mut best_compass = 0usize;
            let mut best_dot = f32::MIN;
            for (i, &(cx, cz)) in COMPASS_DIRS.iter().enumerate() {
                let dot = to_enemy.x * cx + to_enemy.z * cz;
                if dot > best_dot { best_dot = dot; best_compass = i; }
            }
            move_target = best_compass as f32;
        } else {
            let phase = (tick.tick / 50 + agent_hash) % 8;
            move_target = phase as f32;
        }

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

        let action = vec![move_target, shoot, 0.0, use_action];
        raw_buffer.insert(entity, action);
    }
}

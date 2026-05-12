use bevy_ecs::prelude::*;
use glam::Vec3;

use crate::action_space::RawActionBuffer;
use crate::ecs::components::{Agent, Dead, Team};
use crate::ecs::resources::TickState;

use crate::physics3d::Physics3DState;

use super::cs_lite::{
    CsLiteConfig, CsRoundState, Facing3D, PhysicsHandle3D,
    Position3D, RoundPhase,
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
        (Entity, &Position3D, &Facing3D, &Team, &PhysicsHandle3D),
        (With<Agent>, Without<Dead>),
    >,
    mut raw_buffer: ResMut<RawActionBuffer>,
    config: Res<CsLiteConfig>,
    round: Res<CsRoundState>,
    tick: Res<TickState>,
    physics3d: Res<Physics3DState>,
) {
    if round.phase != RoundPhase::Active {
        return;
    }

    let all: Vec<_> = agents
        .iter()
        .map(|(e, p, _f, t, _ph)| (e, p.0, t.0))
        .collect();

    for &(entity, pos, team) in &all {
        if raw_buffer.get(entity).is_some() {
            continue;
        }

        let eye_pos = pos + Vec3::Y * config.eye_height;
        let agent_hash = entity.to_bits();

        // Find nearest enemy
        let mut nearest_enemy_pos = None;
        let mut nearest_enemy_dist = f32::MAX;
        let mut visible_enemy_pos = None;
        let mut visible_enemy_dist = f32::MAX;

        for &(other_e, other_pos, other_team) in &all {
            if other_e == entity || other_team == team {
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

        if visible_enemy_pos.is_some() {
            // Auto-aim handles facing; only shoot sometimes (dummy is bad)
            if (tick.tick + agent_hash) % 3 == 0 {
                shoot = 1.0;
            }

            // Wander randomly while vaguely approaching
            let phase = (tick.tick / 30 + agent_hash) % 8;
            move_target = phase as f32; // random compass direction
        } else if nearest_enemy_pos.is_some() {
            // Can't see enemy — wander randomly
            let phase = (tick.tick / 40 + agent_hash) % 8;
            move_target = phase as f32;
        } else {
            // No enemy — wander toward center
            let phase = (tick.tick / 50 + agent_hash) % 8;
            move_target = phase as f32;
        }

        let action = vec![move_target, shoot];
        raw_buffer.insert(entity, action);
    }
}

use bevy_ecs::prelude::*;
use glam::Vec2;

use crate::ecs::components::{Dead, Facing, Position, Team};
use crate::rewards::RewardFn;

/// Reward based on how well the agent is aiming toward the nearest enemy.
///
/// Computes the dot product between the agent's facing direction and the
/// direction toward the nearest living enemy on a different team. Returns
/// `weight * dot.max(0.0)` — perfect aim yields `weight`, facing away yields 0.
///
/// Returns 0.0 if no enemies exist or if the agent lacks required components.
pub struct AimReward {
    pub weight: f32,
}

impl RewardFn for AimReward {
    fn name(&self) -> &str {
        "aim"
    }

    fn compute(&self, world: &World, agent: Entity) -> f32 {
        let agent_pos = match world.get::<Position>(agent) {
            Some(p) => p.0,
            None => return 0.0,
        };
        let agent_facing = match world.get::<Facing>(agent) {
            Some(f) => f.0,
            None => return 0.0,
        };
        let agent_team = match world.get::<Team>(agent) {
            Some(t) => t.0,
            None => return 0.0,
        };

        // Find nearest living enemy by iterating all entities.
        // We use iter_entities() instead of query_filtered because we only
        // have an immutable &World reference.
        let mut nearest: Option<(f32, Vec2)> = None;

        for entity_ref in world.iter_entities() {
            let entity = entity_ref.id();
            if entity == agent {
                continue;
            }
            // Skip dead entities
            if entity_ref.get::<Dead>().is_some() {
                continue;
            }
            let Some(pos) = entity_ref.get::<Position>() else {
                continue;
            };
            let Some(team) = entity_ref.get::<Team>() else {
                continue;
            };
            if team.0 == agent_team {
                continue;
            }
            let dist = agent_pos.distance(pos.0);
            match &nearest {
                None => nearest = Some((dist, pos.0)),
                Some((best_dist, _)) if dist < *best_dist => {
                    nearest = Some((dist, pos.0));
                }
                _ => {}
            }
        }

        let enemy_pos = match nearest {
            Some((_, pos)) => pos,
            None => return 0.0,
        };

        let to_enemy = enemy_pos - agent_pos;
        if to_enemy.length_squared() < 1e-10 {
            // On top of the enemy — facing doesn't matter, full reward.
            return self.weight;
        }
        let to_enemy_dir = to_enemy.normalize();

        // Agent facing direction from angle
        let facing_dir = Vec2::new(agent_facing.cos(), agent_facing.sin());

        let dot = facing_dir.dot(to_enemy_dir);
        self.weight * dot.max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn spawn_agent(world: &mut World, pos: Vec2, facing: f32, team: u8) -> Entity {
        world
            .spawn((Position(pos), Facing(facing), Team(team)))
            .id()
    }

    fn spawn_enemy(world: &mut World, pos: Vec2, team: u8) -> Entity {
        world
            .spawn((Position(pos), Facing(0.0), Team(team)))
            .id()
    }

    #[test]
    fn aim_reward_name() {
        let reward = AimReward { weight: 1.0 };
        assert_eq!(reward.name(), "aim");
    }

    #[test]
    fn aim_reward_facing_enemy_directly() {
        let mut world = World::new();
        // Agent at origin, facing right (angle 0), enemy to the right
        let agent = spawn_agent(&mut world, Vec2::new(0.0, 0.0), 0.0, 0);
        let _enemy = spawn_enemy(&mut world, Vec2::new(10.0, 0.0), 1);

        let reward = AimReward { weight: 1.0 };
        let r = reward.compute(&world, agent);
        assert!((r - 1.0).abs() < 1e-5, "Expected ~1.0, got {r}");
    }

    #[test]
    fn aim_reward_facing_away_from_enemy() {
        let mut world = World::new();
        // Agent at origin, facing left (PI), enemy to the right
        let agent = spawn_agent(&mut world, Vec2::new(0.0, 0.0), PI, 0);
        let _enemy = spawn_enemy(&mut world, Vec2::new(10.0, 0.0), 1);

        let reward = AimReward { weight: 1.0 };
        let r = reward.compute(&world, agent);
        // dot product should be ~-1, clamped to 0
        assert!(r.abs() < 1e-5, "Expected ~0.0, got {r}");
    }

    #[test]
    fn aim_reward_facing_perpendicular() {
        let mut world = World::new();
        // Agent at origin, facing up (PI/2), enemy to the right
        let agent = spawn_agent(&mut world, Vec2::new(0.0, 0.0), PI / 2.0, 0);
        let _enemy = spawn_enemy(&mut world, Vec2::new(10.0, 0.0), 1);

        let reward = AimReward { weight: 1.0 };
        let r = reward.compute(&world, agent);
        // dot product should be ~0
        assert!(r.abs() < 1e-5, "Expected ~0.0, got {r}");
    }

    #[test]
    fn aim_reward_no_enemies() {
        let mut world = World::new();
        let agent = spawn_agent(&mut world, Vec2::new(0.0, 0.0), 0.0, 0);
        // Teammate only
        let _teammate = spawn_enemy(&mut world, Vec2::new(10.0, 0.0), 0);

        let reward = AimReward { weight: 1.0 };
        assert_eq!(reward.compute(&world, agent), 0.0);
    }

    #[test]
    fn aim_reward_dead_enemies_ignored() {
        let mut world = World::new();
        let agent = spawn_agent(&mut world, Vec2::new(0.0, 0.0), 0.0, 0);
        world.spawn((Position(Vec2::new(10.0, 0.0)), Facing(0.0), Team(1), Dead));

        let reward = AimReward { weight: 1.0 };
        assert_eq!(reward.compute(&world, agent), 0.0);
    }

    #[test]
    fn aim_reward_with_weight() {
        let mut world = World::new();
        let agent = spawn_agent(&mut world, Vec2::new(0.0, 0.0), 0.0, 0);
        let _enemy = spawn_enemy(&mut world, Vec2::new(10.0, 0.0), 1);

        let reward = AimReward { weight: 0.5 };
        let r = reward.compute(&world, agent);
        assert!((r - 0.5).abs() < 1e-5, "Expected ~0.5, got {r}");
    }

    #[test]
    fn aim_reward_no_position_returns_zero() {
        let mut world = World::new();
        let agent = world.spawn((Facing(0.0), Team(0))).id();

        let reward = AimReward { weight: 1.0 };
        assert_eq!(reward.compute(&world, agent), 0.0);
    }

    #[test]
    fn aim_reward_no_facing_returns_zero() {
        let mut world = World::new();
        let agent = world.spawn((Position(Vec2::ZERO), Team(0))).id();

        let reward = AimReward { weight: 1.0 };
        assert_eq!(reward.compute(&world, agent), 0.0);
    }

    #[test]
    fn aim_reward_no_team_returns_zero() {
        let mut world = World::new();
        let agent = world.spawn((Position(Vec2::ZERO), Facing(0.0))).id();

        let reward = AimReward { weight: 1.0 };
        assert_eq!(reward.compute(&world, agent), 0.0);
    }

    #[test]
    fn aim_reward_picks_nearest_enemy() {
        let mut world = World::new();
        // Agent facing right
        let agent = spawn_agent(&mut world, Vec2::new(0.0, 0.0), 0.0, 0);
        // Near enemy to the right (directly ahead)
        let _near = spawn_enemy(&mut world, Vec2::new(5.0, 0.0), 1);
        // Far enemy up (perpendicular)
        let _far = spawn_enemy(&mut world, Vec2::new(0.0, 100.0), 1);

        let reward = AimReward { weight: 1.0 };
        let r = reward.compute(&world, agent);
        // Should aim at nearest (right), dot ~1.0
        assert!((r - 1.0).abs() < 1e-5, "Expected ~1.0, got {r}");
    }
}

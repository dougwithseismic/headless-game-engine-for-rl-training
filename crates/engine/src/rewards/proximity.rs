use bevy_ecs::prelude::*;

use crate::ecs::components::{Dead, Position, Team};
use crate::rewards::RewardFn;

/// Reward based on distance to the nearest enemy.
///
/// Returns `weight * (1.0 - dist / max_range)`, clamped to [0, weight].
/// Closer enemies yield higher rewards. Returns 0.0 if no enemies exist
/// or if the agent lacks Position/Team components.
pub struct ProximityReward {
    pub weight: f32,
    pub max_range: f32,
}

impl RewardFn for ProximityReward {
    fn name(&self) -> &str {
        "proximity"
    }

    fn compute(&self, world: &World, agent: Entity) -> f32 {
        let agent_pos = match world.get::<Position>(agent) {
            Some(p) => p.0,
            None => return 0.0,
        };
        let agent_team = match world.get::<Team>(agent) {
            Some(t) => t.0,
            None => return 0.0,
        };

        let mut min_dist = f32::MAX;

        // Iterate all entities to find living enemies on a different team.
        // We use iter_entities() because we only have an immutable &World.
        for entity_ref in world.iter_entities() {
            let entity = entity_ref.id();
            if entity == agent {
                continue;
            }
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
            if dist < min_dist {
                min_dist = dist;
            }
        }

        if min_dist == f32::MAX {
            return 0.0;
        }

        let normalized = (1.0 - min_dist / self.max_range).clamp(0.0, 1.0);
        self.weight * normalized
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    fn spawn_agent(world: &mut World, pos: Vec2, team: u8) -> Entity {
        world.spawn((Position(pos), Team(team))).id()
    }

    fn spawn_dead_agent(world: &mut World, pos: Vec2, team: u8) -> Entity {
        world.spawn((Position(pos), Team(team), Dead)).id()
    }

    #[test]
    fn proximity_reward_name() {
        let reward = ProximityReward {
            weight: 1.0,
            max_range: 100.0,
        };
        assert_eq!(reward.name(), "proximity");
    }

    #[test]
    fn proximity_reward_close_enemy() {
        let mut world = World::new();
        let agent = spawn_agent(&mut world, Vec2::new(0.0, 0.0), 0);
        let _enemy = spawn_agent(&mut world, Vec2::new(10.0, 0.0), 1);

        let reward = ProximityReward {
            weight: 1.0,
            max_range: 100.0,
        };
        // dist = 10, normalized = 1.0 - 10/100 = 0.9
        let r = reward.compute(&world, agent);
        assert!((r - 0.9).abs() < 1e-5);
    }

    #[test]
    fn proximity_reward_far_enemy() {
        let mut world = World::new();
        let agent = spawn_agent(&mut world, Vec2::new(0.0, 0.0), 0);
        let _enemy = spawn_agent(&mut world, Vec2::new(100.0, 0.0), 1);

        let reward = ProximityReward {
            weight: 1.0,
            max_range: 100.0,
        };
        // dist = 100, normalized = 1.0 - 100/100 = 0.0
        let r = reward.compute(&world, agent);
        assert!((r - 0.0).abs() < 1e-5);
    }

    #[test]
    fn proximity_reward_beyond_max_range_clamps_to_zero() {
        let mut world = World::new();
        let agent = spawn_agent(&mut world, Vec2::new(0.0, 0.0), 0);
        let _enemy = spawn_agent(&mut world, Vec2::new(200.0, 0.0), 1);

        let reward = ProximityReward {
            weight: 1.0,
            max_range: 100.0,
        };
        // dist = 200, normalized = (1.0 - 200/100).clamp(0,1) = 0.0
        let r = reward.compute(&world, agent);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn proximity_reward_no_enemies_returns_zero() {
        let mut world = World::new();
        let agent = spawn_agent(&mut world, Vec2::new(0.0, 0.0), 0);
        // teammate, same team
        let _teammate = spawn_agent(&mut world, Vec2::new(5.0, 0.0), 0);

        let reward = ProximityReward {
            weight: 1.0,
            max_range: 100.0,
        };
        assert_eq!(reward.compute(&world, agent), 0.0);
    }

    #[test]
    fn proximity_reward_dead_enemies_ignored() {
        let mut world = World::new();
        let agent = spawn_agent(&mut world, Vec2::new(0.0, 0.0), 0);
        let _dead_enemy = spawn_dead_agent(&mut world, Vec2::new(5.0, 0.0), 1);

        let reward = ProximityReward {
            weight: 1.0,
            max_range: 100.0,
        };
        assert_eq!(reward.compute(&world, agent), 0.0);
    }

    #[test]
    fn proximity_reward_picks_nearest_enemy() {
        let mut world = World::new();
        let agent = spawn_agent(&mut world, Vec2::new(0.0, 0.0), 0);
        let _far_enemy = spawn_agent(&mut world, Vec2::new(80.0, 0.0), 1);
        let _near_enemy = spawn_agent(&mut world, Vec2::new(20.0, 0.0), 1);

        let reward = ProximityReward {
            weight: 1.0,
            max_range: 100.0,
        };
        // nearest is 20, normalized = 1.0 - 20/100 = 0.8
        let r = reward.compute(&world, agent);
        assert!((r - 0.8).abs() < 1e-5);
    }

    #[test]
    fn proximity_reward_with_weight() {
        let mut world = World::new();
        let agent = spawn_agent(&mut world, Vec2::new(0.0, 0.0), 0);
        let _enemy = spawn_agent(&mut world, Vec2::new(50.0, 0.0), 1);

        let reward = ProximityReward {
            weight: 0.5,
            max_range: 100.0,
        };
        // dist = 50, normalized = 0.5, weighted = 0.5 * 0.5 = 0.25
        let r = reward.compute(&world, agent);
        assert!((r - 0.25).abs() < 1e-5);
    }

    #[test]
    fn proximity_reward_agent_no_position_returns_zero() {
        let mut world = World::new();
        let agent = world.spawn(Team(0)).id();

        let reward = ProximityReward {
            weight: 1.0,
            max_range: 100.0,
        };
        assert_eq!(reward.compute(&world, agent), 0.0);
    }

    #[test]
    fn proximity_reward_agent_no_team_returns_zero() {
        let mut world = World::new();
        let agent = world.spawn(Position(Vec2::ZERO)).id();

        let reward = ProximityReward {
            weight: 1.0,
            max_range: 100.0,
        };
        assert_eq!(reward.compute(&world, agent), 0.0);
    }
}

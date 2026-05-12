use bevy_ecs::prelude::*;

use crate::observation::RewardBuffer;
use crate::rewards::RewardFn;

/// Reads accumulated kill/death rewards from the `RewardBuffer` resource.
///
/// The `RewardBuffer` is populated by combat systems (kills +1, deaths -1).
/// This reward function simply returns whatever has been accumulated for the
/// agent during the current tick.
pub struct CombatReward;

impl RewardFn for CombatReward {
    fn name(&self) -> &str {
        "combat"
    }

    fn compute(&self, world: &World, agent: Entity) -> f32 {
        world
            .get_resource::<RewardBuffer>()
            .map(|buf| buf.get(agent))
            .unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combat_reward_name() {
        assert_eq!(CombatReward.name(), "combat");
    }

    #[test]
    fn combat_reward_reads_from_buffer() {
        let mut world = World::new();
        let agent = world.spawn_empty().id();

        let mut buf = RewardBuffer::default();
        buf.add(agent, 1.0); // kill
        buf.add(agent, -1.0); // death
        buf.add(agent, 1.0); // another kill
        world.insert_resource(buf);

        let reward = CombatReward;
        assert_eq!(reward.compute(&world, agent), 1.0);
    }

    #[test]
    fn combat_reward_returns_zero_when_no_buffer() {
        let mut world = World::new();
        let agent = world.spawn_empty().id();

        let reward = CombatReward;
        assert_eq!(reward.compute(&world, agent), 0.0);
    }

    #[test]
    fn combat_reward_returns_zero_for_missing_agent() {
        let mut world = World::new();
        let agent = world.spawn_empty().id();
        world.insert_resource(RewardBuffer::default());

        let reward = CombatReward;
        assert_eq!(reward.compute(&world, agent), 0.0);
    }

    #[test]
    fn combat_reward_independent_agents() {
        let mut world = World::new();
        let agent_a = world.spawn_empty().id();
        let agent_b = world.spawn_empty().id();

        let mut buf = RewardBuffer::default();
        buf.add(agent_a, 2.0);
        buf.add(agent_b, -1.0);
        world.insert_resource(buf);

        let reward = CombatReward;
        assert_eq!(reward.compute(&world, agent_a), 2.0);
        assert_eq!(reward.compute(&world, agent_b), -1.0);
    }
}

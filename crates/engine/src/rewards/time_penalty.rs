use bevy_ecs::prelude::*;

use crate::rewards::RewardFn;

/// A small negative constant reward per tick.
///
/// Encourages the agent to complete objectives faster. Typically configured
/// with a small negative value like -0.0005.
pub struct TimePenalty {
    pub penalty: f32,
}

impl RewardFn for TimePenalty {
    fn name(&self) -> &str {
        "time_penalty"
    }

    fn compute(&self, _world: &World, _agent: Entity) -> f32 {
        self.penalty
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_penalty_name() {
        let reward = TimePenalty { penalty: -0.001 };
        assert_eq!(reward.name(), "time_penalty");
    }

    #[test]
    fn time_penalty_returns_constant() {
        let mut world = World::new();
        let agent = world.spawn_empty().id();

        let reward = TimePenalty { penalty: -0.0005 };
        assert_eq!(reward.compute(&world, agent), -0.0005);
    }

    #[test]
    fn time_penalty_same_for_all_agents() {
        let mut world = World::new();
        let a = world.spawn_empty().id();
        let b = world.spawn_empty().id();

        let reward = TimePenalty { penalty: -0.001 };
        assert_eq!(reward.compute(&world, a), reward.compute(&world, b));
    }

    #[test]
    fn time_penalty_zero_penalty() {
        let mut world = World::new();
        let agent = world.spawn_empty().id();

        let reward = TimePenalty { penalty: 0.0 };
        assert_eq!(reward.compute(&world, agent), 0.0);
    }
}

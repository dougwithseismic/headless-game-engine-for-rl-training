pub mod aim;
pub mod combat;
pub mod objective;
pub mod proximity;
pub mod time_penalty;

use bevy_ecs::prelude::*;

/// A composable reward function that computes a scalar reward for an agent.
///
/// Implementations must be `Send + Sync` so they can be stored alongside the
/// `TickRunner` and shared across threads when needed.
pub trait RewardFn: Send + Sync {
    /// Human-readable name for this reward component.
    fn name(&self) -> &str;

    /// Compute the reward for `agent` given the current ECS `world` state.
    fn compute(&self, world: &World, agent: Entity) -> f32;
}

/// A weighted collection of `RewardFn` components.
///
/// The final reward is the weighted sum: `sum(weight_i * reward_i(world, agent))`.
///
/// # Example
///
/// ```ignore
/// let reward = CompositeReward::new()
///     .add(1.0, CombatReward)
///     .add(0.1, ProximityReward { weight: 1.0, max_range: 500.0 })
///     .add(1.0, TimePenalty { penalty: -0.0005 });
/// ```
pub struct CompositeReward {
    components: Vec<(f32, Box<dyn RewardFn>)>,
}

impl Default for CompositeReward {
    fn default() -> Self {
        Self::new()
    }
}

impl CompositeReward {
    pub fn new() -> Self {
        Self {
            components: vec![],
        }
    }

    /// Add a reward function with a weight (builder pattern).
    pub fn add(mut self, weight: f32, reward: impl RewardFn + 'static) -> Self {
        self.components.push((weight, Box::new(reward)));
        self
    }

    /// Add a boxed reward function with a weight (builder pattern).
    pub fn add_boxed(mut self, weight: f32, reward: Box<dyn RewardFn>) -> Self {
        self.components.push((weight, reward));
        self
    }

    /// Number of reward components.
    pub fn len(&self) -> usize {
        self.components.len()
    }

    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }
}

impl RewardFn for CompositeReward {
    fn name(&self) -> &str {
        "composite"
    }

    fn compute(&self, world: &World, agent: Entity) -> f32 {
        self.components
            .iter()
            .map(|(w, r)| w * r.compute(world, agent))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ConstantReward(f32);

    impl RewardFn for ConstantReward {
        fn name(&self) -> &str {
            "constant"
        }
        fn compute(&self, _world: &World, _agent: Entity) -> f32 {
            self.0
        }
    }

    #[test]
    fn composite_reward_empty_returns_zero() {
        let reward = CompositeReward::new();
        let mut world = World::new();
        let agent = world.spawn_empty().id();
        assert_eq!(reward.compute(&world, agent), 0.0);
    }

    #[test]
    fn composite_reward_single_component() {
        let reward = CompositeReward::new().add(1.0, ConstantReward(5.0));
        let mut world = World::new();
        let agent = world.spawn_empty().id();
        assert_eq!(reward.compute(&world, agent), 5.0);
    }

    #[test]
    fn composite_reward_weighted_sum() {
        let reward = CompositeReward::new()
            .add(2.0, ConstantReward(3.0))
            .add(0.5, ConstantReward(10.0));
        let mut world = World::new();
        let agent = world.spawn_empty().id();
        // 2.0 * 3.0 + 0.5 * 10.0 = 6.0 + 5.0 = 11.0
        assert_eq!(reward.compute(&world, agent), 11.0);
    }

    #[test]
    fn composite_reward_negative_weights() {
        let reward = CompositeReward::new()
            .add(1.0, ConstantReward(10.0))
            .add(-1.0, ConstantReward(3.0));
        let mut world = World::new();
        let agent = world.spawn_empty().id();
        // 1.0 * 10.0 + (-1.0) * 3.0 = 7.0
        assert_eq!(reward.compute(&world, agent), 7.0);
    }

    #[test]
    fn composite_reward_name() {
        let reward = CompositeReward::new();
        assert_eq!(reward.name(), "composite");
    }

    #[test]
    fn composite_reward_len_and_empty() {
        let reward = CompositeReward::new();
        assert!(reward.is_empty());
        assert_eq!(reward.len(), 0);

        let reward = reward.add(1.0, ConstantReward(0.0));
        assert!(!reward.is_empty());
        assert_eq!(reward.len(), 1);
    }

    #[test]
    fn composite_reward_default() {
        let reward = CompositeReward::default();
        assert!(reward.is_empty());
    }

    #[test]
    fn composite_reward_add_boxed() {
        let boxed: Box<dyn RewardFn> = Box::new(ConstantReward(7.0));
        let reward = CompositeReward::new().add_boxed(1.0, boxed);
        let mut world = World::new();
        let agent = world.spawn_empty().id();
        assert_eq!(reward.compute(&world, agent), 7.0);
    }
}

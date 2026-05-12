use bevy_ecs::prelude::*;

use crate::rewards::RewardFn;

/// A generic goal-completion reward using a closure.
///
/// Use this for custom, game-specific objective rewards that don't warrant
/// a dedicated struct.
///
/// # Example
///
/// ```ignore
/// let reward = ObjectiveReward::new("capture_flag", Box::new(|world, agent| {
///     // Check if agent captured the flag
///     if world.get::<FlagCarrier>(agent).is_some() { 1.0 } else { 0.0 }
/// }));
/// ```
pub struct ObjectiveReward {
    name: String,
    func: Box<dyn Fn(&World, Entity) -> f32 + Send + Sync>,
}

impl ObjectiveReward {
    pub fn new(
        name: impl Into<String>,
        func: Box<dyn Fn(&World, Entity) -> f32 + Send + Sync>,
    ) -> Self {
        Self {
            name: name.into(),
            func,
        }
    }
}

impl RewardFn for ObjectiveReward {
    fn name(&self) -> &str {
        &self.name
    }

    fn compute(&self, world: &World, agent: Entity) -> f32 {
        (self.func)(world, agent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn objective_reward_name() {
        let reward = ObjectiveReward::new("test_obj", Box::new(|_, _| 0.0));
        assert_eq!(reward.name(), "test_obj");
    }

    #[test]
    fn objective_reward_calls_closure() {
        let mut world = World::new();
        let agent = world.spawn_empty().id();

        let reward = ObjectiveReward::new("always_five", Box::new(|_, _| 5.0));
        assert_eq!(reward.compute(&world, agent), 5.0);
    }

    #[test]
    fn objective_reward_closure_receives_correct_entity() {
        let mut world = World::new();
        let agent_a = world.spawn_empty().id();
        let agent_b = world.spawn_empty().id();

        // Closure returns different values based on entity
        let target = agent_a;
        let reward = ObjectiveReward::new(
            "entity_check",
            Box::new(move |_, agent| if agent == target { 1.0 } else { 0.0 }),
        );

        assert_eq!(reward.compute(&world, agent_a), 1.0);
        assert_eq!(reward.compute(&world, agent_b), 0.0);
    }

    #[test]
    fn objective_reward_closure_can_read_world() {
        use crate::ecs::components::Position;
        use glam::Vec2;

        let mut world = World::new();
        let agent = world.spawn(Position(Vec2::new(3.0, 4.0))).id();

        // Closure reads position from world
        let reward = ObjectiveReward::new(
            "at_origin",
            Box::new(|world, agent| {
                let pos = world.get::<Position>(agent).unwrap();
                if pos.0.length() < 1.0 {
                    1.0
                } else {
                    0.0
                }
            }),
        );

        // Agent at (3,4), not at origin
        assert_eq!(reward.compute(&world, agent), 0.0);
    }
}

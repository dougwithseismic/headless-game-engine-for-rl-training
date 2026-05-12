use bevy_ecs::prelude::*;

use crate::action_space::ActionMaskBuffer;
use crate::features::{FeatureExtractor, ObsShape};

/// Extracts the action mask for the agent.
///
/// Reads from `ActionMaskBuffer`. If no mask is set for the agent, returns
/// all 1.0 (every action is valid).
///
/// Output shape: Vector(mask_size)
pub struct ActionMaskExtractor {
    mask_size: usize,
}

impl ActionMaskExtractor {
    pub fn new(mask_size: usize) -> Self {
        Self { mask_size }
    }
}

impl FeatureExtractor for ActionMaskExtractor {
    fn name(&self) -> &str {
        "action_mask"
    }

    fn shape(&self) -> ObsShape {
        ObsShape::Vector(self.mask_size)
    }

    fn extract(&self, world: &World, agent: Entity, buf: &mut [f32]) {
        debug_assert!(buf.len() >= self.mask_size);

        let mask_buf = world.get_resource::<ActionMaskBuffer>();

        match mask_buf.and_then(|m| m.get(agent)) {
            Some(mask) => {
                let len = mask.len().min(self.mask_size);
                for i in 0..len {
                    buf[i] = if mask[i] { 1.0 } else { 0.0 };
                }
                // If mask is shorter than mask_size, remaining are already 0.0
                // (treated as invalid). If mask is longer, we truncate.
            }
            None => {
                // No mask set: everything valid
                for v in buf.iter_mut().take(self.mask_size) {
                    *v = 1.0;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action_space::ActionMaskBuffer;

    #[test]
    fn shape_and_name() {
        let ext = ActionMaskExtractor::new(5);
        assert_eq!(ext.name(), "action_mask");
        assert_eq!(ext.shape(), ObsShape::Vector(5));
        assert_eq!(ext.shape().flat_size(), 5);
    }

    #[test]
    fn reads_mask_from_buffer() {
        let mut world = World::new();
        let agent = world.spawn_empty().id();

        let mut mask_buf = ActionMaskBuffer::default();
        mask_buf.set(agent, vec![true, false, true, true, false]);
        world.insert_resource(mask_buf);

        let ext = ActionMaskExtractor::new(5);
        let mut buf = vec![0.0; 5];
        ext.extract(&world, agent, &mut buf);

        assert_eq!(buf, vec![1.0, 0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn no_mask_returns_all_ones() {
        let mut world = World::new();
        let agent = world.spawn_empty().id();
        world.insert_resource(ActionMaskBuffer::default());

        let ext = ActionMaskExtractor::new(4);
        let mut buf = vec![0.0; 4];
        ext.extract(&world, agent, &mut buf);

        assert_eq!(buf, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn no_mask_resource_returns_all_ones() {
        let mut world = World::new();
        let agent = world.spawn_empty().id();
        // No ActionMaskBuffer inserted at all

        let ext = ActionMaskExtractor::new(3);
        let mut buf = vec![0.0; 3];
        ext.extract(&world, agent, &mut buf);

        assert_eq!(buf, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn mask_shorter_than_size_pads_zeros() {
        let mut world = World::new();
        let agent = world.spawn_empty().id();

        let mut mask_buf = ActionMaskBuffer::default();
        mask_buf.set(agent, vec![true, false]); // only 2, but size is 4
        world.insert_resource(mask_buf);

        let ext = ActionMaskExtractor::new(4);
        let mut buf = vec![0.0; 4];
        ext.extract(&world, agent, &mut buf);

        assert_eq!(buf, vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn mask_longer_than_size_truncates() {
        let mut world = World::new();
        let agent = world.spawn_empty().id();

        let mut mask_buf = ActionMaskBuffer::default();
        mask_buf.set(agent, vec![true, false, true, true, false, true]);
        world.insert_resource(mask_buf);

        let ext = ActionMaskExtractor::new(3);
        let mut buf = vec![0.0; 3];
        ext.extract(&world, agent, &mut buf);

        assert_eq!(buf, vec![1.0, 0.0, 1.0]);
    }
}

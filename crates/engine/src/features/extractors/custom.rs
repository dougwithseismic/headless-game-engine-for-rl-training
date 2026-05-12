use bevy_ecs::prelude::*;

use crate::features::{FeatureExtractor, ObsDtype, ObsShape};

/// A closure-based extractor for one-off game-specific features.
///
/// Wraps a user-provided function that reads from the ECS world and writes
/// feature values into a buffer. The user must provide the name, shape, and
/// extraction logic.
pub struct CustomExtractor {
    name: String,
    obs_shape: ObsShape,
    dtype: ObsDtype,
    extract_fn: Box<dyn Fn(&World, Entity, &mut [f32]) + Send + Sync>,
}

impl CustomExtractor {
    /// Create a custom extractor with default dtype (F32).
    pub fn new(
        name: impl Into<String>,
        shape: ObsShape,
        extract_fn: impl Fn(&World, Entity, &mut [f32]) + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            obs_shape: shape,
            dtype: ObsDtype::F32,
            extract_fn: Box::new(extract_fn),
        }
    }

    /// Set the dtype hint (builder pattern).
    pub fn with_dtype(mut self, dtype: ObsDtype) -> Self {
        self.dtype = dtype;
        self
    }
}

impl FeatureExtractor for CustomExtractor {
    fn name(&self) -> &str {
        &self.name
    }

    fn shape(&self) -> ObsShape {
        self.obs_shape.clone()
    }

    fn dtype(&self) -> ObsDtype {
        self.dtype
    }

    fn extract(&self, world: &World, agent: Entity, buf: &mut [f32]) {
        (self.extract_fn)(world, agent, buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_extractor_calls_closure() {
        let ext = CustomExtractor::new("custom_feat", ObsShape::Vector(3), |_world, _agent, buf| {
            buf[0] = 1.0;
            buf[1] = 2.0;
            buf[2] = 3.0;
        });

        assert_eq!(ext.name(), "custom_feat");
        assert_eq!(ext.shape(), ObsShape::Vector(3));
        assert_eq!(ext.dtype(), ObsDtype::F32);

        let mut world = World::new();
        let agent = world.spawn_empty().id();
        let mut buf = vec![0.0; 3];
        ext.extract(&world, agent, &mut buf);

        assert_eq!(buf, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn custom_extractor_with_dtype() {
        let ext = CustomExtractor::new("binary", ObsShape::Vector(4), |_w, _a, buf| {
            buf[0] = 1.0;
            buf[1] = 0.0;
            buf[2] = 1.0;
            buf[3] = 1.0;
        })
        .with_dtype(ObsDtype::Bool);

        assert_eq!(ext.dtype(), ObsDtype::Bool);
    }

    #[test]
    fn custom_extractor_reads_world() {
        use crate::ecs::components::Health;

        let ext = CustomExtractor::new("hp_only", ObsShape::Vector(1), |world, agent, buf| {
            if let Some(hp) = world.get::<Health>(agent) {
                buf[0] = hp.current;
            }
        });

        let mut world = World::new();
        let agent = world
            .spawn(Health {
                current: 42.0,
                max: 100.0,
            })
            .id();

        let mut buf = vec![0.0; 1];
        ext.extract(&world, agent, &mut buf);
        assert!((buf[0] - 42.0).abs() < 1e-5);
    }

    #[test]
    fn custom_extractor_spatial_shape() {
        let ext =
            CustomExtractor::new("minimap", ObsShape::Spatial(8, 8, 3), |_w, _a, _buf| {});

        assert_eq!(ext.shape(), ObsShape::Spatial(8, 8, 3));
        assert_eq!(ext.shape().flat_size(), 192);
    }
}

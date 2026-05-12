pub mod dict;
pub mod flat;
pub mod spatial;

use std::collections::HashMap;

use crate::features::ExtractorSet;
use crate::observation::ObsFeature;

/// Transforms raw extractor output into a specific tensor layout.
///
/// Converters sit between `ExtractorSet::observe()` and the Python boundary,
/// reshaping named feature buffers for different network architectures.
pub trait ObsConverter: Send + Sync {
    /// Human-readable name of this converter.
    fn name(&self) -> &str;

    /// Describe the output observation features given the current extractors.
    fn output_features(&self, extractors: &ExtractorSet) -> Vec<ObsFeature>;

    /// Convert raw extractor output into the converter's output format.
    fn convert(
        &self,
        raw: &HashMap<String, Vec<f32>>,
        extractors: &ExtractorSet,
    ) -> HashMap<String, Vec<f32>>;
}

// Re-export concrete converters for convenience.
pub use dict::DictConverter;
pub use flat::FlatConverter;
pub use spatial::SpatialConverter;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{ExtractorSet, FeatureExtractor, ObsShape};
    use bevy_ecs::prelude::*;

    // ── Test helpers ──

    pub(crate) struct MockExtractor {
        pub name: String,
        pub obs_shape: ObsShape,
        pub fill_value: f32,
    }

    impl FeatureExtractor for MockExtractor {
        fn name(&self) -> &str {
            &self.name
        }

        fn shape(&self) -> ObsShape {
            self.obs_shape.clone()
        }

        fn extract(&self, _world: &World, _agent: Entity, buf: &mut [f32]) {
            for v in buf.iter_mut() {
                *v = self.fill_value;
            }
        }
    }

    pub(crate) fn make_extractors() -> ExtractorSet {
        ExtractorSet::new()
            .add(MockExtractor {
                name: "pos".into(),
                obs_shape: ObsShape::Vector(3),
                fill_value: 1.0,
            })
            .add(MockExtractor {
                name: "hp".into(),
                obs_shape: ObsShape::Vector(2),
                fill_value: 2.0,
            })
    }

    pub(crate) fn make_raw_obs(extractors: &ExtractorSet) -> HashMap<String, Vec<f32>> {
        let mut world = World::new();
        let agent = world.spawn_empty().id();
        extractors.observe(&world, agent)
    }

    pub(crate) fn make_mixed_extractors() -> ExtractorSet {
        ExtractorSet::new()
            .add(MockExtractor {
                name: "self_state".into(),
                obs_shape: ObsShape::Vector(5),
                fill_value: 1.0,
            })
            .add(MockExtractor {
                name: "entities".into(),
                obs_shape: ObsShape::EntityList(4, 3),
                fill_value: 2.0,
            })
            .add(MockExtractor {
                name: "minimap".into(),
                obs_shape: ObsShape::Spatial(8, 8, 2),
                fill_value: 3.0,
            })
    }
}

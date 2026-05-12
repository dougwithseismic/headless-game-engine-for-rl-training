pub mod extractors;

use std::collections::HashMap;

use bevy_ecs::prelude::*;
use serde::Serialize;

use crate::observation::{ObsFeature, ObservationSpaceDef};

/// Shape of a feature tensor produced by an extractor.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ObsShape {
    /// Flat vector of values, e.g. [11] for self-state.
    Vector(usize),
    /// 2D spatial grid with channels, e.g. [64, 64, 8] for minimap.
    Spatial(usize, usize, usize),
    /// Variable-length entity list, e.g. [max_entities, features_per_entity].
    EntityList(usize, usize),
}

impl ObsShape {
    /// Total number of scalar elements when flattened.
    pub fn flat_size(&self) -> usize {
        match self {
            ObsShape::Vector(n) => *n,
            ObsShape::Spatial(w, h, c) => w * h * c,
            ObsShape::EntityList(max, feat) => max * feat,
        }
    }

    /// Convert to the shape vector used by `ObsFeature`.
    pub fn to_shape_vec(&self) -> Vec<usize> {
        match self {
            ObsShape::Vector(n) => vec![*n],
            ObsShape::Spatial(w, h, c) => vec![*w, *h, *c],
            ObsShape::EntityList(max, feat) => vec![*max, *feat],
        }
    }
}

/// Data type hint for observation values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ObsDtype {
    F32,
    U8,
    Bool,
}

/// A composable piece that extracts a named feature tensor from the ECS world.
pub trait FeatureExtractor: Send + Sync {
    /// Name of this feature (used as the key in observation dicts).
    fn name(&self) -> &str;

    /// Shape of the output tensor.
    fn shape(&self) -> ObsShape;

    /// Data type hint.
    fn dtype(&self) -> ObsDtype {
        ObsDtype::F32
    }

    /// Extract feature values for a specific agent into the provided buffer.
    /// Buffer is pre-allocated to `shape().flat_size()` and zero-initialized.
    fn extract(&self, world: &World, agent: Entity, buf: &mut [f32]);
}

/// An ordered collection of feature extractors.
pub struct ExtractorSet {
    extractors: Vec<Box<dyn FeatureExtractor>>,
}

impl Default for ExtractorSet {
    fn default() -> Self {
        Self::new()
    }
}

impl ExtractorSet {
    pub fn new() -> Self {
        Self {
            extractors: vec![],
        }
    }

    /// Add a typed extractor (builder pattern).
    pub fn add(mut self, extractor: impl FeatureExtractor + 'static) -> Self {
        self.extractors.push(Box::new(extractor));
        self
    }

    /// Add a boxed extractor (builder pattern).
    pub fn add_boxed(mut self, extractor: Box<dyn FeatureExtractor>) -> Self {
        self.extractors.push(extractor);
        self
    }

    /// Total flat size across all extractors.
    pub fn total_flat_size(&self) -> usize {
        self.extractors.iter().map(|e| e.shape().flat_size()).sum()
    }

    /// Generate an `ObservationSpaceDef` from the extractors.
    pub fn observation_space_def(&self) -> ObservationSpaceDef {
        let features = self
            .extractors
            .iter()
            .map(|e| ObsFeature {
                name: e.name().to_string(),
                shape: e.shape().to_shape_vec(),
            })
            .collect();
        ObservationSpaceDef { features }
    }

    /// Extract all features for an agent, returning named buffers.
    pub fn observe(&self, world: &World, agent: Entity) -> HashMap<String, Vec<f32>> {
        let mut result = HashMap::new();
        for extractor in &self.extractors {
            let size = extractor.shape().flat_size();
            let mut buf = vec![0.0; size];
            extractor.extract(world, agent, &mut buf);
            result.insert(extractor.name().to_string(), buf);
        }
        result
    }

    /// Access the underlying extractor slice.
    pub fn extractors(&self) -> &[Box<dyn FeatureExtractor>] {
        &self.extractors
    }

    pub fn len(&self) -> usize {
        self.extractors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.extractors.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ObsShape::flat_size ──

    #[test]
    fn obs_shape_vector_flat_size() {
        assert_eq!(ObsShape::Vector(11).flat_size(), 11);
        assert_eq!(ObsShape::Vector(0).flat_size(), 0);
        assert_eq!(ObsShape::Vector(1).flat_size(), 1);
    }

    #[test]
    fn obs_shape_spatial_flat_size() {
        assert_eq!(ObsShape::Spatial(64, 64, 8).flat_size(), 64 * 64 * 8);
        assert_eq!(ObsShape::Spatial(1, 1, 1).flat_size(), 1);
        assert_eq!(ObsShape::Spatial(10, 20, 3).flat_size(), 600);
    }

    #[test]
    fn obs_shape_entity_list_flat_size() {
        assert_eq!(ObsShape::EntityList(8, 5).flat_size(), 40);
        assert_eq!(ObsShape::EntityList(0, 5).flat_size(), 0);
        assert_eq!(ObsShape::EntityList(10, 0).flat_size(), 0);
    }

    #[test]
    fn obs_shape_serialize() {
        let v = ObsShape::Vector(4);
        let json = serde_json::to_string(&v).unwrap();
        assert!(json.contains("Vector"));
        assert!(json.contains("4"));
    }

    // ── ObsShape::to_shape_vec ──

    #[test]
    fn obs_shape_to_shape_vec() {
        assert_eq!(ObsShape::Vector(5).to_shape_vec(), vec![5]);
        assert_eq!(ObsShape::Spatial(3, 4, 2).to_shape_vec(), vec![3, 4, 2]);
        assert_eq!(ObsShape::EntityList(8, 6).to_shape_vec(), vec![8, 6]);
    }

    // ── Mock extractor for testing ExtractorSet ──

    struct MockExtractor {
        name: String,
        size: usize,
        fill_value: f32,
    }

    impl FeatureExtractor for MockExtractor {
        fn name(&self) -> &str {
            &self.name
        }

        fn shape(&self) -> ObsShape {
            ObsShape::Vector(self.size)
        }

        fn extract(&self, _world: &World, _agent: Entity, buf: &mut [f32]) {
            for v in buf.iter_mut() {
                *v = self.fill_value;
            }
        }
    }

    // ── ExtractorSet::new ──

    #[test]
    fn extractor_set_new_is_empty() {
        let set = ExtractorSet::new();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
        assert_eq!(set.total_flat_size(), 0);
    }

    #[test]
    fn extractor_set_default_is_empty() {
        let set = ExtractorSet::default();
        assert!(set.is_empty());
    }

    // ── ExtractorSet::add ──

    #[test]
    fn extractor_set_add_increments_count() {
        let set = ExtractorSet::new()
            .add(MockExtractor {
                name: "a".into(),
                size: 3,
                fill_value: 1.0,
            })
            .add(MockExtractor {
                name: "b".into(),
                size: 5,
                fill_value: 2.0,
            });
        assert_eq!(set.len(), 2);
        assert!(!set.is_empty());
    }

    // ── ExtractorSet::total_flat_size ──

    #[test]
    fn extractor_set_total_flat_size() {
        let set = ExtractorSet::new()
            .add(MockExtractor {
                name: "a".into(),
                size: 3,
                fill_value: 0.0,
            })
            .add(MockExtractor {
                name: "b".into(),
                size: 7,
                fill_value: 0.0,
            });
        assert_eq!(set.total_flat_size(), 10);
    }

    // ── ExtractorSet::observe ──

    #[test]
    fn extractor_set_observe_returns_named_buffers() {
        let mut world = World::new();
        let agent = world.spawn_empty().id();

        let set = ExtractorSet::new()
            .add(MockExtractor {
                name: "pos".into(),
                size: 2,
                fill_value: 1.5,
            })
            .add(MockExtractor {
                name: "hp".into(),
                size: 1,
                fill_value: 0.8,
            });

        let obs = set.observe(&world, agent);
        assert_eq!(obs.len(), 2);
        assert_eq!(obs["pos"], vec![1.5, 1.5]);
        assert_eq!(obs["hp"], vec![0.8]);
    }

    #[test]
    fn extractor_set_observe_zero_initializes_buffer() {
        // An extractor that does not write anything should get zeros
        struct NoopExtractor;
        impl FeatureExtractor for NoopExtractor {
            fn name(&self) -> &str {
                "noop"
            }
            fn shape(&self) -> ObsShape {
                ObsShape::Vector(4)
            }
            fn extract(&self, _world: &World, _agent: Entity, _buf: &mut [f32]) {
                // intentionally do nothing
            }
        }

        let mut world = World::new();
        let agent = world.spawn_empty().id();
        let set = ExtractorSet::new().add(NoopExtractor);
        let obs = set.observe(&world, agent);
        assert_eq!(obs["noop"], vec![0.0, 0.0, 0.0, 0.0]);
    }

    // ── ExtractorSet::observation_space_def ──

    #[test]
    fn extractor_set_observation_space_def() {
        let set = ExtractorSet::new()
            .add(MockExtractor {
                name: "self_state".into(),
                size: 11,
                fill_value: 0.0,
            })
            .add(MockExtractor {
                name: "action_mask".into(),
                size: 4,
                fill_value: 0.0,
            });

        let def = set.observation_space_def();
        assert_eq!(def.features.len(), 2);
        assert_eq!(def.features[0].name, "self_state");
        assert_eq!(def.features[0].shape, vec![11]);
        assert_eq!(def.features[1].name, "action_mask");
        assert_eq!(def.features[1].shape, vec![4]);
    }

    #[test]
    fn extractor_set_observation_space_def_entity_list() {
        struct EntListExtractor;
        impl FeatureExtractor for EntListExtractor {
            fn name(&self) -> &str {
                "entities"
            }
            fn shape(&self) -> ObsShape {
                ObsShape::EntityList(8, 5)
            }
            fn extract(&self, _world: &World, _agent: Entity, _buf: &mut [f32]) {}
        }

        let set = ExtractorSet::new().add(EntListExtractor);
        let def = set.observation_space_def();
        assert_eq!(def.features[0].shape, vec![8, 5]);
    }

    // ── ExtractorSet::add_boxed ──

    #[test]
    fn extractor_set_add_boxed() {
        let boxed: Box<dyn FeatureExtractor> = Box::new(MockExtractor {
            name: "boxed".into(),
            size: 2,
            fill_value: 0.0,
        });
        let set = ExtractorSet::new().add_boxed(boxed);
        assert_eq!(set.len(), 1);
        assert_eq!(set.extractors()[0].name(), "boxed");
    }

    // ── ExtractorSet::extractors ──

    #[test]
    fn extractor_set_extractors_returns_slice() {
        let set = ExtractorSet::new()
            .add(MockExtractor {
                name: "x".into(),
                size: 1,
                fill_value: 0.0,
            })
            .add(MockExtractor {
                name: "y".into(),
                size: 1,
                fill_value: 0.0,
            });
        let extractors = set.extractors();
        assert_eq!(extractors.len(), 2);
        assert_eq!(extractors[0].name(), "x");
        assert_eq!(extractors[1].name(), "y");
    }

    // ── ObsDtype default ──

    #[test]
    fn default_dtype_is_f32() {
        let ext = MockExtractor {
            name: "test".into(),
            size: 1,
            fill_value: 0.0,
        };
        assert_eq!(ext.dtype(), ObsDtype::F32);
    }
}

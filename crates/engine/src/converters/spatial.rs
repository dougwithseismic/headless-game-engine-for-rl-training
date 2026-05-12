use std::collections::HashMap;

use crate::features::{ExtractorSet, ObsShape};
use crate::observation::ObsFeature;

use super::ObsConverter;

/// Separates features by shape type and passes them through.
///
/// - `Vector` features are concatenated into a single "flat_features" buffer
/// - `EntityList` features are passed through with their original names
/// - `Spatial` features are passed through with their original names
///
/// This is a placeholder that demonstrates the pattern. Full spatial rendering
/// (projecting entities onto a grid) will come later when we have actual spatial
/// games.
pub struct SpatialConverter;

impl ObsConverter for SpatialConverter {
    fn name(&self) -> &str {
        "spatial"
    }

    fn output_features(&self, extractors: &ExtractorSet) -> Vec<ObsFeature> {
        let mut features = Vec::new();

        // Collect total flat size for Vector features.
        let flat_total: usize = extractors
            .extractors()
            .iter()
            .filter(|e| matches!(e.shape(), ObsShape::Vector(_)))
            .map(|e| e.shape().flat_size())
            .sum();

        if flat_total > 0 {
            features.push(ObsFeature {
                name: "flat_features".to_string(),
                shape: vec![flat_total],
            });
        }

        // EntityList and Spatial features keep their original names/shapes.
        for ext in extractors.extractors() {
            match ext.shape() {
                ObsShape::EntityList(_, _) | ObsShape::Spatial(_, _, _) => {
                    features.push(ObsFeature {
                        name: ext.name().to_string(),
                        shape: ext.shape().to_shape_vec(),
                    });
                }
                ObsShape::Vector(_) => {} // already handled above
            }
        }

        features
    }

    fn convert(
        &self,
        raw: &HashMap<String, Vec<f32>>,
        extractors: &ExtractorSet,
    ) -> HashMap<String, Vec<f32>> {
        let mut result = HashMap::new();

        // Concatenate all Vector features into "flat_features".
        let mut flat = Vec::new();
        for ext in extractors.extractors() {
            if matches!(ext.shape(), ObsShape::Vector(_)) {
                if let Some(buf) = raw.get(ext.name()) {
                    flat.extend_from_slice(buf);
                } else {
                    flat.resize(flat.len() + ext.shape().flat_size(), 0.0);
                }
            }
        }
        if !flat.is_empty() {
            result.insert("flat_features".to_string(), flat);
        }

        // Pass through EntityList and Spatial features unchanged.
        for ext in extractors.extractors() {
            match ext.shape() {
                ObsShape::EntityList(_, _) | ObsShape::Spatial(_, _, _) => {
                    if let Some(buf) = raw.get(ext.name()) {
                        result.insert(ext.name().to_string(), buf.clone());
                    } else {
                        let zeros = vec![0.0; ext.shape().flat_size()];
                        result.insert(ext.name().to_string(), zeros);
                    }
                }
                ObsShape::Vector(_) => {} // already handled
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::converters::tests::{make_mixed_extractors, make_raw_obs, MockExtractor};
    use crate::features::ExtractorSet;

    #[test]
    fn spatial_converter_name() {
        assert_eq!(SpatialConverter.name(), "spatial");
    }

    #[test]
    fn spatial_converter_separates_by_shape_type() {
        let extractors = make_mixed_extractors();
        let features = SpatialConverter.output_features(&extractors);

        // Should have: flat_features (from self_state), entities, minimap
        assert_eq!(features.len(), 3);

        assert_eq!(features[0].name, "flat_features");
        assert_eq!(features[0].shape, vec![5]); // Vector(5)

        assert_eq!(features[1].name, "entities");
        assert_eq!(features[1].shape, vec![4, 3]); // EntityList(4, 3)

        assert_eq!(features[2].name, "minimap");
        assert_eq!(features[2].shape, vec![8, 8, 2]); // Spatial(8, 8, 2)
    }

    #[test]
    fn spatial_converter_convert_groups_vectors() {
        let extractors = make_mixed_extractors();
        let raw = make_raw_obs(&extractors);

        let result = SpatialConverter.convert(&raw, &extractors);

        // flat_features: self_state values (fill=1.0, size=5)
        let flat = &result["flat_features"];
        assert_eq!(flat.len(), 5);
        assert!(flat.iter().all(|&v| v == 1.0));

        // entities: passed through (fill=2.0, size=12)
        let entities = &result["entities"];
        assert_eq!(entities.len(), 12);
        assert!(entities.iter().all(|&v| v == 2.0));

        // minimap: passed through (fill=3.0, size=128)
        let minimap = &result["minimap"];
        assert_eq!(minimap.len(), 128);
        assert!(minimap.iter().all(|&v| v == 3.0));
    }

    #[test]
    fn spatial_converter_vector_only() {
        let extractors = ExtractorSet::new()
            .add(MockExtractor {
                name: "a".into(),
                obs_shape: ObsShape::Vector(3),
                fill_value: 1.0,
            })
            .add(MockExtractor {
                name: "b".into(),
                obs_shape: ObsShape::Vector(2),
                fill_value: 2.0,
            });

        let features = SpatialConverter.output_features(&extractors);
        assert_eq!(features.len(), 1);
        assert_eq!(features[0].name, "flat_features");
        assert_eq!(features[0].shape, vec![5]);

        let raw = make_raw_obs(&extractors);
        let result = SpatialConverter.convert(&raw, &extractors);
        assert_eq!(result.len(), 1);
        let flat = &result["flat_features"];
        assert_eq!(flat, &[1.0, 1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn spatial_converter_no_vectors() {
        let extractors = ExtractorSet::new().add(MockExtractor {
            name: "grid".into(),
            obs_shape: ObsShape::Spatial(4, 4, 1),
            fill_value: 5.0,
        });

        let features = SpatialConverter.output_features(&extractors);
        assert_eq!(features.len(), 1);
        assert_eq!(features[0].name, "grid");
        assert_eq!(features[0].shape, vec![4, 4, 1]);

        let raw = make_raw_obs(&extractors);
        let result = SpatialConverter.convert(&raw, &extractors);
        assert_eq!(result.len(), 1);
        assert!(!result.contains_key("flat_features"));
        assert_eq!(result["grid"].len(), 16);
    }

    #[test]
    fn spatial_converter_empty_extractors() {
        let extractors = ExtractorSet::new();
        let raw = HashMap::new();

        let features = SpatialConverter.output_features(&extractors);
        assert!(features.is_empty());

        let result = SpatialConverter.convert(&raw, &extractors);
        assert!(result.is_empty());
    }
}

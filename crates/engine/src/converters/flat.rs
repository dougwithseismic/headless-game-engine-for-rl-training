use std::collections::HashMap;

use crate::features::ExtractorSet;
use crate::observation::ObsFeature;

use super::ObsConverter;

/// Concatenates all feature buffers into a single flat vector named "obs".
///
/// Useful for simple MLP-based policies that take a single flat input.
/// The concatenation order matches the extractor insertion order.
pub struct FlatConverter;

impl ObsConverter for FlatConverter {
    fn name(&self) -> &str {
        "flat"
    }

    fn output_features(&self, extractors: &ExtractorSet) -> Vec<ObsFeature> {
        vec![ObsFeature {
            name: "obs".to_string(),
            shape: vec![extractors.total_flat_size()],
        }]
    }

    fn convert(
        &self,
        raw: &HashMap<String, Vec<f32>>,
        extractors: &ExtractorSet,
    ) -> HashMap<String, Vec<f32>> {
        let total = extractors.total_flat_size();
        let mut flat = Vec::with_capacity(total);

        // Iterate in extractor order to ensure deterministic concatenation.
        for ext in extractors.extractors() {
            if let Some(buf) = raw.get(ext.name()) {
                flat.extend_from_slice(buf);
            } else {
                // Missing buffer: fill with zeros to maintain shape.
                flat.resize(flat.len() + ext.shape().flat_size(), 0.0);
            }
        }

        let mut result = HashMap::new();
        result.insert("obs".to_string(), flat);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::converters::tests::{make_extractors, make_raw_obs};

    #[test]
    fn flat_converter_name() {
        assert_eq!(FlatConverter.name(), "flat");
    }

    #[test]
    fn flat_converter_output_features_single_obs() {
        let extractors = make_extractors();
        let features = FlatConverter.output_features(&extractors);
        assert_eq!(features.len(), 1);
        assert_eq!(features[0].name, "obs");
        assert_eq!(features[0].shape, vec![5]); // 3 + 2
    }

    #[test]
    fn flat_converter_concatenates_in_extractor_order() {
        let extractors = make_extractors();
        let raw = make_raw_obs(&extractors);

        let result = FlatConverter.convert(&raw, &extractors);
        assert_eq!(result.len(), 1);

        let obs = &result["obs"];
        assert_eq!(obs.len(), 5);
        // pos (fill=1.0) comes first, then hp (fill=2.0)
        assert_eq!(&obs[..3], &[1.0, 1.0, 1.0]);
        assert_eq!(&obs[3..], &[2.0, 2.0]);
    }

    #[test]
    fn flat_converter_handles_missing_buffer_with_zeros() {
        let extractors = make_extractors();
        // Provide only the "pos" buffer, omit "hp"
        let mut raw = HashMap::new();
        raw.insert("pos".to_string(), vec![1.0, 1.0, 1.0]);

        let result = FlatConverter.convert(&raw, &extractors);
        let obs = &result["obs"];
        assert_eq!(obs.len(), 5);
        assert_eq!(&obs[..3], &[1.0, 1.0, 1.0]);
        assert_eq!(&obs[3..], &[0.0, 0.0]); // zeros for missing "hp"
    }

    #[test]
    fn flat_converter_empty_extractors() {
        let extractors = ExtractorSet::new();
        let raw = HashMap::new();

        let features = FlatConverter.output_features(&extractors);
        assert_eq!(features[0].shape, vec![0]);

        let result = FlatConverter.convert(&raw, &extractors);
        assert_eq!(result["obs"].len(), 0);
    }
}

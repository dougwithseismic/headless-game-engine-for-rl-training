use std::collections::HashMap;

use crate::features::ExtractorSet;
use crate::observation::ObsFeature;

use super::ObsConverter;

/// Pass-through converter: returns raw extractor output unchanged.
///
/// Useful for policies with separate input heads per feature
/// (e.g., entity attention networks, multi-input architectures).
pub struct DictConverter;

impl ObsConverter for DictConverter {
    fn name(&self) -> &str {
        "dict"
    }

    fn output_features(&self, extractors: &ExtractorSet) -> Vec<ObsFeature> {
        extractors
            .extractors()
            .iter()
            .map(|e| ObsFeature {
                name: e.name().to_string(),
                shape: e.shape().to_shape_vec(),
            })
            .collect()
    }

    fn convert(
        &self,
        raw: &HashMap<String, Vec<f32>>,
        _extractors: &ExtractorSet,
    ) -> HashMap<String, Vec<f32>> {
        raw.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::converters::tests::{make_extractors, make_raw_obs};

    #[test]
    fn dict_converter_name() {
        assert_eq!(DictConverter.name(), "dict");
    }

    #[test]
    fn dict_converter_output_features_mirrors_extractors() {
        let extractors = make_extractors();
        let features = DictConverter.output_features(&extractors);
        assert_eq!(features.len(), 2);
        assert_eq!(features[0].name, "pos");
        assert_eq!(features[0].shape, vec![3]);
        assert_eq!(features[1].name, "hp");
        assert_eq!(features[1].shape, vec![2]);
    }

    #[test]
    fn dict_converter_passthrough() {
        let extractors = make_extractors();
        let raw = make_raw_obs(&extractors);

        let result = DictConverter.convert(&raw, &extractors);
        assert_eq!(result, raw);
    }

    #[test]
    fn dict_converter_empty_extractors() {
        let extractors = ExtractorSet::new();
        let raw = HashMap::new();

        let features = DictConverter.output_features(&extractors);
        assert!(features.is_empty());

        let result = DictConverter.convert(&raw, &extractors);
        assert!(result.is_empty());
    }
}

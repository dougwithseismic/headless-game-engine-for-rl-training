use serde::Serialize;

use crate::action_space::ActionSpaceDef;

// ---------------------------------------------------------------------------
// Action Space types (function-call model)
// ---------------------------------------------------------------------------

/// A typed argument to an action function.
#[derive(Debug, Clone, Serialize)]
pub enum ArgType {
    /// Click a position on a 2D grid (for RTS/tower defense spatial actions).
    Spatial2D { width: u32, height: u32 },
    /// Pick from N discrete choices.
    Discrete { n: u32 },
    /// One or more continuous values in [low, high].
    Continuous {
        size: usize,
        low: Vec<f32>,
        high: Vec<f32>,
    },
    /// Select an entity from a set.
    EntitySelect { max: u32 },
}

impl ArgType {
    /// Number of floats needed to represent this argument in a flat vector.
    pub fn flat_size(&self) -> usize {
        match self {
            ArgType::Spatial2D { .. } => 2, // x, y
            ArgType::Discrete { .. } => 1,
            ArgType::Continuous { size, .. } => *size,
            ArgType::EntitySelect { .. } => 1,
        }
    }
}

/// A named action function with typed arguments.
#[derive(Debug, Clone, Serialize)]
pub struct ActionFn {
    pub name: String,
    pub id: usize,
    pub args: Vec<ArgType>,
}

impl ActionFn {
    /// Number of floats needed to represent all args for this function.
    pub fn flat_size(&self) -> usize {
        self.args.iter().map(|a| a.flat_size()).sum()
    }
}

/// Full action space definition using the function-call model.
#[derive(Debug, Clone, Serialize)]
pub struct ActionSpace {
    pub functions: Vec<ActionFn>,
}

impl ActionSpace {
    pub fn new(functions: Vec<ActionFn>) -> Self {
        Self { functions }
    }

    /// Number of action functions available.
    pub fn num_functions(&self) -> usize {
        self.functions.len()
    }

    /// Compute the flat size needed to encode all actions
    /// (for compatibility with the FlatTranslator).
    pub fn flat_size(&self) -> usize {
        self.functions.iter().map(|f| f.flat_size()).sum()
    }

    /// Convert from the legacy `ActionSpaceDef` format.
    pub fn from_legacy(legacy: &ActionSpaceDef) -> Self {
        let functions = legacy
            .heads
            .iter()
            .enumerate()
            .map(|(i, head)| {
                let (name, args) = match head {
                    crate::action_space::ActionHead::Continuous {
                        name,
                        size,
                        low,
                        high,
                    } => (
                        name.clone(),
                        vec![ArgType::Continuous {
                            size: *size,
                            low: low.clone(),
                            high: high.clone(),
                        }],
                    ),
                    crate::action_space::ActionHead::Discrete { name, n } => {
                        (name.clone(), vec![ArgType::Discrete { n: *n as u32 }])
                    }
                };
                ActionFn { name, id: i, args }
            })
            .collect();
        Self { functions }
    }
}

// ---------------------------------------------------------------------------
// ActionTranslator trait + implementations
// ---------------------------------------------------------------------------

/// Translates between external action representations and the flat `Vec<f32>`
/// that the engine's `RawActionBuffer` expects.
pub trait ActionTranslator: Send + Sync {
    /// The shape of the input the translator expects from the agent.
    fn input_size(&self, space: &ActionSpace) -> usize;

    /// Translate agent output to the flat action format for the engine.
    fn translate(&self, input: &[f32], space: &ActionSpace) -> Vec<f32>;
}

/// Identity translator: input is already a flat vector, pass through directly.
/// This is backward-compatible with the current system.
pub struct FlatTranslator;

impl ActionTranslator for FlatTranslator {
    fn input_size(&self, space: &ActionSpace) -> usize {
        space.flat_size()
    }

    fn translate(&self, input: &[f32], space: &ActionSpace) -> Vec<f32> {
        // Pass through, truncating or zero-padding to match expected size.
        let expected = space.flat_size();
        if input.len() == expected {
            input.to_vec()
        } else if input.len() > expected {
            input[..expected].to_vec()
        } else {
            let mut out = input.to_vec();
            out.resize(expected, 0.0);
            out
        }
    }
}

/// Function-call translator: input is `[function_id, arg0, arg1, ...]`.
///
/// Decodes which function was called and its arguments, then produces a flat
/// vector with zeros for un-called function args. Only the selected function's
/// slot is filled with the provided arguments.
pub struct FunctionCallTranslator;

impl ActionTranslator for FunctionCallTranslator {
    fn input_size(&self, space: &ActionSpace) -> usize {
        // 1 (function selector) + max args across all functions
        let max_args: usize = space
            .functions
            .iter()
            .map(|f| f.flat_size())
            .max()
            .unwrap_or(0);
        1 + max_args
    }

    fn translate(&self, input: &[f32], space: &ActionSpace) -> Vec<f32> {
        let total_flat = space.flat_size();
        let mut output = vec![0.0; total_flat];

        if input.is_empty() || space.functions.is_empty() {
            return output;
        }

        // First element is the function index (rounded to nearest int, clamped).
        let fn_idx = (input[0].round() as usize).min(space.functions.len() - 1);
        let args = &input[1..];

        // Find the offset into the flat vector for the selected function.
        let mut offset = 0;
        for (i, func) in space.functions.iter().enumerate() {
            if i == fn_idx {
                let func_size = func.flat_size();
                let copy_len = args.len().min(func_size);
                output[offset..offset + copy_len].copy_from_slice(&args[..copy_len]);
                break;
            }
            offset += func.flat_size();
        }

        output
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action_space::{ActionHead, ActionSpaceDef};

    fn sample_space() -> ActionSpace {
        ActionSpace::new(vec![
            ActionFn {
                name: "move".into(),
                id: 0,
                args: vec![ArgType::Continuous {
                    size: 2,
                    low: vec![-1.0; 2],
                    high: vec![1.0; 2],
                }],
            },
            ActionFn {
                name: "shoot".into(),
                id: 1,
                args: vec![ArgType::Discrete { n: 2 }],
            },
            ActionFn {
                name: "target".into(),
                id: 2,
                args: vec![ArgType::Spatial2D {
                    width: 64,
                    height: 64,
                }],
            },
            ActionFn {
                name: "focus".into(),
                id: 3,
                args: vec![ArgType::EntitySelect { max: 10 }],
            },
        ])
    }

    #[test]
    fn flat_size_computes_correctly_for_mixed_arg_types() {
        let space = sample_space();
        // move: 2 (continuous) + shoot: 1 (discrete) + target: 2 (spatial2d) + focus: 1 (entity) = 6
        assert_eq!(space.flat_size(), 6);
    }

    #[test]
    fn num_functions_returns_count() {
        let space = sample_space();
        assert_eq!(space.num_functions(), 4);
    }

    #[test]
    fn from_legacy_converts_action_space_def_correctly() {
        let legacy = ActionSpaceDef::new(vec![
            ActionHead::Continuous {
                name: "move_dir".into(),
                size: 2,
                low: vec![-1.0; 2],
                high: vec![1.0; 2],
            },
            ActionHead::Discrete {
                name: "fire".into(),
                n: 3,
            },
            ActionHead::Continuous {
                name: "aim".into(),
                size: 1,
                low: vec![-3.14],
                high: vec![3.14],
            },
        ]);

        let space = ActionSpace::from_legacy(&legacy);
        assert_eq!(space.num_functions(), 3);
        assert_eq!(space.flat_size(), legacy.total_size);

        // Verify function names and ids
        assert_eq!(space.functions[0].name, "move_dir");
        assert_eq!(space.functions[0].id, 0);
        assert_eq!(space.functions[1].name, "fire");
        assert_eq!(space.functions[1].id, 1);
        assert_eq!(space.functions[2].name, "aim");
        assert_eq!(space.functions[2].id, 2);

        // Verify arg types
        match &space.functions[0].args[0] {
            ArgType::Continuous { size, low, high } => {
                assert_eq!(*size, 2);
                assert_eq!(low, &vec![-1.0; 2]);
                assert_eq!(high, &vec![1.0; 2]);
            }
            other => panic!("expected Continuous, got {:?}", other),
        }
        match &space.functions[1].args[0] {
            ArgType::Discrete { n } => assert_eq!(*n, 3),
            other => panic!("expected Discrete, got {:?}", other),
        }
        match &space.functions[2].args[0] {
            ArgType::Continuous { size, .. } => assert_eq!(*size, 1),
            other => panic!("expected Continuous, got {:?}", other),
        }
    }

    #[test]
    fn flat_translator_passes_through_unchanged() {
        let space = sample_space();
        let translator = FlatTranslator;

        let input = vec![0.5, -0.3, 1.0, 32.0, 48.0, 7.0];
        let output = translator.translate(&input, &space);

        assert_eq!(input, output);
    }

    #[test]
    fn flat_translator_pads_short_input() {
        let space = sample_space(); // flat_size = 6
        let translator = FlatTranslator;

        let input = vec![0.5, -0.3];
        let output = translator.translate(&input, &space);

        assert_eq!(output.len(), 6);
        assert_eq!(output[0], 0.5);
        assert_eq!(output[1], -0.3);
        assert_eq!(output[2], 0.0);
        assert_eq!(output[3], 0.0);
        assert_eq!(output[4], 0.0);
        assert_eq!(output[5], 0.0);
    }

    #[test]
    fn flat_translator_truncates_long_input() {
        let space = sample_space(); // flat_size = 6
        let translator = FlatTranslator;

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let output = translator.translate(&input, &space);

        assert_eq!(output.len(), 6);
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn flat_translator_input_size_matches_flat_size() {
        let space = sample_space();
        let translator = FlatTranslator;

        assert_eq!(translator.input_size(&space), space.flat_size());
    }

    #[test]
    fn function_call_translator_decodes_first_function() {
        let space = sample_space();
        let translator = FunctionCallTranslator;

        // Call function 0 ("move") with args [0.7, -0.2]
        let input = vec![0.0, 0.7, -0.2];
        let output = translator.translate(&input, &space);

        assert_eq!(output.len(), 6);
        assert_eq!(output[0], 0.7); // move arg 0
        assert_eq!(output[1], -0.2); // move arg 1
        assert_eq!(output[2], 0.0); // shoot (not called)
        assert_eq!(output[3], 0.0); // target x (not called)
        assert_eq!(output[4], 0.0); // target y (not called)
        assert_eq!(output[5], 0.0); // focus (not called)
    }

    #[test]
    fn function_call_translator_decodes_second_function() {
        let space = sample_space();
        let translator = FunctionCallTranslator;

        // Call function 1 ("shoot") with arg [1.0]
        let input = vec![1.0, 1.0];
        let output = translator.translate(&input, &space);

        assert_eq!(output.len(), 6);
        assert_eq!(output[0], 0.0); // move (not called)
        assert_eq!(output[1], 0.0); // move (not called)
        assert_eq!(output[2], 1.0); // shoot arg
        assert_eq!(output[3], 0.0); // target (not called)
        assert_eq!(output[4], 0.0); // target (not called)
        assert_eq!(output[5], 0.0); // focus (not called)
    }

    #[test]
    fn function_call_translator_decodes_last_function() {
        let space = sample_space();
        let translator = FunctionCallTranslator;

        // Call function 3 ("focus") with arg [5.0]
        let input = vec![3.0, 5.0];
        let output = translator.translate(&input, &space);

        assert_eq!(output.len(), 6);
        assert_eq!(output[0], 0.0); // move
        assert_eq!(output[1], 0.0); // move
        assert_eq!(output[2], 0.0); // shoot
        assert_eq!(output[3], 0.0); // target x
        assert_eq!(output[4], 0.0); // target y
        assert_eq!(output[5], 5.0); // focus arg
    }

    #[test]
    fn function_call_translator_clamps_out_of_range_function_id() {
        let space = sample_space();
        let translator = FunctionCallTranslator;

        // Function id 99 is out of range — should clamp to last function (3)
        let input = vec![99.0, 3.0];
        let output = translator.translate(&input, &space);

        assert_eq!(output.len(), 6);
        assert_eq!(output[5], 3.0); // focus (last function) gets the arg
    }

    #[test]
    fn function_call_translator_handles_empty_input() {
        let space = sample_space();
        let translator = FunctionCallTranslator;

        let output = translator.translate(&[], &space);
        assert_eq!(output, vec![0.0; 6]);
    }

    #[test]
    fn function_call_translator_handles_empty_space() {
        let space = ActionSpace::new(vec![]);
        let translator = FunctionCallTranslator;

        let output = translator.translate(&[0.0, 1.0], &space);
        assert!(output.is_empty());
    }

    #[test]
    fn function_call_translator_input_size() {
        let space = sample_space();
        let translator = FunctionCallTranslator;

        // Max arg flat size is 2 (from "move" or "target"), so input_size = 1 + 2 = 3
        assert_eq!(translator.input_size(&space), 3);
    }

    #[test]
    fn arg_type_serializes_to_json() {
        let spatial = ArgType::Spatial2D {
            width: 64,
            height: 64,
        };
        let json = serde_json::to_value(&spatial).unwrap();
        assert_eq!(json["Spatial2D"]["width"], 64);
        assert_eq!(json["Spatial2D"]["height"], 64);

        let discrete = ArgType::Discrete { n: 5 };
        let json = serde_json::to_value(&discrete).unwrap();
        assert_eq!(json["Discrete"]["n"], 5);

        let continuous = ArgType::Continuous {
            size: 3,
            low: vec![-1.0; 3],
            high: vec![1.0; 3],
        };
        let json = serde_json::to_value(&continuous).unwrap();
        assert_eq!(json["Continuous"]["size"], 3);

        let entity = ArgType::EntitySelect { max: 8 };
        let json = serde_json::to_value(&entity).unwrap();
        assert_eq!(json["EntitySelect"]["max"], 8);
    }

    #[test]
    fn action_fn_serializes_to_json() {
        let func = ActionFn {
            name: "attack".into(),
            id: 0,
            args: vec![
                ArgType::Discrete { n: 2 },
                ArgType::EntitySelect { max: 4 },
            ],
        };
        let json = serde_json::to_value(&func).unwrap();
        assert_eq!(json["name"], "attack");
        assert_eq!(json["id"], 0);
        assert!(json["args"].is_array());
        assert_eq!(json["args"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn action_space_serializes_to_json() {
        let space = sample_space();
        let json = serde_json::to_value(&space).unwrap();
        assert!(json["functions"].is_array());
        assert_eq!(json["functions"].as_array().unwrap().len(), 4);
    }
}

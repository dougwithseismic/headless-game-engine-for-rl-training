use serde::{Deserialize, Serialize};

use crate::error::EngineError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameConfig {
    pub title: String,
    pub tick_rate: u32,
    pub arena: ArenaConfig,
    pub movement: MovementConfig,
    pub combat: CombatConfig,
    pub spawning: SpawningConfig,
    pub teams: TeamsConfig,
    #[serde(default)]
    pub obstacles: Vec<ObstacleConfig>,
    /// Number of simulation ticks per logical step. When > 1, calling
    /// `step_with_mul` or `step_auto` will repeat the last action for this
    /// many ticks internally, accumulating rewards and returning observations
    /// from the final tick. Defaults to 1 if absent from JSON.
    #[serde(default)]
    pub step_mul: Option<u32>,
    /// Mode-specific config extensions (arbitrary JSON for custom scenarios).
    ///
    /// Custom scenarios can read their own configuration from this field
    /// without modifying the core struct. For example, a MOBA scenario could
    /// store `{ "creep_interval": 30 }` here.
    #[serde(default)]
    pub extra: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArenaConfig {
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MovementConfig {
    pub max_speed: f32,
    pub acceleration: f32,
    pub friction: f32,
    #[serde(default = "default_turn_rate")]
    pub turn_rate: f32,
}

fn default_turn_rate() -> f32 {
    32.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombatConfig {
    pub default_weapon: WeaponConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeaponConfig {
    pub damage: f32,
    pub fire_rate: f32,
    pub range: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawningConfig {
    pub respawn_delay: f32,
    #[serde(default = "default_round_time_limit")]
    pub round_time_limit: f32,
}

fn default_round_time_limit() -> f32 {
    15.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamsConfig {
    pub count: u8,
    pub players_per_team: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObstacleConfig {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl std::str::FromStr for GameConfig {
    type Err = EngineError;

    fn from_str(json: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(json).map_err(|e| EngineError::ConfigLoad(e.to_string()))
    }
}

impl GameConfig {
    pub fn from_file(path: &str) -> Result<Self, EngineError> {
        let contents =
            std::fs::read_to_string(path).map_err(|e| EngineError::ConfigLoad(e.to_string()))?;
        contents.parse()
    }

    pub fn extra_f32(&self, key: &str, default: f32) -> f32 {
        self.extra
            .get(key)
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(default)
    }

    pub fn extra_str<'a>(&'a self, key: &str, default: &'a str) -> &'a str {
        self.extra
            .get(key)
            .and_then(|v| v.as_str())
            .unwrap_or(default)
    }

    pub fn extra_usize(&self, key: &str, default: usize) -> usize {
        self.extra
            .get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(default)
    }

    pub fn extra_bool(&self, key: &str, default: bool) -> bool {
        self.extra
            .get(key)
            .and_then(|v| v.as_bool())
            .unwrap_or(default)
    }

    /// Read a JSON array of floats from `extra`, returning `default` if missing or malformed.
    pub fn extra_f32_array<const N: usize>(&self, key: &str, default: [f32; N]) -> [f32; N] {
        self.extra
            .get(key)
            .and_then(|v| v.as_array())
            .and_then(|arr| {
                if arr.len() != N {
                    return None;
                }
                let mut out = [0.0f32; N];
                for (i, val) in arr.iter().enumerate() {
                    out[i] = val.as_f64()? as f32;
                }
                Some(out)
            })
            .unwrap_or(default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_json(extra: &str) -> String {
        format!(
            r#"{{
                "title": "test",
                "tick_rate": 64,
                "arena": {{ "width": 100.0, "height": 100.0 }},
                "movement": {{ "max_speed": 5.0, "acceleration": 100.0, "friction": 50.0 }},
                "combat": {{ "default_weapon": {{ "damage": 10.0, "fire_rate": 0.5, "range": 200.0 }} }},
                "spawning": {{ "respawn_delay": 2.0 }},
                "teams": {{ "count": 2, "players_per_team": 1 }}
                {extra}
            }}"#
        )
    }

    #[test]
    fn step_mul_deserializes_when_present() {
        let json = minimal_json(r#", "step_mul": 4"#);
        let config: GameConfig = json.parse().expect("failed to parse config");
        assert_eq!(config.step_mul, Some(4));
    }

    #[test]
    fn step_mul_defaults_to_none_when_absent() {
        let json = minimal_json("");
        let config: GameConfig = json.parse().expect("failed to parse config");
        assert_eq!(config.step_mul, None);
    }

    #[test]
    fn step_mul_deserializes_as_one() {
        let json = minimal_json(r#", "step_mul": 1"#);
        let config: GameConfig = json.parse().expect("failed to parse config");
        assert_eq!(config.step_mul, Some(1));
    }

    #[test]
    fn step_mul_serializes_roundtrip() {
        let json = minimal_json(r#", "step_mul": 8"#);
        let config: GameConfig = json.parse().expect("failed to parse config");
        let serialized = serde_json::to_string(&config).expect("failed to serialize");
        let deserialized: GameConfig =
            serde_json::from_str(&serialized).expect("failed to deserialize");
        assert_eq!(deserialized.step_mul, Some(8));
    }
}

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
}

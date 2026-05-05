use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("config load failed: {0}")]
    ConfigLoad(String),

    #[error("invalid action: {0}")]
    InvalidAction(String),

    #[error("scenario setup failed: {0}")]
    ScenarioSetup(String),
}

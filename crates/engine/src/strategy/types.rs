use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    pub tick: u64,
    pub scenario: String,
    pub summary: String,
    pub structured: serde_json::Value,
    pub agents: Vec<AgentSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSnapshot {
    pub id: u32,
    pub team: u8,
    pub position: [f32; 3],
    pub health: f32,
    pub alive: bool,
    #[serde(default)]
    pub custom: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentSpec {
    pub name: String,
    pub description: String,
    pub params: Vec<ParamSpec>,
    pub scope: IntentScope,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntentScope {
    Agent,
    Team,
    Global,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamSpec {
    pub name: String,
    pub param_type: ParamType,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParamType {
    Position,
    Float { min: f32, max: f32 },
    Enum(Vec<String>),
    Bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Directive {
    pub intent: String,
    #[serde(default)]
    pub target_agents: Vec<u32>,
    #[serde(default)]
    pub params: serde_json::Value,
    #[serde(default)]
    pub reasoning: Option<String>,
}

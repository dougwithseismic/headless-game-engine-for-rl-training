use super::types::*;

pub trait StrategyProvider: Send + Sync {
    fn decide(&self, snapshot: &StateSnapshot, intents: &[IntentSpec]) -> Vec<Directive>;
    fn decision_interval(&self) -> u64;
    fn name(&self) -> &str;
}

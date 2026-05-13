use bevy_ecs::prelude::*;

use super::types::*;

pub trait StrategyBridge: Send + Sync {
    fn snapshot(&self, world: &mut World) -> StateSnapshot;
    fn available_intents(&self, world: &mut World) -> Vec<IntentSpec>;
    fn apply_directive(&self, world: &mut World, directive: &Directive) -> u32;
}

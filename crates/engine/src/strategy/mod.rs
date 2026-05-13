pub mod bridge;
pub mod provider;
pub mod providers;
pub mod types;

pub use bridge::StrategyBridge;
pub use provider::StrategyProvider;
pub use types::*;

use bevy_ecs::prelude::*;

use crate::ecs::resources::TickState;

#[derive(Resource)]
pub struct StrategyState {
    pub bridge: Box<dyn StrategyBridge>,
    pub providers: Vec<Box<dyn StrategyProvider>>,
    pub inbox: Vec<Directive>,
    pub last_directives: Vec<Directive>,
    pub last_snapshot: Option<StateSnapshot>,
    pub last_intents: Option<Vec<IntentSpec>>,
    provider_timers: Vec<u64>,
}

impl StrategyState {
    pub fn new(
        bridge: Box<dyn StrategyBridge>,
        providers: Vec<Box<dyn StrategyProvider>>,
    ) -> Self {
        let timer_count = providers.len();
        Self {
            bridge,
            providers,
            inbox: Vec::new(),
            last_directives: Vec::new(),
            last_snapshot: None,
            last_intents: None,
            provider_timers: vec![0; timer_count],
        }
    }
}

pub fn strategy_system(world: &mut World) {
    if !world.contains_resource::<StrategyState>() {
        return;
    }

    let tick = world.resource::<TickState>().tick;

    let mut state = world.remove_resource::<StrategyState>().unwrap();

    let mut all_directives: Vec<Directive> = Vec::new();

    let mut needs_snapshot = !state.inbox.is_empty();
    for (i, provider) in state.providers.iter().enumerate() {
        let _ = provider;
        if tick >= state.provider_timers[i] {
            needs_snapshot = true;
        }
    }

    if needs_snapshot {
        let snapshot = state.bridge.snapshot(world);
        let intents = state.bridge.available_intents(world);


        for (i, provider) in state.providers.iter().enumerate() {
            if tick >= state.provider_timers[i] {
                let directives = provider.decide(&snapshot, &intents);
                all_directives.extend(directives);
                state.provider_timers[i] = tick + provider.decision_interval();
            }
        }

        state.last_snapshot = Some(snapshot);
        state.last_intents = Some(intents);
    }

    all_directives.append(&mut state.inbox);

    for directive in &all_directives {
        state.bridge.apply_directive(world, directive);
    }

    state.last_directives = all_directives;

    world.insert_resource(state);
}

use std::sync::mpsc::Receiver;
use std::sync::Mutex;

use crate::strategy::*;

pub struct ChannelProvider {
    receiver: Mutex<Receiver<Directive>>,
}

impl ChannelProvider {
    pub fn new(receiver: Receiver<Directive>) -> Self {
        Self {
            receiver: Mutex::new(receiver),
        }
    }
}

impl StrategyProvider for ChannelProvider {
    fn decide(&self, _snapshot: &StateSnapshot, _intents: &[IntentSpec]) -> Vec<Directive> {
        let rx = self.receiver.lock().unwrap();
        let mut directives = Vec::new();
        while let Ok(d) = rx.try_recv() {
            directives.push(d);
        }
        directives
    }

    fn decision_interval(&self) -> u64 {
        1
    }

    fn name(&self) -> &str {
        "channel"
    }
}

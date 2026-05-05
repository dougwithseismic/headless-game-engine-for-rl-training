use tokio::sync::broadcast;

use ghostlobby_engine::telemetry::TelemetryEvent;

use crate::TelemetrySink;

#[derive(Clone)]
pub struct WsSink {
    sender: broadcast::Sender<String>,
}

impl WsSink {
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self { sender }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<String> {
        self.sender.subscribe()
    }
}

impl TelemetrySink for WsSink {
    fn emit(&self, events: &[TelemetryEvent]) {
        for event in events {
            if let Ok(json) = serde_json::to_string(event) {
                let _ = self.sender.send(json);
            }
        }
    }

    fn flush(&self) {}
}

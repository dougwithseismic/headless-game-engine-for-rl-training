use std::collections::VecDeque;
use std::sync::Mutex;

use ghostlobby_engine::telemetry::TelemetryEvent;

use crate::TelemetrySink;

pub struct BufferSink {
    buffer: Mutex<VecDeque<TelemetryEvent>>,
    capacity: usize,
}

impl BufferSink {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
        }
    }

    pub fn drain(&self) -> Vec<TelemetryEvent> {
        if let Ok(mut buf) = self.buffer.lock() {
            buf.drain(..).collect()
        } else {
            Vec::new()
        }
    }

    pub fn len(&self) -> usize {
        self.buffer.lock().map(|b| b.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl TelemetrySink for BufferSink {
    fn emit(&self, events: &[TelemetryEvent]) {
        if let Ok(mut buf) = self.buffer.lock() {
            for event in events {
                if buf.len() >= self.capacity {
                    buf.pop_front();
                }
                buf.push_back(event.clone());
            }
        }
    }

    fn flush(&self) {}
}

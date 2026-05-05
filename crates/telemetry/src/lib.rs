use ghostlobby_engine::telemetry::TelemetryEvent;

pub mod buffer_sink;
pub mod file_sink;
pub mod ws_sink;

pub trait TelemetrySink: Send + Sync {
    fn emit(&self, events: &[TelemetryEvent]);
    fn flush(&self);
}

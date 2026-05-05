use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::sync::Mutex;

use ghostlobby_engine::telemetry::TelemetryEvent;

use crate::TelemetrySink;

pub struct FileSink {
    writer: Mutex<BufWriter<File>>,
}

impl FileSink {
    pub fn new(path: &str) -> std::io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            writer: Mutex::new(BufWriter::new(file)),
        })
    }
}

impl TelemetrySink for FileSink {
    fn emit(&self, events: &[TelemetryEvent]) {
        if let Ok(mut writer) = self.writer.lock() {
            for event in events {
                if let Ok(json) = serde_json::to_string(event) {
                    let _ = writeln!(writer, "{}", json);
                }
            }
            let _ = writer.flush();
        }
    }

    fn flush(&self) {
        if let Ok(mut writer) = self.writer.lock() {
            let _ = writer.flush();
        }
    }
}

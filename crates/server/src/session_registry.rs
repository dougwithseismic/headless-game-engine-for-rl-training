use std::fs;
use std::path::PathBuf;
use std::process::Command;

use serde::{Deserialize, Serialize};
use tracing::warn;

#[derive(Serialize, Deserialize, Clone)]
pub struct SessionEntry {
    pub pid: u32,
    pub port: u16,
    pub title: String,
    pub config_path: String,
    pub scenario: String,
    pub started_at: String,
}

fn registry_path() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    Some(PathBuf::from(home).join(".ghostlobby").join("sessions.json"))
}

fn read_entries(path: &PathBuf) -> Vec<SessionEntry> {
    fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn write_entries(path: &PathBuf, entries: &[SessionEntry]) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let tmp = path.with_extension("json.tmp");
    if let Ok(data) = serde_json::to_string_pretty(entries) {
        if fs::write(&tmp, data).is_ok() {
            let _ = fs::rename(&tmp, path);
        }
    }
}

fn is_pid_alive(pid: u32) -> bool {
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

pub fn register(entry: &SessionEntry) {
    let Some(path) = registry_path() else { return };
    let mut entries = read_entries(&path);
    entries.retain(|e| is_pid_alive(e.pid));
    entries.retain(|e| e.pid != entry.pid);
    entries.push(entry.clone());
    write_entries(&path, &entries);
}

pub fn unregister(pid: u32) {
    let Some(path) = registry_path() else { return };
    let mut entries = read_entries(&path);
    let before = entries.len();
    entries.retain(|e| e.pid != pid);
    if entries.len() != before {
        write_entries(&path, &entries);
    } else {
        warn!("no session entry found for pid {pid}");
    }
}

use std::fs::File;
use std::io::Write;
use std::path::Path;

use super::ReplayFrame;

/// Serializes and deserializes `ReplayFrame`s to/from JSON Lines format
/// (one JSON object per line). This is a simple interchange format that
/// can be converted to NPZ or other formats in Python.
pub struct ReplayWriter;

impl ReplayWriter {
    /// Write frames to a JSON Lines file (one JSON object per line).
    pub fn write_jsonl(frames: &[ReplayFrame], path: &Path) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        for frame in frames {
            let json = serde_json::to_string(frame)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            writeln!(file, "{}", json)?;
        }
        Ok(())
    }

    /// Read frames from a JSON Lines file.
    pub fn read_jsonl(path: &Path) -> std::io::Result<Vec<ReplayFrame>> {
        let content = std::fs::read_to_string(path)?;
        let mut frames = Vec::new();
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let frame: ReplayFrame = serde_json::from_str(line)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            frames.push(frame);
        }
        Ok(frames)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn make_test_frame(tick: u64) -> ReplayFrame {
        let mut obs = HashMap::new();
        obs.insert(0, {
            let mut m = HashMap::new();
            m.insert("self_state".to_string(), vec![1.0, 2.0, 3.0]);
            m.insert("action_mask".to_string(), vec![1.0, 1.0]);
            m
        });
        obs.insert(1, {
            let mut m = HashMap::new();
            m.insert("self_state".to_string(), vec![4.0, 5.0, 6.0]);
            m
        });

        let mut actions = HashMap::new();
        actions.insert(0, vec![0.5, -0.3]);
        actions.insert(1, vec![1.0, 0.0]);

        let mut rewards = HashMap::new();
        rewards.insert(0, 1.5);
        rewards.insert(1, -0.5);

        let mut dones = HashMap::new();
        dones.insert(0, false);
        dones.insert(1, tick >= 5);

        ReplayFrame {
            tick,
            agent_obs: obs,
            agent_actions: actions,
            agent_rewards: rewards,
            agent_dones: dones,
        }
    }

    #[test]
    fn write_and_read_roundtrip() {
        let frames: Vec<ReplayFrame> = (0..5).map(make_test_frame).collect();

        let dir = std::env::temp_dir().join("ghostlobby_replay_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_roundtrip.jsonl");

        ReplayWriter::write_jsonl(&frames, &path).unwrap();
        let loaded = ReplayWriter::read_jsonl(&path).unwrap();

        assert_eq!(loaded.len(), frames.len());
        for (original, loaded) in frames.iter().zip(loaded.iter()) {
            assert_eq!(original, loaded);
        }

        // Clean up
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn write_and_read_empty_frames() {
        let frames: Vec<ReplayFrame> = vec![];

        let dir = std::env::temp_dir().join("ghostlobby_replay_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_empty.jsonl");

        ReplayWriter::write_jsonl(&frames, &path).unwrap();
        let loaded = ReplayWriter::read_jsonl(&path).unwrap();

        assert_eq!(loaded.len(), 0);

        // Clean up
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }
}

pub mod writer;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A single frame of recorded gameplay data, containing observations,
/// actions, rewards, and done flags for all agents at a given tick.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReplayFrame {
    pub tick: u64,
    pub agent_obs: HashMap<usize, HashMap<String, Vec<f32>>>,
    pub agent_actions: HashMap<usize, Vec<f32>>,
    pub agent_rewards: HashMap<usize, f32>,
    pub agent_dones: HashMap<usize, bool>,
}

/// Records `ReplayFrame`s during gameplay for later use in imitation learning.
///
/// Recording is optional — when not recording, `record_frame` is a no-op
/// so there is zero overhead.
pub struct ReplayRecorder {
    frames: Vec<ReplayFrame>,
    recording: bool,
}

impl ReplayRecorder {
    pub fn new() -> Self {
        Self {
            frames: Vec::new(),
            recording: false,
        }
    }

    /// Start recording. Clears any previously captured frames.
    pub fn start(&mut self) {
        self.recording = true;
        self.frames.clear();
    }

    /// Stop recording. Frames are retained for retrieval.
    pub fn stop(&mut self) {
        self.recording = false;
    }

    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Record a frame. Does nothing if not currently recording.
    pub fn record_frame(&mut self, frame: ReplayFrame) {
        if self.recording {
            self.frames.push(frame);
        }
    }

    pub fn frames(&self) -> &[ReplayFrame] {
        &self.frames
    }

    /// Take all recorded frames, leaving the internal buffer empty.
    pub fn drain_frames(&mut self) -> Vec<ReplayFrame> {
        std::mem::take(&mut self.frames)
    }

    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }
}

impl Default for ReplayRecorder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(tick: u64) -> ReplayFrame {
        let mut obs = HashMap::new();
        obs.insert(0, {
            let mut m = HashMap::new();
            m.insert("self_state".to_string(), vec![1.0, 2.0, 3.0]);
            m
        });

        let mut actions = HashMap::new();
        actions.insert(0, vec![0.5, -0.3]);

        let mut rewards = HashMap::new();
        rewards.insert(0, 1.0);

        let mut dones = HashMap::new();
        dones.insert(0, false);

        ReplayFrame {
            tick,
            agent_obs: obs,
            agent_actions: actions,
            agent_rewards: rewards,
            agent_dones: dones,
        }
    }

    #[test]
    fn start_stop_state_transitions() {
        let mut recorder = ReplayRecorder::new();
        assert!(!recorder.is_recording());

        recorder.start();
        assert!(recorder.is_recording());

        recorder.stop();
        assert!(!recorder.is_recording());
    }

    #[test]
    fn record_frame_only_when_recording() {
        let mut recorder = ReplayRecorder::new();

        // Not recording — frame should be ignored
        recorder.record_frame(make_frame(0));
        assert_eq!(recorder.num_frames(), 0);

        // Start recording — frame should be captured
        recorder.start();
        recorder.record_frame(make_frame(1));
        recorder.record_frame(make_frame(2));
        assert_eq!(recorder.num_frames(), 2);

        // Stop — further frames should be ignored
        recorder.stop();
        recorder.record_frame(make_frame(3));
        assert_eq!(recorder.num_frames(), 2);
    }

    #[test]
    fn drain_frames_returns_and_clears() {
        let mut recorder = ReplayRecorder::new();
        recorder.start();
        recorder.record_frame(make_frame(0));
        recorder.record_frame(make_frame(1));
        recorder.record_frame(make_frame(2));

        let drained = recorder.drain_frames();
        assert_eq!(drained.len(), 3);
        assert_eq!(drained[0].tick, 0);
        assert_eq!(drained[1].tick, 1);
        assert_eq!(drained[2].tick, 2);

        // Buffer should now be empty
        assert_eq!(recorder.num_frames(), 0);
        assert!(recorder.frames().is_empty());
    }

    #[test]
    fn num_frames_counts_correctly() {
        let mut recorder = ReplayRecorder::new();
        assert_eq!(recorder.num_frames(), 0);

        recorder.start();
        for i in 0..5 {
            recorder.record_frame(make_frame(i));
        }
        assert_eq!(recorder.num_frames(), 5);
    }

    #[test]
    fn start_clears_previous_frames() {
        let mut recorder = ReplayRecorder::new();
        recorder.start();
        recorder.record_frame(make_frame(0));
        recorder.record_frame(make_frame(1));
        assert_eq!(recorder.num_frames(), 2);

        // Re-starting should clear
        recorder.start();
        assert_eq!(recorder.num_frames(), 0);
    }
}

use std::collections::HashMap;

use bevy_ecs::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObsFeature {
    pub name: String,
    pub shape: Vec<usize>,
}

#[derive(Resource, Debug, Clone, Serialize, Deserialize)]
pub struct ObservationSpaceDef {
    pub features: Vec<ObsFeature>,
}

pub struct ObsWriter {
    pub buffers: HashMap<String, Vec<f32>>,
}

impl Default for ObsWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl ObsWriter {
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
        }
    }

    pub fn write(&mut self, name: &str, data: &[f32]) {
        self.buffers
            .entry(name.to_string())
            .or_default()
            .extend_from_slice(data);
    }

    pub fn write_padded(&mut self, name: &str, data: &[f32], total_len: usize) {
        let buf = self.buffers.entry(name.to_string()).or_default();
        buf.extend_from_slice(data);
        buf.resize(total_len, 0.0);
    }
}

#[derive(Resource, Debug, Clone, Default)]
pub struct AgentRegistry {
    pub agents: Vec<Entity>,
    pub max_agents: usize,
}

impl AgentRegistry {
    pub fn new(agents: Vec<Entity>, max_agents: usize) -> Self {
        Self { agents, max_agents }
    }

    pub fn index_of(&self, entity: Entity) -> Option<usize> {
        self.agents.iter().position(|&e| e == entity)
    }
}

#[derive(Resource, Debug, Clone, Default)]
pub struct RewardBuffer {
    pub rewards: HashMap<Entity, f32>,
}

impl RewardBuffer {
    pub fn add(&mut self, entity: Entity, amount: f32) {
        *self.rewards.entry(entity).or_default() += amount;
    }

    pub fn get(&self, entity: Entity) -> f32 {
        self.rewards.get(&entity).copied().unwrap_or(0.0)
    }

    pub fn clear(&mut self) {
        self.rewards.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::prelude::World;

    fn spawn_entities(world: &mut World, count: usize) -> Vec<Entity> {
        (0..count).map(|_| world.spawn_empty().id()).collect()
    }

    // ── ObsWriter::write ──

    #[test]
    fn obs_writer_write_stores_data() {
        let mut writer = ObsWriter::new();
        writer.write("pos", &[1.0, 2.0, 3.0]);

        let buf = writer.buffers.get("pos").unwrap();
        assert_eq!(buf, &[1.0, 2.0, 3.0]);
    }

    // ── ObsWriter::write_padded ──

    #[test]
    fn obs_writer_write_padded_pads_with_zeros() {
        let mut writer = ObsWriter::new();
        writer.write_padded("hp", &[100.0, 80.0], 5);

        let buf = writer.buffers.get("hp").unwrap();
        assert_eq!(buf.len(), 5);
        assert_eq!(buf, &[100.0, 80.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn obs_writer_write_padded_exact_length_no_extra() {
        let mut writer = ObsWriter::new();
        writer.write_padded("hp", &[1.0, 2.0, 3.0], 3);

        let buf = writer.buffers.get("hp").unwrap();
        assert_eq!(buf.len(), 3);
        assert_eq!(buf, &[1.0, 2.0, 3.0]);
    }

    // ── ObsWriter::write multiple calls ──

    #[test]
    fn obs_writer_write_accumulates_data() {
        let mut writer = ObsWriter::new();
        writer.write("vel", &[1.0, 2.0]);
        writer.write("vel", &[3.0, 4.0]);

        let buf = writer.buffers.get("vel").unwrap();
        assert_eq!(buf, &[1.0, 2.0, 3.0, 4.0]);
    }

    // ── AgentRegistry::new ──

    #[test]
    fn agent_registry_new_stores_agents_and_max() {
        let mut world = World::new();
        let entities = spawn_entities(&mut world, 3);

        let registry = AgentRegistry::new(entities.clone(), 5);
        assert_eq!(registry.agents.len(), 3);
        assert_eq!(registry.max_agents, 5);
        assert_eq!(registry.agents, entities);
    }

    // ── AgentRegistry::index_of ──

    #[test]
    fn agent_registry_index_of_returns_correct_index() {
        let mut world = World::new();
        let entities = spawn_entities(&mut world, 3);

        let registry = AgentRegistry::new(entities.clone(), 3);
        assert_eq!(registry.index_of(entities[0]), Some(0));
        assert_eq!(registry.index_of(entities[1]), Some(1));
        assert_eq!(registry.index_of(entities[2]), Some(2));
    }

    #[test]
    fn agent_registry_index_of_returns_none_for_missing() {
        let mut world = World::new();
        let entities = spawn_entities(&mut world, 2);
        let missing = world.spawn_empty().id();

        let registry = AgentRegistry::new(entities, 2);
        assert_eq!(registry.index_of(missing), None);
    }

    // ── RewardBuffer::add ──

    #[test]
    fn reward_buffer_add_accumulates() {
        let mut world = World::new();
        let e = world.spawn_empty().id();

        let mut buf = RewardBuffer::default();
        buf.add(e, 1.0);
        buf.add(e, 2.5);

        assert_eq!(buf.get(e), 3.5);
    }

    #[test]
    fn reward_buffer_add_independent_entities() {
        let mut world = World::new();
        let entities = spawn_entities(&mut world, 2);

        let mut buf = RewardBuffer::default();
        buf.add(entities[0], 10.0);
        buf.add(entities[1], 20.0);

        assert_eq!(buf.get(entities[0]), 10.0);
        assert_eq!(buf.get(entities[1]), 20.0);
    }

    // ── RewardBuffer::get ──

    #[test]
    fn reward_buffer_get_returns_zero_for_missing() {
        let mut world = World::new();
        let e = world.spawn_empty().id();

        let buf = RewardBuffer::default();
        assert_eq!(buf.get(e), 0.0);
    }

    // ── RewardBuffer::clear ──

    #[test]
    fn reward_buffer_clear_empties_all() {
        let mut world = World::new();
        let entities = spawn_entities(&mut world, 2);

        let mut buf = RewardBuffer::default();
        buf.add(entities[0], 5.0);
        buf.add(entities[1], 10.0);
        buf.clear();

        assert!(buf.rewards.is_empty());
        assert_eq!(buf.get(entities[0]), 0.0);
        assert_eq!(buf.get(entities[1]), 0.0);
    }

    // ── ObservationSpaceDef serialization round-trip ──

    #[test]
    fn observation_space_def_serialize_roundtrip() {
        let def = ObservationSpaceDef {
            features: vec![
                ObsFeature {
                    name: "position".to_string(),
                    shape: vec![2],
                },
                ObsFeature {
                    name: "enemies".to_string(),
                    shape: vec![5, 4],
                },
            ],
        };

        let json = serde_json::to_string(&def).unwrap();
        let deserialized: ObservationSpaceDef = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.features.len(), 2);
        assert_eq!(deserialized.features[0].name, "position");
        assert_eq!(deserialized.features[0].shape, vec![2]);
        assert_eq!(deserialized.features[1].name, "enemies");
        assert_eq!(deserialized.features[1].shape, vec![5, 4]);
    }
}

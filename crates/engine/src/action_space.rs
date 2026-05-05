use std::collections::HashMap;

use bevy_ecs::prelude::*;
use serde::{Deserialize, Serialize};

pub type ActionDict = Vec<f32>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionHead {
    Continuous {
        name: String,
        size: usize,
        low: Vec<f32>,
        high: Vec<f32>,
    },
    Discrete {
        name: String,
        n: usize,
    },
}

impl ActionHead {
    pub fn name(&self) -> &str {
        match self {
            ActionHead::Continuous { name, .. } => name,
            ActionHead::Discrete { name, .. } => name,
        }
    }

    pub fn flat_size(&self) -> usize {
        match self {
            ActionHead::Continuous { size, .. } => *size,
            ActionHead::Discrete { .. } => 1,
        }
    }
}

#[derive(Resource, Debug, Clone, Serialize, Deserialize)]
pub struct ActionSpaceDef {
    pub heads: Vec<ActionHead>,
    pub total_size: usize,
}

impl ActionSpaceDef {
    pub fn new(heads: Vec<ActionHead>) -> Self {
        let total_size = heads.iter().map(|h| h.flat_size()).sum();
        Self { heads, total_size }
    }

    pub fn head_layout(&self) -> Vec<(usize, usize)> {
        let mut layout = Vec::with_capacity(self.heads.len());
        let mut offset = 0;
        for head in &self.heads {
            let size = head.flat_size();
            layout.push((offset, size));
            offset += size;
        }
        layout
    }

    pub fn extract_head<'a>(&self, action: &'a [f32], head_index: usize) -> &'a [f32] {
        let layout = self.head_layout();
        let (offset, size) = layout[head_index];
        &action[offset..offset + size]
    }
}

#[derive(Resource, Debug, Clone, Default)]
pub struct RawActionBuffer {
    pub actions: HashMap<Entity, ActionDict>,
}

impl RawActionBuffer {
    pub fn insert(&mut self, entity: Entity, action: ActionDict) {
        self.actions.insert(entity, action);
    }

    pub fn get(&self, entity: Entity) -> Option<&ActionDict> {
        self.actions.get(&entity)
    }

    pub fn clear(&mut self) {
        self.actions.clear();
    }
}

#[derive(Resource, Debug, Clone, Default)]
pub struct PendingActions {
    pub actions: HashMap<Entity, ActionDict>,
}

impl PendingActions {
    pub fn insert(&mut self, entity: Entity, action: ActionDict) {
        self.actions.insert(entity, action);
    }

    pub fn drain(&mut self) -> HashMap<Entity, ActionDict> {
        std::mem::take(&mut self.actions)
    }
}

#[derive(Resource, Debug, Clone, Default)]
pub struct ActionMaskBuffer {
    pub masks: HashMap<Entity, Vec<bool>>,
}

impl ActionMaskBuffer {
    pub fn set(&mut self, entity: Entity, mask: Vec<bool>) {
        self.masks.insert(entity, mask);
    }

    pub fn get(&self, entity: Entity) -> Option<&[bool]> {
        self.masks.get(&entity).map(|v| v.as_slice())
    }

    pub fn clear(&mut self) {
        self.masks.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_space_def_new_mixed_heads() {
        let space = ActionSpaceDef::new(vec![
            ActionHead::Continuous {
                name: "move".into(),
                size: 3,
                low: vec![-1.0; 3],
                high: vec![1.0; 3],
            },
            ActionHead::Discrete {
                name: "jump".into(),
                n: 2,
            },
            ActionHead::Continuous {
                name: "aim".into(),
                size: 2,
                low: vec![-1.0; 2],
                high: vec![1.0; 2],
            },
        ]);
        // 3 (continuous) + 1 (discrete) + 2 (continuous) = 6
        assert_eq!(space.total_size, 6);
        assert_eq!(space.heads.len(), 3);
    }

    #[test]
    fn head_layout_offsets_and_sizes() {
        let space = ActionSpaceDef::new(vec![
            ActionHead::Continuous {
                name: "move".into(),
                size: 3,
                low: vec![-1.0; 3],
                high: vec![1.0; 3],
            },
            ActionHead::Discrete {
                name: "jump".into(),
                n: 2,
            },
            ActionHead::Continuous {
                name: "aim".into(),
                size: 2,
                low: vec![-1.0; 2],
                high: vec![1.0; 2],
            },
        ]);
        let layout = space.head_layout();
        assert_eq!(layout.len(), 3);
        assert_eq!(layout[0], (0, 3)); // move: offset 0, size 3
        assert_eq!(layout[1], (3, 1)); // jump: offset 3, size 1
        assert_eq!(layout[2], (4, 2)); // aim:  offset 4, size 2
    }

    #[test]
    fn extract_head_returns_correct_slices() {
        let space = ActionSpaceDef::new(vec![
            ActionHead::Continuous {
                name: "move".into(),
                size: 3,
                low: vec![-1.0; 3],
                high: vec![1.0; 3],
            },
            ActionHead::Discrete {
                name: "jump".into(),
                n: 2,
            },
            ActionHead::Continuous {
                name: "aim".into(),
                size: 2,
                low: vec![-1.0; 2],
                high: vec![1.0; 2],
            },
        ]);
        let action = vec![0.5, -0.3, 0.8, 1.0, 0.1, -0.9];
        assert_eq!(space.extract_head(&action, 0), &[0.5, -0.3, 0.8]);
        assert_eq!(space.extract_head(&action, 1), &[1.0]);
        assert_eq!(space.extract_head(&action, 2), &[0.1, -0.9]);
    }

    #[test]
    fn fps_action_space() {
        // FPS: move_dir(2 continuous), look_angle(1 continuous), shoot(discrete 2)
        let space = ActionSpaceDef::new(vec![
            ActionHead::Continuous {
                name: "move_dir".into(),
                size: 2,
                low: vec![-1.0; 2],
                high: vec![1.0; 2],
            },
            ActionHead::Continuous {
                name: "look_angle".into(),
                size: 1,
                low: vec![-std::f32::consts::PI],
                high: vec![std::f32::consts::PI],
            },
            ActionHead::Discrete {
                name: "shoot".into(),
                n: 2,
            },
        ]);
        // 2 + 1 + 1 = 4
        assert_eq!(space.total_size, 4);

        let action = vec![0.7, -0.2, 1.57, 1.0];
        assert_eq!(space.extract_head(&action, 0), &[0.7, -0.2]); // move_dir
        assert_eq!(space.extract_head(&action, 1), &[1.57]);       // look_angle
        assert_eq!(space.extract_head(&action, 2), &[1.0]);        // shoot
    }

    #[test]
    fn racing_action_space() {
        // Racing: steer(1 continuous), throttle(1 continuous), brake(discrete 2)
        let space = ActionSpaceDef::new(vec![
            ActionHead::Continuous {
                name: "steer".into(),
                size: 1,
                low: vec![-1.0],
                high: vec![1.0],
            },
            ActionHead::Continuous {
                name: "throttle".into(),
                size: 1,
                low: vec![0.0],
                high: vec![1.0],
            },
            ActionHead::Discrete {
                name: "brake".into(),
                n: 2,
            },
        ]);
        // 1 + 1 + 1 = 3
        assert_eq!(space.total_size, 3);
    }

    #[test]
    fn raw_action_buffer_insert_get_clear() {
        let mut buf = RawActionBuffer::default();
        let entity = Entity::from_raw(42);

        assert!(buf.get(entity).is_none());

        buf.insert(entity, vec![1.0, 2.0, 3.0]);
        assert_eq!(buf.get(entity), Some(&vec![1.0, 2.0, 3.0]));

        let entity2 = Entity::from_raw(99);
        buf.insert(entity2, vec![4.0]);
        assert_eq!(buf.get(entity2), Some(&vec![4.0]));

        buf.clear();
        assert!(buf.get(entity).is_none());
        assert!(buf.get(entity2).is_none());
    }

    #[test]
    fn action_mask_buffer_set_get_clear() {
        let mut buf = ActionMaskBuffer::default();
        let entity = Entity::from_raw(7);

        assert!(buf.get(entity).is_none());

        buf.set(entity, vec![true, false, true]);
        assert_eq!(buf.get(entity), Some([true, false, true].as_slice()));

        let entity2 = Entity::from_raw(13);
        buf.set(entity2, vec![false, false]);
        assert_eq!(buf.get(entity2), Some([false, false].as_slice()));

        buf.clear();
        assert!(buf.get(entity).is_none());
        assert!(buf.get(entity2).is_none());
    }

    #[test]
    fn action_head_flat_size() {
        let continuous = ActionHead::Continuous {
            name: "move".into(),
            size: 4,
            low: vec![-1.0; 4],
            high: vec![1.0; 4],
        };
        assert_eq!(continuous.flat_size(), 4);

        let discrete = ActionHead::Discrete {
            name: "fire".into(),
            n: 5,
        };
        assert_eq!(discrete.flat_size(), 1);
    }

    #[test]
    fn action_head_name() {
        let continuous = ActionHead::Continuous {
            name: "velocity".into(),
            size: 3,
            low: vec![-1.0; 3],
            high: vec![1.0; 3],
        };
        assert_eq!(continuous.name(), "velocity");

        let discrete = ActionHead::Discrete {
            name: "weapon_select".into(),
            n: 4,
        };
        assert_eq!(discrete.name(), "weapon_select");
    }
}

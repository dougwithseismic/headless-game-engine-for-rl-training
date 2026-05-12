use bevy_ecs::prelude::*;

use crate::ecs::components::{Agent, Dead, Health, Position, Team, Velocity};
use crate::ecs::resources::WorldBounds;
use crate::features::{FeatureExtractor, ObsShape};

/// Extracts a fixed-length list of other agents' states relative to the observer.
///
/// For each visible agent (up to `max_entities`), writes features_per_entity values:
///   [0] relative x (normalized by arena diagonal)
///   [1] relative y (normalized by arena diagonal)
///   [2] relative vx (normalized by arena diagonal)
///   [3] relative vy (normalized by arena diagonal)
///   [4] hp fraction (0..1)
///   [5] same team (1.0 if same team, 0.0 if different, 0.5 if no team info)
///   [6] alive (1.0 if alive, 0.0 if dead)
///
/// Entities are sorted by distance (nearest first). Remaining slots are zero-padded.
pub struct EntityListExtractor {
    max_entities: usize,
}

/// Features written per entity in the list.
const FEATURES_PER_ENTITY: usize = 7;

impl EntityListExtractor {
    pub fn new(max_entities: usize) -> Self {
        Self { max_entities }
    }
}

impl FeatureExtractor for EntityListExtractor {
    fn name(&self) -> &str {
        "entities"
    }

    fn shape(&self) -> ObsShape {
        ObsShape::EntityList(self.max_entities, FEATURES_PER_ENTITY)
    }

    fn extract(&self, world: &World, agent: Entity, buf: &mut [f32]) {
        let agent_pos = match world.get::<Position>(agent) {
            Some(p) => p.0,
            None => return, // can't compute relative positions without our own position
        };

        let agent_team = world.get::<Team>(agent).map(|t| t.0);
        let diagonal = world
            .get_resource::<WorldBounds>()
            .map(|b| b.diagonal())
            .unwrap_or(1.0);
        let inv_diag = if diagonal > 0.0 {
            1.0 / diagonal
        } else {
            1.0
        };

        // Collect other agents with positions, sorted by distance.
        // We iterate all entities and check for Agent + Position components
        // via individual component access, since we only have &World.
        let mut others: Vec<(Entity, f32)> = Vec::new();

        for entity_ref in world.iter_entities() {
            let entity_id = entity_ref.id();
            if entity_id == agent {
                continue;
            }
            // Only consider entities that are agents with a position
            if entity_ref.get::<Agent>().is_none() {
                continue;
            }
            if let Some(pos) = entity_ref.get::<Position>() {
                let dist = agent_pos.distance(pos.0);
                others.push((entity_id, dist));
            }
        }

        others.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (i, &(entity, _dist)) in others.iter().take(self.max_entities).enumerate() {
            let base = i * FEATURES_PER_ENTITY;
            if base + FEATURES_PER_ENTITY > buf.len() {
                break;
            }

            // Relative position
            if let Some(pos) = world.get::<Position>(entity) {
                let rel = pos.0 - agent_pos;
                buf[base] = rel.x * inv_diag;
                buf[base + 1] = rel.y * inv_diag;
            }

            // Relative velocity
            if let Some(vel) = world.get::<Velocity>(entity) {
                buf[base + 2] = vel.0.x * inv_diag;
                buf[base + 3] = vel.0.y * inv_diag;
            }

            // Health fraction
            if let Some(hp) = world.get::<Health>(entity) {
                buf[base + 4] = if hp.max > 0.0 {
                    hp.current / hp.max
                } else {
                    0.0
                };
            }

            // Same team
            buf[base + 5] = match (agent_team, world.get::<Team>(entity).map(|t| t.0)) {
                (Some(a), Some(b)) => {
                    if a == b {
                        1.0
                    } else {
                        0.0
                    }
                }
                _ => 0.5, // unknown
            };

            // Alive
            buf[base + 6] = if world.get::<Dead>(entity).is_some() {
                0.0
            } else {
                1.0
            };
        }
        // Remaining slots are already zero-initialized
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::components::*;
    use glam::Vec2;

    fn make_world() -> World {
        let mut world = World::new();
        world.insert_resource(WorldBounds {
            width: 100.0,
            height: 100.0,
        });
        world
    }

    #[test]
    fn shape_and_name() {
        let ext = EntityListExtractor::new(4);
        assert_eq!(ext.name(), "entities");
        assert_eq!(ext.shape(), ObsShape::EntityList(4, 7));
        assert_eq!(ext.shape().flat_size(), 28);
    }

    #[test]
    fn extracts_other_agents_relative_to_observer() {
        let mut world = make_world();
        let diagonal = WorldBounds {
            width: 100.0,
            height: 100.0,
        }
        .diagonal();

        let observer = world
            .spawn((
                Agent { source_id: 0 },
                Position(Vec2::new(10.0, 10.0)),
                Team(0),
            ))
            .id();

        world.spawn((
            Agent { source_id: 1 },
            Position(Vec2::new(30.0, 10.0)),
            Health {
                current: 50.0,
                max: 100.0,
            },
            Team(1),
        ));

        let ext = EntityListExtractor::new(4);
        let mut buf = vec![0.0; 28];
        ext.extract(&world, observer, &mut buf);

        // Relative x: (30 - 10) / diagonal = 20 / diagonal
        let expected_rx = 20.0 / diagonal;
        assert!((buf[0] - expected_rx).abs() < 1e-4);
        // Relative y: 0
        assert!((buf[1]).abs() < 1e-5);
        // Health: 50/100 = 0.5
        assert!((buf[4] - 0.5).abs() < 1e-5);
        // Different team
        assert!((buf[5]).abs() < 1e-5);
        // Alive
        assert!((buf[6] - 1.0).abs() < 1e-5);

        // Second slot should be all zeros (only 1 other agent)
        for i in 7..14 {
            assert_eq!(buf[i], 0.0, "buf[{}] should be 0.0 (padding)", i);
        }
    }

    #[test]
    fn sorted_by_distance_nearest_first() {
        let mut world = make_world();

        let observer = world
            .spawn((
                Agent { source_id: 0 },
                Position(Vec2::new(0.0, 0.0)),
            ))
            .id();

        // Far agent (index 1)
        world.spawn((
            Agent { source_id: 1 },
            Position(Vec2::new(100.0, 0.0)),
        ));

        // Near agent (index 2)
        world.spawn((
            Agent { source_id: 2 },
            Position(Vec2::new(10.0, 0.0)),
        ));

        let ext = EntityListExtractor::new(4);
        let mut buf = vec![0.0; 28];
        ext.extract(&world, observer, &mut buf);

        let diagonal = WorldBounds {
            width: 100.0,
            height: 100.0,
        }
        .diagonal();

        // First slot should be the nearer agent (x=10)
        let expected_near = 10.0 / diagonal;
        let expected_far = 100.0 / diagonal;
        assert!(
            (buf[0] - expected_near).abs() < 1e-4,
            "nearest agent should be first, got {}",
            buf[0]
        );
        assert!(
            (buf[7] - expected_far).abs() < 1e-4,
            "farthest agent should be second, got {}",
            buf[7]
        );
    }

    #[test]
    fn zero_pads_when_fewer_than_max() {
        let mut world = make_world();

        let observer = world
            .spawn((
                Agent { source_id: 0 },
                Position(Vec2::new(0.0, 0.0)),
            ))
            .id();

        // Only 1 other agent
        world.spawn((
            Agent { source_id: 1 },
            Position(Vec2::new(10.0, 0.0)),
        ));

        let ext = EntityListExtractor::new(3);
        let mut buf = vec![0.0; 21]; // 3 * 7
        ext.extract(&world, observer, &mut buf);

        // Slots 2 and 3 should be all zeros
        for i in 7..21 {
            assert_eq!(buf[i], 0.0, "buf[{}] should be zero-padded", i);
        }
    }

    #[test]
    fn no_position_on_observer_returns_zeros() {
        let mut world = make_world();
        let observer = world.spawn(Agent { source_id: 0 }).id();

        world.spawn((
            Agent { source_id: 1 },
            Position(Vec2::new(10.0, 0.0)),
        ));

        let ext = EntityListExtractor::new(2);
        let mut buf = vec![0.0; 14];
        ext.extract(&world, observer, &mut buf);

        for v in &buf {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn dead_agent_shows_zero_alive() {
        let mut world = make_world();

        let observer = world
            .spawn((
                Agent { source_id: 0 },
                Position(Vec2::new(0.0, 0.0)),
            ))
            .id();

        world.spawn((
            Agent { source_id: 1 },
            Position(Vec2::new(10.0, 0.0)),
            Dead,
        ));

        let ext = EntityListExtractor::new(2);
        let mut buf = vec![0.0; 14];
        ext.extract(&world, observer, &mut buf);

        assert_eq!(buf[6], 0.0); // alive = 0 for dead agent
    }

    #[test]
    fn same_team_detected() {
        let mut world = make_world();

        let observer = world
            .spawn((
                Agent { source_id: 0 },
                Position(Vec2::new(0.0, 0.0)),
                Team(1),
            ))
            .id();

        world.spawn((
            Agent { source_id: 1 },
            Position(Vec2::new(10.0, 0.0)),
            Team(1),
        ));

        let ext = EntityListExtractor::new(2);
        let mut buf = vec![0.0; 14];
        ext.extract(&world, observer, &mut buf);

        assert!((buf[5] - 1.0).abs() < 1e-5); // same team
    }
}

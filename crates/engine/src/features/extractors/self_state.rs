use bevy_ecs::prelude::*;

use crate::ecs::components::{Dead, Facing, Health, Position, Velocity, Weapon};
use crate::ecs::resources::WorldBounds;
use crate::features::{FeatureExtractor, ObsShape};

/// Extracts the observing agent's own state: position, velocity, facing, health,
/// weapon cooldown, and alive status.
///
/// Output layout (Vector(9)):
///   [0] x / arena_width   (normalized 0..1)
///   [1] y / arena_height  (normalized 0..1)
///   [2] vx / max_speed    (normalized -1..1)
///   [3] vy / max_speed    (normalized -1..1)
///   [4] facing / PI       (normalized -1..1)
///   [5] hp / max_hp        (0..1)
///   [6] weapon cooldown fraction (0..1)
///   [7] alive (1.0 if alive, 0.0 if dead)
///   [8] max_speed (raw, for downstream use)
///
/// Missing components yield 0.0. If no WorldBounds resource exists, position is
/// not normalized (raw values written).
pub struct SelfStateExtractor {
    max_speed: f32,
}

impl SelfStateExtractor {
    /// `max_speed` is used for velocity normalization.
    pub fn new(max_speed: f32) -> Self {
        Self { max_speed }
    }
}

const SELF_STATE_SIZE: usize = 9;

impl FeatureExtractor for SelfStateExtractor {
    fn name(&self) -> &str {
        "self_state"
    }

    fn shape(&self) -> ObsShape {
        ObsShape::Vector(SELF_STATE_SIZE)
    }

    fn extract(&self, world: &World, agent: Entity, buf: &mut [f32]) {
        debug_assert!(buf.len() >= SELF_STATE_SIZE);

        let bounds = world.get_resource::<WorldBounds>();

        // Position (normalized by arena bounds if available)
        if let Some(pos) = world.get::<Position>(agent) {
            if let Some(b) = bounds {
                buf[0] = if b.width > 0.0 {
                    pos.0.x / b.width
                } else {
                    pos.0.x
                };
                buf[1] = if b.height > 0.0 {
                    pos.0.y / b.height
                } else {
                    pos.0.y
                };
            } else {
                buf[0] = pos.0.x;
                buf[1] = pos.0.y;
            }
        }

        // Velocity (normalized by max_speed)
        if let Some(vel) = world.get::<Velocity>(agent) {
            if self.max_speed > 0.0 {
                buf[2] = vel.0.x / self.max_speed;
                buf[3] = vel.0.y / self.max_speed;
            } else {
                buf[2] = vel.0.x;
                buf[3] = vel.0.y;
            }
        }

        // Facing (normalized by PI)
        if let Some(facing) = world.get::<Facing>(agent) {
            buf[4] = facing.0 / std::f32::consts::PI;
        }

        // Health (fraction)
        if let Some(hp) = world.get::<Health>(agent) {
            buf[5] = if hp.max > 0.0 {
                hp.current / hp.max
            } else {
                0.0
            };
        }

        // Weapon cooldown fraction
        if let Some(weapon) = world.get::<Weapon>(agent) {
            buf[6] = if weapon.fire_rate > 0.0 {
                weapon.cooldown_remaining / weapon.fire_rate
            } else {
                0.0
            };
        }

        // Alive status
        buf[7] = if world.get::<Dead>(agent).is_some() {
            0.0
        } else {
            1.0
        };

        // Max speed (raw value, useful for policy normalization)
        buf[8] = self.max_speed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::components::*;
    use crate::ecs::resources::WorldBounds;
    use glam::Vec2;

    fn make_world_with_bounds() -> World {
        let mut world = World::new();
        world.insert_resource(WorldBounds {
            width: 100.0,
            height: 200.0,
        });
        world
    }

    #[test]
    fn extracts_full_agent_state() {
        let mut world = make_world_with_bounds();
        let agent = world
            .spawn((
                Position(Vec2::new(50.0, 100.0)),
                Velocity(Vec2::new(5.0, -5.0)),
                Facing(std::f32::consts::FRAC_PI_2),
                Health {
                    current: 75.0,
                    max: 100.0,
                },
                Weapon {
                    damage: 10.0,
                    fire_rate: 0.5,
                    range: 100.0,
                    cooldown_remaining: 0.25,
                },
            ))
            .id();

        let ext = SelfStateExtractor::new(10.0);
        let mut buf = vec![0.0; 9];
        ext.extract(&world, agent, &mut buf);

        // Position normalized: 50/100 = 0.5, 100/200 = 0.5
        assert!((buf[0] - 0.5).abs() < 1e-5);
        assert!((buf[1] - 0.5).abs() < 1e-5);
        // Velocity normalized: 5/10 = 0.5, -5/10 = -0.5
        assert!((buf[2] - 0.5).abs() < 1e-5);
        assert!((buf[3] - (-0.5)).abs() < 1e-5);
        // Facing: (PI/2) / PI = 0.5
        assert!((buf[4] - 0.5).abs() < 1e-5);
        // Health: 75/100 = 0.75
        assert!((buf[5] - 0.75).abs() < 1e-5);
        // Weapon cooldown: 0.25/0.5 = 0.5
        assert!((buf[6] - 0.5).abs() < 1e-5);
        // Alive
        assert!((buf[7] - 1.0).abs() < 1e-5);
        // Max speed
        assert!((buf[8] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn missing_components_return_zeros() {
        let mut world = make_world_with_bounds();
        let agent = world.spawn_empty().id();

        let ext = SelfStateExtractor::new(10.0);
        let mut buf = vec![0.0; 9];
        ext.extract(&world, agent, &mut buf);

        // Position, velocity, facing, health, weapon cooldown all 0.0
        for i in 0..7 {
            assert_eq!(buf[i], 0.0, "buf[{}] should be 0.0", i);
        }
        // Alive is 1.0 (no Dead component)
        assert_eq!(buf[7], 1.0);
        // Max speed
        assert_eq!(buf[8], 10.0);
    }

    #[test]
    fn dead_agent_returns_zero_alive() {
        let mut world = make_world_with_bounds();
        let agent = world.spawn(Dead).id();

        let ext = SelfStateExtractor::new(10.0);
        let mut buf = vec![0.0; 9];
        ext.extract(&world, agent, &mut buf);

        assert_eq!(buf[7], 0.0);
    }

    #[test]
    fn no_world_bounds_uses_raw_position() {
        let mut world = World::new();
        let agent = world
            .spawn(Position(Vec2::new(42.0, 99.0)))
            .id();

        let ext = SelfStateExtractor::new(10.0);
        let mut buf = vec![0.0; 9];
        ext.extract(&world, agent, &mut buf);

        assert!((buf[0] - 42.0).abs() < 1e-5);
        assert!((buf[1] - 99.0).abs() < 1e-5);
    }

    #[test]
    fn shape_and_name() {
        let ext = SelfStateExtractor::new(10.0);
        assert_eq!(ext.name(), "self_state");
        assert_eq!(ext.shape(), ObsShape::Vector(9));
        assert_eq!(ext.shape().flat_size(), 9);
    }

    #[test]
    fn zero_max_speed_uses_raw_velocity() {
        let mut world = make_world_with_bounds();
        let agent = world
            .spawn((
                Position(Vec2::ZERO),
                Velocity(Vec2::new(3.0, 4.0)),
            ))
            .id();

        let ext = SelfStateExtractor::new(0.0);
        let mut buf = vec![0.0; 9];
        ext.extract(&world, agent, &mut buf);

        assert!((buf[2] - 3.0).abs() < 1e-5);
        assert!((buf[3] - 4.0).abs() < 1e-5);
    }
}

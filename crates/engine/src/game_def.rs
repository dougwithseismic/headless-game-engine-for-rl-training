use bevy_ecs::prelude::*;

use crate::config::GameConfig;

/// How the simulation advances time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameTickMode {
    /// Continuous simulation — tick as fast as possible or at a fixed rate.
    RealTime,
    /// Discrete turns — agents submit actions, then one tick resolves.
    TurnBased,
}

impl Default for GameTickMode {
    fn default() -> Self {
        Self::RealTime
    }
}

/// Defines a game's setup and systems without coupling to observation/reward.
///
/// `GameDef` is the successor to `Scenario` — it handles only game rules and
/// ECS setup. Observation extraction, reward computation, and action-space
/// definitions will be composed separately (Phases 2-6).
///
/// Physics (if needed) is inserted as a World resource *before* `setup()` is
/// called, so implementations can access it via `world.resource_mut::<PhysicsState>()`.
pub trait GameDef: Send + Sync {
    /// Human-readable name for this game definition.
    fn name(&self) -> &str;

    /// Whether the game runs in real-time or turn-based mode.
    fn tick_mode(&self) -> GameTickMode {
        GameTickMode::RealTime
    }

    /// Whether this game needs a 2D physics world.
    /// When true, the engine creates and inserts `PhysicsState` before `setup()`.
    fn needs_physics(&self) -> bool {
        true
    }

    /// Set up ECS world state (spawn entities, insert resources, etc.).
    /// If `needs_physics()` returns true, `PhysicsState` is already available
    /// as a world resource.
    fn setup(&self, world: &mut World, config: &GameConfig);

    /// Register game-specific systems into the tick schedule.
    fn register_systems(&self, schedule: &mut Schedule);

    /// Whether the episode is done for a given agent.
    fn is_done(&self, _world: &World, _agent: Entity) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubGameDef;

    impl GameDef for StubGameDef {
        fn name(&self) -> &str {
            "stub"
        }

        fn setup(&self, _world: &mut World, _config: &GameConfig) {}

        fn register_systems(&self, _schedule: &mut Schedule) {}
    }

    #[test]
    fn default_tick_mode_is_realtime() {
        let def = StubGameDef;
        assert_eq!(def.tick_mode(), GameTickMode::RealTime);
    }

    #[test]
    fn default_needs_physics_is_true() {
        let def = StubGameDef;
        assert!(def.needs_physics());
    }

    #[test]
    fn default_is_done_returns_false() {
        let def = StubGameDef;
        let mut world = World::new();
        let entity = world.spawn_empty().id();
        assert!(!def.is_done(&world, entity));
    }

    struct NoPhysicsGameDef;

    impl GameDef for NoPhysicsGameDef {
        fn name(&self) -> &str {
            "no-physics"
        }

        fn needs_physics(&self) -> bool {
            false
        }

        fn setup(&self, _world: &mut World, _config: &GameConfig) {}

        fn register_systems(&self, _schedule: &mut Schedule) {}
    }

    #[test]
    fn can_opt_out_of_physics() {
        let def = NoPhysicsGameDef;
        assert!(!def.needs_physics());
    }

    struct TurnBasedGameDef;

    impl GameDef for TurnBasedGameDef {
        fn name(&self) -> &str {
            "turn-based"
        }

        fn tick_mode(&self) -> GameTickMode {
            GameTickMode::TurnBased
        }

        fn setup(&self, _world: &mut World, _config: &GameConfig) {}

        fn register_systems(&self, _schedule: &mut Schedule) {}
    }

    #[test]
    fn can_set_turn_based_mode() {
        let def = TurnBasedGameDef;
        assert_eq!(def.tick_mode(), GameTickMode::TurnBased);
    }

    #[test]
    fn game_tick_mode_default_is_realtime() {
        assert_eq!(GameTickMode::default(), GameTickMode::RealTime);
    }
}

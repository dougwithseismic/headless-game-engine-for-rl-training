use glam::Vec2;
use rapier2d::prelude::ColliderHandle;
use std::collections::HashMap;

use crate::physics::PhysicsState;

/// Result of a single sensor ray cast from an agent's position.
///
/// Used by RL agents to perceive their surroundings. Each ray reports a
/// normalized distance and a classification of what (if anything) was hit.
#[derive(Debug, Clone, Copy)]
pub struct RayResult {
    /// Distance to the nearest hit, normalized to `[0.0, 1.0]`.
    /// `1.0` means nothing was hit within `max_range`.
    pub distance: f32,
    /// Classification of the hit object:
    /// - `0.0`  -- nothing hit
    /// - `0.33` -- wall / obstacle / same-team agent
    /// - `0.66` -- enemy agent (different team)
    pub hit_type: f32,
}

/// Hit-type constant: nothing was hit.
pub const HIT_NONE: f32 = 0.0;
/// Hit-type constant: wall, obstacle, or same-team agent.
pub const HIT_WALL: f32 = 0.33;
/// Hit-type constant: enemy agent (different team).
pub const HIT_ENEMY: f32 = 0.66;

/// Cast `num_rays` evenly-spaced sensor rays in a full 360-degree sweep from `origin`.
///
/// Each ray is tested against the physics world via [`PhysicsState::cast_ray`],
/// excluding `self_collider` so the caster never detects itself.
///
/// # Arguments
///
/// * `physics`         -- The current physics state (provides the query pipeline).
/// * `origin`          -- World-space position to cast from.
/// * `self_collider`   -- The caster's own collider handle, excluded from results.
/// * `num_rays`        -- How many rays to cast (evenly distributed over 360 degrees).
/// * `max_range`       -- Maximum ray distance in world units.
/// * `collider_types`  -- Maps collider handles to hit-type values. Colliders not
///   present in this map default to [`HIT_WALL`].
///
/// # Returns
///
/// A `Vec<RayResult>` of length `num_rays`, ordered by increasing angle starting
/// from the +X axis.
pub fn cast_sensor_rays(
    physics: &PhysicsState,
    origin: Vec2,
    self_collider: ColliderHandle,
    num_rays: usize,
    max_range: f32,
    collider_types: &HashMap<ColliderHandle, f32>,
) -> Vec<RayResult> {
    let angle_step = std::f32::consts::TAU / num_rays as f32;

    (0..num_rays)
        .map(|i| {
            let angle = i as f32 * angle_step;
            let direction = Vec2::new(angle.cos(), angle.sin());

            match physics.cast_ray(origin, direction, max_range, Some(self_collider)) {
                Some((collider_handle, toi)) => {
                    let distance = (toi / max_range).clamp(0.0, 1.0);
                    let hit_type = collider_types
                        .get(&collider_handle)
                        .copied()
                        .unwrap_or(HIT_WALL);
                    RayResult { distance, hit_type }
                }
                None => RayResult {
                    distance: 1.0,
                    hit_type: HIT_NONE,
                },
            }
        })
        .collect()
}

/// Build a collider-type classification map from pre-queried obstacle and agent data.
///
/// This is a pure helper that keeps [`cast_sensor_rays`] free from ECS dependencies.
/// The caller is responsible for querying the ECS world and passing the relevant
/// collider handles.
///
/// # Classification rules
///
/// | Collider source         | Hit-type value |
/// |-------------------------|----------------|
/// | Obstacle                | `0.33` (wall)  |
/// | Agent on the same team  | `0.33` (wall)  |
/// | Agent on a different team | `0.66` (enemy) |
///
/// # Arguments
///
/// * `obstacles`  -- Collider handles of all obstacle entities.
/// * `agents`     -- `(ColliderHandle, team_id)` pairs for every agent.
/// * `self_team`  -- The querying agent's team, used to distinguish friend from foe.
pub fn build_collider_type_map(
    obstacles: &[ColliderHandle],
    agents: &[(ColliderHandle, u8)],
    self_team: u8,
) -> HashMap<ColliderHandle, f32> {
    let mut map = HashMap::with_capacity(obstacles.len() + agents.len());

    for &handle in obstacles {
        map.insert(handle, HIT_WALL);
    }

    for &(handle, team) in agents {
        if team == self_team {
            map.insert(handle, HIT_WALL);
        } else {
            map.insert(handle, HIT_ENEMY);
        }
    }

    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    const DT: f32 = 1.0 / 60.0;

    /// Create a zero-gravity physics world.
    fn empty_physics() -> PhysicsState {
        PhysicsState::new(Vec2::ZERO, DT)
    }

    // ========================================================================
    // cast_sensor_rays -- basic properties
    // ========================================================================

    #[test]
    fn returns_exactly_num_rays_results() {
        let mut physics = empty_physics();
        let (_, self_ch) = physics.add_dynamic_body(Vec2::ZERO, 0.5);
        physics.query_pipeline.update(&physics.collider_set);

        for num_rays in [1, 8, 16, 64, 128] {
            let results = cast_sensor_rays(
                &physics,
                Vec2::ZERO,
                self_ch,
                num_rays,
                100.0,
                &HashMap::new(),
            );
            assert_eq!(
                results.len(),
                num_rays,
                "expected {num_rays} results, got {}",
                results.len()
            );
        }
    }

    // ========================================================================
    // cast_sensor_rays -- empty world (no obstacles)
    // ========================================================================

    #[test]
    fn empty_world_all_rays_miss() {
        let mut physics = empty_physics();
        let (_, self_ch) = physics.add_dynamic_body(Vec2::ZERO, 0.5);
        physics.query_pipeline.update(&physics.collider_set);

        let results = cast_sensor_rays(
            &physics,
            Vec2::ZERO,
            self_ch,
            64,
            100.0,
            &HashMap::new(),
        );

        for (i, r) in results.iter().enumerate() {
            assert!(
                (r.distance - 1.0).abs() < f32::EPSILON,
                "ray {i}: expected distance=1.0, got {}",
                r.distance
            );
            assert!(
                (r.hit_type - HIT_NONE).abs() < f32::EPSILON,
                "ray {i}: expected hit_type=0.0, got {}",
                r.hit_type
            );
        }
    }

    // ========================================================================
    // cast_sensor_rays -- self collider excluded
    // ========================================================================

    #[test]
    fn self_collider_is_excluded() {
        // Only body in the world is the agent itself.
        // All rays should miss (distance=1.0) because the self collider is excluded.
        let mut physics = empty_physics();
        let (_, self_ch) = physics.add_dynamic_body(Vec2::ZERO, 0.5);
        physics.query_pipeline.update(&physics.collider_set);

        let results = cast_sensor_rays(
            &physics,
            Vec2::ZERO,
            self_ch,
            64,
            100.0,
            &HashMap::new(),
        );

        for (i, r) in results.iter().enumerate() {
            assert!(
                (r.distance - 1.0).abs() < f32::EPSILON,
                "ray {i}: self-collider should be excluded, but got distance={}",
                r.distance
            );
        }
    }

    // ========================================================================
    // cast_sensor_rays -- wall hit detection
    // ========================================================================

    #[test]
    fn ray_hits_wall_with_correct_distance_and_type() {
        let mut physics = empty_physics();

        // Agent at origin
        let (_, self_ch) = physics.add_dynamic_body(Vec2::ZERO, 0.25);

        // Place a wall at x=10, half-extent 1.0 in x => near edge at x=9.0
        let (_, wall_ch) = physics.add_static_body(Vec2::new(10.0, 0.0), Vec2::new(1.0, 5.0));

        physics.query_pipeline.update(&physics.collider_set);

        let mut collider_types = HashMap::new();
        collider_types.insert(wall_ch, HIT_WALL);

        let max_range = 100.0;

        // Use 4 rays: 0=+X, 1=+Y, 2=-X, 3=-Y
        let results = cast_sensor_rays(
            &physics,
            Vec2::ZERO,
            self_ch,
            4,
            max_range,
            &collider_types,
        );

        // Ray 0 (+X direction) should hit the wall
        let ray_px = &results[0];
        let expected_dist = 9.0 / max_range; // wall edge at 9.0, max_range 100
        assert!(
            (ray_px.distance - expected_dist).abs() < 0.01,
            "ray +X: expected distance ~{expected_dist}, got {}",
            ray_px.distance
        );
        assert!(
            (ray_px.hit_type - HIT_WALL).abs() < f32::EPSILON,
            "ray +X: expected hit_type={HIT_WALL}, got {}",
            ray_px.hit_type
        );

        // Other 3 rays should miss
        for i in 1..4 {
            assert!(
                (results[i].distance - 1.0).abs() < f32::EPSILON,
                "ray {i}: expected miss (distance=1.0), got {}",
                results[i].distance
            );
        }
    }

    // ========================================================================
    // cast_sensor_rays -- enemy hit detection
    // ========================================================================

    #[test]
    fn ray_hits_enemy_with_correct_type() {
        let mut physics = empty_physics();

        // Agent at origin
        let (_, self_ch) = physics.add_dynamic_body(Vec2::ZERO, 0.25);

        // Enemy agent at (5.0, 0.0), radius 0.5 => near edge at x=4.5
        let (_, enemy_ch) = physics.add_dynamic_body(Vec2::new(5.0, 0.0), 0.5);

        physics.query_pipeline.update(&physics.collider_set);

        let mut collider_types = HashMap::new();
        collider_types.insert(enemy_ch, HIT_ENEMY);

        let results = cast_sensor_rays(
            &physics,
            Vec2::ZERO,
            self_ch,
            4,
            100.0,
            &collider_types,
        );

        // Ray 0 (+X direction) should hit the enemy
        assert!(
            (results[0].hit_type - HIT_ENEMY).abs() < f32::EPSILON,
            "ray +X: expected hit_type={HIT_ENEMY}, got {}",
            results[0].hit_type
        );
        assert!(
            results[0].distance < 1.0,
            "ray +X: should have hit something (distance < 1.0), got {}",
            results[0].distance
        );
    }

    // ========================================================================
    // cast_sensor_rays -- unknown collider defaults to HIT_WALL
    // ========================================================================

    #[test]
    fn unknown_collider_defaults_to_wall() {
        let mut physics = empty_physics();

        let (_, self_ch) = physics.add_dynamic_body(Vec2::ZERO, 0.25);
        // Object not registered in collider_types
        physics.add_static_body(Vec2::new(5.0, 0.0), Vec2::new(1.0, 5.0));

        physics.query_pipeline.update(&physics.collider_set);

        // Empty collider_types map -- the wall's handle is NOT in it
        let results = cast_sensor_rays(
            &physics,
            Vec2::ZERO,
            self_ch,
            4,
            100.0,
            &HashMap::new(),
        );

        // Ray 0 (+X) should hit the unknown object and default to HIT_WALL
        assert!(
            results[0].distance < 1.0,
            "ray +X should hit something"
        );
        assert!(
            (results[0].hit_type - HIT_WALL).abs() < f32::EPSILON,
            "unknown collider should default to HIT_WALL, got {}",
            results[0].hit_type
        );
    }

    // ========================================================================
    // cast_sensor_rays -- distance normalization and clamping
    // ========================================================================

    #[test]
    fn distance_is_normalized_to_max_range() {
        let mut physics = empty_physics();

        let (_, self_ch) = physics.add_dynamic_body(Vec2::ZERO, 0.25);
        // Wall edge at x=9.0 (center 10, half-extent 1)
        let (_, wall_ch) = physics.add_static_body(Vec2::new(10.0, 0.0), Vec2::new(1.0, 5.0));

        physics.query_pipeline.update(&physics.collider_set);

        let mut collider_types = HashMap::new();
        collider_types.insert(wall_ch, HIT_WALL);

        // With max_range=10, wall at 9.0 => normalized distance = 9.0/10.0 = 0.9
        let results_10 = cast_sensor_rays(
            &physics,
            Vec2::ZERO,
            self_ch,
            4,
            10.0,
            &collider_types,
        );
        assert!(
            (results_10[0].distance - 0.9).abs() < 0.02,
            "with max_range=10, expected ~0.9, got {}",
            results_10[0].distance
        );

        // With max_range=50, wall at 9.0 => normalized distance = 9.0/50.0 = 0.18
        let results_50 = cast_sensor_rays(
            &physics,
            Vec2::ZERO,
            self_ch,
            4,
            50.0,
            &collider_types,
        );
        assert!(
            (results_50[0].distance - 0.18).abs() < 0.02,
            "with max_range=50, expected ~0.18, got {}",
            results_50[0].distance
        );
    }

    // ========================================================================
    // cast_sensor_rays -- multiple objects at different angles
    // ========================================================================

    #[test]
    fn multiple_objects_at_different_angles() {
        let mut physics = empty_physics();

        let (_, self_ch) = physics.add_dynamic_body(Vec2::ZERO, 0.25);

        // Wall in +X direction at x=10
        let (_, wall_ch) = physics.add_static_body(Vec2::new(10.0, 0.0), Vec2::new(1.0, 5.0));
        // Wall in +Y direction at y=20
        let (_, wall2_ch) = physics.add_static_body(Vec2::new(0.0, 20.0), Vec2::new(5.0, 1.0));

        physics.query_pipeline.update(&physics.collider_set);

        let mut collider_types = HashMap::new();
        collider_types.insert(wall_ch, HIT_WALL);
        collider_types.insert(wall2_ch, HIT_ENEMY); // classify as enemy for testing

        // 4 rays: 0=+X, 1=+Y, 2=-X, 3=-Y
        let results = cast_sensor_rays(
            &physics,
            Vec2::ZERO,
            self_ch,
            4,
            100.0,
            &collider_types,
        );

        // Ray 0 (+X) should hit wall
        assert!(results[0].distance < 1.0, "+X ray should hit wall");
        assert!(
            (results[0].hit_type - HIT_WALL).abs() < f32::EPSILON,
            "+X should be wall"
        );

        // Ray 1 (+Y) should hit "enemy" (we labelled the +Y wall as enemy)
        assert!(results[1].distance < 1.0, "+Y ray should hit something");
        assert!(
            (results[1].hit_type - HIT_ENEMY).abs() < f32::EPSILON,
            "+Y should be enemy"
        );

        // Rays 2 (-X) and 3 (-Y) should miss
        assert!(
            (results[2].distance - 1.0).abs() < f32::EPSILON,
            "-X should miss"
        );
        assert!(
            (results[3].distance - 1.0).abs() < f32::EPSILON,
            "-Y should miss"
        );
    }

    // ========================================================================
    // build_collider_type_map tests
    // ========================================================================

    #[test]
    fn build_map_classifies_obstacles_as_wall() {
        let h1 = ColliderHandle::from_raw_parts(1, 0);
        let h2 = ColliderHandle::from_raw_parts(2, 0);

        let map = build_collider_type_map(&[h1, h2], &[], 0);

        assert_eq!(map.len(), 2);
        assert!((map[&h1] - HIT_WALL).abs() < f32::EPSILON);
        assert!((map[&h2] - HIT_WALL).abs() < f32::EPSILON);
    }

    #[test]
    fn build_map_classifies_enemies_as_enemy() {
        let enemy_handle = ColliderHandle::from_raw_parts(5, 0);

        // self_team = 0, enemy on team 1
        let map = build_collider_type_map(&[], &[(enemy_handle, 1)], 0);

        assert_eq!(map.len(), 1);
        assert!(
            (map[&enemy_handle] - HIT_ENEMY).abs() < f32::EPSILON,
            "enemy agent should be classified as HIT_ENEMY"
        );
    }

    #[test]
    fn build_map_classifies_same_team_as_wall() {
        let ally_handle = ColliderHandle::from_raw_parts(3, 0);

        // self_team = 1, ally also on team 1
        let map = build_collider_type_map(&[], &[(ally_handle, 1)], 1);

        assert_eq!(map.len(), 1);
        assert!(
            (map[&ally_handle] - HIT_WALL).abs() < f32::EPSILON,
            "same-team agent should be classified as HIT_WALL"
        );
    }

    #[test]
    fn build_map_mixed_obstacles_and_agents() {
        let obs_h = ColliderHandle::from_raw_parts(10, 0);
        let ally_h = ColliderHandle::from_raw_parts(20, 0);
        let enemy_h = ColliderHandle::from_raw_parts(30, 0);

        let map = build_collider_type_map(
            &[obs_h],
            &[(ally_h, 2), (enemy_h, 3)],
            2, // self_team = 2
        );

        assert_eq!(map.len(), 3);
        assert!((map[&obs_h] - HIT_WALL).abs() < f32::EPSILON, "obstacle -> wall");
        assert!((map[&ally_h] - HIT_WALL).abs() < f32::EPSILON, "ally -> wall");
        assert!((map[&enemy_h] - HIT_ENEMY).abs() < f32::EPSILON, "enemy -> enemy");
    }

    #[test]
    fn build_map_empty_inputs() {
        let map = build_collider_type_map(&[], &[], 0);
        assert!(map.is_empty());
    }
}

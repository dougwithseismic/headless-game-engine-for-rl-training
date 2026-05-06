use bevy_ecs::prelude::*;
use glam::Vec2;
use rapier2d::prelude::*;

#[derive(Resource)]
pub struct PhysicsState {
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,
    pub gravity: Vector<f32>,
    pub integration_parameters: IntegrationParameters,
    pub physics_pipeline: PhysicsPipeline,
    pub island_manager: IslandManager,
    pub broad_phase: DefaultBroadPhase,
    pub narrow_phase: NarrowPhase,
    pub impulse_joint_set: ImpulseJointSet,
    pub multibody_joint_set: MultibodyJointSet,
    pub ccd_solver: CCDSolver,
    pub query_pipeline: QueryPipeline,
}

impl PhysicsState {
    pub fn new(gravity: Vec2, dt: f32) -> Self {
        let params = IntegrationParameters { dt, ..Default::default() };

        Self {
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            gravity: vector![gravity.x, gravity.y],
            integration_parameters: params,
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
        }
    }

    pub fn step(&mut self) {
        self.physics_pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            Some(&mut self.query_pipeline),
            &(),
            &(),
        );
    }

    pub fn add_dynamic_body(
        &mut self,
        pos: Vec2,
        radius: f32,
    ) -> (RigidBodyHandle, ColliderHandle) {
        let body = RigidBodyBuilder::dynamic()
            .translation(vector![pos.x, pos.y])
            .linear_damping(5.0)
            .build();
        let body_handle = self.rigid_body_set.insert(body);

        let collider = ColliderBuilder::ball(radius)
            .restitution(0.0)
            .friction(0.5)
            .active_events(ActiveEvents::COLLISION_EVENTS)
            .build();
        let collider_handle =
            self.collider_set
                .insert_with_parent(collider, body_handle, &mut self.rigid_body_set);

        (body_handle, collider_handle)
    }

    pub fn add_static_body(
        &mut self,
        pos: Vec2,
        half_extents: Vec2,
    ) -> (RigidBodyHandle, ColliderHandle) {
        let body = RigidBodyBuilder::fixed()
            .translation(vector![pos.x, pos.y])
            .build();
        let body_handle = self.rigid_body_set.insert(body);

        let collider = ColliderBuilder::cuboid(half_extents.x, half_extents.y)
            .restitution(0.0)
            .friction(0.5)
            .build();
        let collider_handle =
            self.collider_set
                .insert_with_parent(collider, body_handle, &mut self.rigid_body_set);

        (body_handle, collider_handle)
    }

    pub fn cast_ray(
        &self,
        origin: Vec2,
        direction: Vec2,
        max_toi: f32,
        exclude_collider: Option<ColliderHandle>,
    ) -> Option<(ColliderHandle, f32)> {
        let ray = Ray::new(
            point![origin.x, origin.y],
            vector![direction.x, direction.y],
        );

        let predicate = |handle: ColliderHandle, _collider: &Collider| -> bool {
            if let Some(exclude) = exclude_collider {
                handle != exclude
            } else {
                true
            }
        };
        let filter = QueryFilter::default().predicate(&predicate);

        self.query_pipeline
            .cast_ray(&self.rigid_body_set, &self.collider_set, &ray, max_toi, true, filter)
    }

    pub fn has_line_of_sight(
        &self,
        from: Vec2,
        to: Vec2,
        exclude: Option<ColliderHandle>,
    ) -> bool {
        let delta = to - from;
        let dist = delta.length();
        if dist < 0.1 {
            return true;
        }
        let dir = delta / dist;
        self.cast_ray(from, dir, dist, exclude).is_none()
    }

    pub fn body_position(&self, handle: RigidBodyHandle) -> Option<Vec2> {
        self.rigid_body_set
            .get(handle)
            .map(|b| {
                let t = b.translation();
                Vec2::new(t.x, t.y)
            })
    }

    pub fn body_velocity(&self, handle: RigidBodyHandle) -> Option<Vec2> {
        self.rigid_body_set
            .get(handle)
            .map(|b| {
                let v = b.linvel();
                Vec2::new(v.x, v.y)
            })
    }

    pub fn set_body_linvel(&mut self, handle: RigidBodyHandle, vel: Vec2) {
        if let Some(body) = self.rigid_body_set.get_mut(handle) {
            body.set_linvel(vector![vel.x, vel.y], true);
        }
    }

    pub fn set_body_position(&mut self, handle: RigidBodyHandle, pos: Vec2) {
        if let Some(body) = self.rigid_body_set.get_mut(handle) {
            body.set_translation(vector![pos.x, pos.y], true);
        }
    }

    pub fn collider_for_body(&self, body_handle: RigidBodyHandle) -> Option<ColliderHandle> {
        self.rigid_body_set
            .get(body_handle)
            .and_then(|b| b.colliders().first().copied())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 60.0;

    fn zero_gravity() -> PhysicsState {
        PhysicsState::new(Vec2::ZERO, DT)
    }

    // ---- 1. new() ----

    #[test]
    fn new_sets_dt_and_gravity() {
        let gravity = Vec2::new(0.0, -9.81);
        let dt = 0.02;
        let state = PhysicsState::new(gravity, dt);

        assert!((state.integration_parameters.dt - dt).abs() < f32::EPSILON);
        assert!((state.gravity.x - 0.0).abs() < f32::EPSILON);
        assert!((state.gravity.y - (-9.81)).abs() < f32::EPSILON);
    }

    #[test]
    fn new_zero_gravity() {
        let state = zero_gravity();
        assert!((state.gravity.x).abs() < f32::EPSILON);
        assert!((state.gravity.y).abs() < f32::EPSILON);
    }

    // ---- 2. add_dynamic_body ----

    #[test]
    fn add_dynamic_body_returns_valid_handles() {
        let mut state = zero_gravity();
        let pos = Vec2::new(10.0, 20.0);
        let (bh, ch) = state.add_dynamic_body(pos, 0.5);

        assert!(state.rigid_body_set.get(bh).is_some());
        assert!(state.collider_set.get(ch).is_some());
    }

    #[test]
    fn add_dynamic_body_at_correct_position() {
        let mut state = zero_gravity();
        let pos = Vec2::new(3.0, 7.0);
        let (bh, _) = state.add_dynamic_body(pos, 1.0);

        let read_pos = state.body_position(bh).unwrap();
        assert!((read_pos.x - 3.0).abs() < f32::EPSILON);
        assert!((read_pos.y - 7.0).abs() < f32::EPSILON);
    }

    #[test]
    fn add_dynamic_body_is_dynamic() {
        let mut state = zero_gravity();
        let (bh, _) = state.add_dynamic_body(Vec2::ZERO, 0.5);
        let body = state.rigid_body_set.get(bh).unwrap();
        assert!(body.is_dynamic());
    }

    // ---- 3. add_static_body ----

    #[test]
    fn add_static_body_returns_valid_handles() {
        let mut state = zero_gravity();
        let (bh, ch) = state.add_static_body(Vec2::new(5.0, 5.0), Vec2::new(1.0, 1.0));

        assert!(state.rigid_body_set.get(bh).is_some());
        assert!(state.collider_set.get(ch).is_some());
    }

    #[test]
    fn add_static_body_at_correct_position() {
        let mut state = zero_gravity();
        let pos = Vec2::new(-4.0, 12.0);
        let (bh, _) = state.add_static_body(pos, Vec2::new(2.0, 3.0));

        let read_pos = state.body_position(bh).unwrap();
        assert!((read_pos.x - (-4.0)).abs() < f32::EPSILON);
        assert!((read_pos.y - 12.0).abs() < f32::EPSILON);
    }

    #[test]
    fn add_static_body_is_fixed() {
        let mut state = zero_gravity();
        let (bh, _) = state.add_static_body(Vec2::ZERO, Vec2::new(1.0, 1.0));
        let body = state.rigid_body_set.get(bh).unwrap();
        assert!(body.is_fixed());
    }

    // ---- 4. step ----

    #[test]
    fn step_moves_dynamic_body_with_velocity() {
        let mut state = zero_gravity();
        let (bh, _) = state.add_dynamic_body(Vec2::ZERO, 0.5);

        // Apply a large velocity so movement is visible despite damping
        state.set_body_linvel(bh, Vec2::new(100.0, 0.0));

        let pos_before = state.body_position(bh).unwrap();
        state.step();
        let pos_after = state.body_position(bh).unwrap();

        // Body should have moved in the +x direction
        assert!(pos_after.x > pos_before.x, "body did not move: before={pos_before:?}, after={pos_after:?}");
    }

    #[test]
    fn step_does_not_move_static_body() {
        let mut state = zero_gravity();
        let (bh, _) = state.add_static_body(Vec2::new(5.0, 5.0), Vec2::new(1.0, 1.0));

        state.step();

        let pos = state.body_position(bh).unwrap();
        assert!((pos.x - 5.0).abs() < f32::EPSILON);
        assert!((pos.y - 5.0).abs() < f32::EPSILON);
    }

    // ---- 5. body_position / body_velocity round-trip ----

    #[test]
    fn body_position_round_trip() {
        let mut state = zero_gravity();
        let (bh, _) = state.add_dynamic_body(Vec2::ZERO, 0.5);

        state.set_body_position(bh, Vec2::new(42.0, -13.0));
        let pos = state.body_position(bh).unwrap();
        assert!((pos.x - 42.0).abs() < f32::EPSILON);
        assert!((pos.y - (-13.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn body_velocity_round_trip() {
        let mut state = zero_gravity();
        let (bh, _) = state.add_dynamic_body(Vec2::ZERO, 0.5);

        state.set_body_linvel(bh, Vec2::new(5.0, -3.0));
        let vel = state.body_velocity(bh).unwrap();
        assert!((vel.x - 5.0).abs() < f32::EPSILON);
        assert!((vel.y - (-3.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn body_position_invalid_handle_returns_none() {
        let state = zero_gravity();
        let fake = RigidBodyHandle::from_raw_parts(999, 0);
        assert!(state.body_position(fake).is_none());
    }

    #[test]
    fn body_velocity_invalid_handle_returns_none() {
        let state = zero_gravity();
        let fake = RigidBodyHandle::from_raw_parts(999, 0);
        assert!(state.body_velocity(fake).is_none());
    }

    // ---- 6. set_body_linvel ----

    #[test]
    fn set_body_linvel_causes_movement() {
        let mut state = zero_gravity();
        let (bh, _) = state.add_dynamic_body(Vec2::new(0.0, 0.0), 0.5);

        state.set_body_linvel(bh, Vec2::new(0.0, 200.0));
        state.step();

        let pos = state.body_position(bh).unwrap();
        assert!(pos.y > 0.0, "body should have moved in +y, got {}", pos.y);
    }

    // ---- 7. cast_ray ----

    #[test]
    fn cast_ray_hits_static_wall() {
        let mut state = zero_gravity();

        // Place a wall at x=10
        state.add_static_body(Vec2::new(10.0, 0.0), Vec2::new(1.0, 5.0));

        // Update the query pipeline so the ray cast can find the collider
        state.query_pipeline.update(&state.collider_set);

        // Cast ray from origin toward +x
        let hit = state.cast_ray(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), 100.0, None);

        assert!(hit.is_some(), "ray should have hit the wall");
        let (_, toi) = hit.unwrap();
        // Wall center at x=10 with half-extent 1.0, so nearest edge at x=9.0
        assert!((toi - 9.0).abs() < 0.1, "toi should be ~9.0, got {toi}");
    }

    #[test]
    fn cast_ray_misses_when_aimed_away() {
        let mut state = zero_gravity();

        // Place a wall at x=10
        state.add_static_body(Vec2::new(10.0, 0.0), Vec2::new(1.0, 5.0));

        state.query_pipeline.update(&state.collider_set);

        // Cast ray from origin toward -x (away from wall)
        let hit = state.cast_ray(Vec2::new(0.0, 0.0), Vec2::new(-1.0, 0.0), 100.0, None);

        assert!(hit.is_none(), "ray should not hit anything when aimed away");
    }

    // ---- 8. cast_ray with exclude ----

    #[test]
    fn cast_ray_exclude_collider() {
        let mut state = zero_gravity();

        // Place two walls: one close at x=5, one far at x=15
        let (_, close_ch) = state.add_static_body(Vec2::new(5.0, 0.0), Vec2::new(1.0, 5.0));
        state.add_static_body(Vec2::new(15.0, 0.0), Vec2::new(1.0, 5.0));

        state.query_pipeline.update(&state.collider_set);

        // Without exclude: should hit close wall (edge at x=4)
        let hit_no_exclude = state.cast_ray(Vec2::ZERO, Vec2::new(1.0, 0.0), 100.0, None);
        assert!(hit_no_exclude.is_some());
        let (hit_handle, toi_close) = hit_no_exclude.unwrap();
        assert_eq!(hit_handle, close_ch);
        assert!((toi_close - 4.0).abs() < 0.1, "expected toi ~4.0, got {toi_close}");

        // With exclude on the close wall: should hit the far wall (edge at x=14)
        let hit_exclude = state.cast_ray(Vec2::ZERO, Vec2::new(1.0, 0.0), 100.0, Some(close_ch));
        assert!(hit_exclude.is_some(), "should still hit the far wall");
        let (_, toi_far) = hit_exclude.unwrap();
        assert!((toi_far - 14.0).abs() < 0.1, "expected toi ~14.0, got {toi_far}");
    }

    // ---- 9. collision: body doesn't pass through wall ----

    #[test]
    fn collision_body_does_not_pass_through_wall() {
        let mut state = zero_gravity();

        // Dynamic body at origin, radius 0.5, no damping for this test
        let body = RigidBodyBuilder::dynamic()
            .translation(vector![0.0, 0.0])
            .linear_damping(0.0)
            .linvel(vector![50.0, 0.0])
            .build();
        let bh = state.rigid_body_set.insert(body);
        let collider = ColliderBuilder::ball(0.5).restitution(0.0).build();
        state
            .collider_set
            .insert_with_parent(collider, bh, &mut state.rigid_body_set);

        // Wall at x=10, half-extent 5.0 in x (spans x=5..15)
        state.add_static_body(Vec2::new(10.0, 0.0), Vec2::new(5.0, 50.0));

        for _ in 0..120 {
            state.step();
        }

        let pos = state.body_position(bh).unwrap();
        // Wall near edge is at x=5.0, body radius is 0.5, so body center should stop at ~4.5
        assert!(
            pos.x < 6.0,
            "body should not pass through wall: body.x={}, wall edge at x=5.0",
            pos.x
        );
    }
}

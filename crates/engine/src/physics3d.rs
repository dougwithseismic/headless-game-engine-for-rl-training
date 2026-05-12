use bevy_ecs::prelude::*;
use glam::Vec3;
use rapier3d::prelude as r3d;
use std::collections::HashMap;

fn vec3r(x: f32, y: f32, z: f32) -> r3d::Vector<f32> {
    r3d::Vector::new(x, y, z)
}

fn point3r(x: f32, y: f32, z: f32) -> rapier3d::na::Point3<f32> {
    rapier3d::na::Point3::new(x, y, z)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColliderTag {
    Wall,
    Obstacle,
    Agent { team: u8 },
    Target,
}

#[derive(Resource)]
pub struct Physics3DState {
    pub rigid_body_set: r3d::RigidBodySet,
    pub collider_set: r3d::ColliderSet,
    pub integration_parameters: r3d::IntegrationParameters,
    pub physics_pipeline: r3d::PhysicsPipeline,
    pub island_manager: r3d::IslandManager,
    pub broad_phase: r3d::DefaultBroadPhase,
    pub narrow_phase: r3d::NarrowPhase,
    pub impulse_joint_set: r3d::ImpulseJointSet,
    pub multibody_joint_set: r3d::MultibodyJointSet,
    pub ccd_solver: r3d::CCDSolver,
    pub query_pipeline: r3d::QueryPipeline,
    pub gravity: r3d::Vector<f32>,
    pub collider_tags: HashMap<r3d::ColliderHandle, ColliderTag>,
}

impl Physics3DState {
    pub fn new(arena: (f32, f32, f32), gravity_mag: f32, dt: f32) -> Self {
        let gravity = vec3r(0.0, -gravity_mag, 0.0);
        let integration_parameters = r3d::IntegrationParameters {
            dt,
            ..Default::default()
        };

        let mut rigid_body_set = r3d::RigidBodySet::new();
        let mut collider_set = r3d::ColliderSet::new();
        let mut collider_tags = HashMap::new();

        let (aw, ah, ad) = arena;
        let wall_t = 10.0;

        let walls: [(r3d::Vector<f32>, f32, f32, f32); 6] = [
            // Floor (y=0)
            (vec3r(aw / 2.0, -wall_t / 2.0, ad / 2.0), aw / 2.0 + wall_t, wall_t / 2.0, ad / 2.0 + wall_t),
            // Ceiling (y=ah)
            (vec3r(aw / 2.0, ah + wall_t / 2.0, ad / 2.0), aw / 2.0 + wall_t, wall_t / 2.0, ad / 2.0 + wall_t),
            // Wall -X
            (vec3r(-wall_t / 2.0, ah / 2.0, ad / 2.0), wall_t / 2.0, ah / 2.0 + wall_t, ad / 2.0 + wall_t),
            // Wall +X
            (vec3r(aw + wall_t / 2.0, ah / 2.0, ad / 2.0), wall_t / 2.0, ah / 2.0 + wall_t, ad / 2.0 + wall_t),
            // Wall -Z
            (vec3r(aw / 2.0, ah / 2.0, -wall_t / 2.0), aw / 2.0 + wall_t, ah / 2.0 + wall_t, wall_t / 2.0),
            // Wall +Z
            (vec3r(aw / 2.0, ah / 2.0, ad + wall_t / 2.0), aw / 2.0 + wall_t, ah / 2.0 + wall_t, wall_t / 2.0),
        ];

        for (pos, hx, hy, hz) in walls {
            let body = r3d::RigidBodyBuilder::fixed().translation(pos).build();
            let bh = rigid_body_set.insert(body);
            let col = r3d::ColliderBuilder::cuboid(hx, hy, hz)
                .restitution(0.2)
                .friction(0.8)
                .build();
            let ch = collider_set.insert_with_parent(col, bh, &mut rigid_body_set);
            collider_tags.insert(ch, ColliderTag::Wall);
        }

        Self {
            rigid_body_set,
            collider_set,
            integration_parameters,
            physics_pipeline: r3d::PhysicsPipeline::new(),
            island_manager: r3d::IslandManager::new(),
            broad_phase: r3d::DefaultBroadPhase::new(),
            narrow_phase: r3d::NarrowPhase::new(),
            impulse_joint_set: r3d::ImpulseJointSet::new(),
            multibody_joint_set: r3d::MultibodyJointSet::new(),
            ccd_solver: r3d::CCDSolver::new(),
            query_pipeline: r3d::QueryPipeline::new(),
            gravity,
            collider_tags,
        }
    }

    pub fn add_capsule_agent(
        &mut self,
        spawn_pos: Vec3,
        half_height: f32,
        radius: f32,
        team: u8,
    ) -> (r3d::RigidBodyHandle, r3d::ColliderHandle) {
        let body = r3d::RigidBodyBuilder::dynamic()
            .translation(vec3r(spawn_pos.x, spawn_pos.y, spawn_pos.z))
            .linear_damping(5.0)
            .locked_axes(r3d::LockedAxes::ROTATION_LOCKED)
            .build();
        let bh = self.rigid_body_set.insert(body);

        let col = r3d::ColliderBuilder::capsule_y(half_height, radius)
            .restitution(0.0)
            .friction(0.5)
            .build();
        let ch = self.collider_set.insert_with_parent(col, bh, &mut self.rigid_body_set);
        self.collider_tags.insert(ch, ColliderTag::Agent { team });
        (bh, ch)
    }

    pub fn add_static_box(
        &mut self,
        pos: Vec3,
        half_extents: Vec3,
    ) -> (r3d::RigidBodyHandle, r3d::ColliderHandle) {
        let body = r3d::RigidBodyBuilder::fixed()
            .translation(vec3r(pos.x, pos.y, pos.z))
            .build();
        let bh = self.rigid_body_set.insert(body);
        let col = r3d::ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z)
            .restitution(0.1)
            .friction(0.6)
            .build();
        let ch = self.collider_set.insert_with_parent(col, bh, &mut self.rigid_body_set);
        self.collider_tags.insert(ch, ColliderTag::Obstacle);
        (bh, ch)
    }

    pub fn add_static_sphere(
        &mut self,
        pos: Vec3,
        radius: f32,
    ) -> (r3d::RigidBodyHandle, r3d::ColliderHandle) {
        let body = r3d::RigidBodyBuilder::fixed()
            .translation(vec3r(pos.x, pos.y, pos.z))
            .build();
        let bh = self.rigid_body_set.insert(body);
        let col = r3d::ColliderBuilder::ball(radius).build();
        let ch = self.collider_set.insert_with_parent(col, bh, &mut self.rigid_body_set);
        self.collider_tags.insert(ch, ColliderTag::Target);
        (bh, ch)
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

    pub fn cast_ray(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_toi: f32,
        exclude: Option<r3d::ColliderHandle>,
    ) -> Option<(r3d::ColliderHandle, f32)> {
        let ray = r3d::Ray::new(
            point3r(origin.x, origin.y, origin.z),
            vec3r(direction.x, direction.y, direction.z),
        );

        let predicate = move |handle: r3d::ColliderHandle, _: &r3d::Collider| -> bool {
            if let Some(excl) = exclude {
                handle != excl
            } else {
                true
            }
        };
        let filter = r3d::QueryFilter::default().predicate(&predicate);

        self.query_pipeline
            .cast_ray(
                &self.rigid_body_set,
                &self.collider_set,
                &ray,
                max_toi,
                true,
                filter,
            )
    }

    pub fn cast_ray_classified(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_toi: f32,
        exclude: Option<r3d::ColliderHandle>,
        self_team: u8,
    ) -> (f32, f32) {
        let ray = r3d::Ray::new(
            point3r(origin.x, origin.y, origin.z),
            vec3r(direction.x, direction.y, direction.z),
        );

        let excl = exclude.unwrap_or(r3d::ColliderHandle::from_raw_parts(u32::MAX, 0));
        let predicate = move |handle: r3d::ColliderHandle, _: &r3d::Collider| handle != excl;
        let filter = r3d::QueryFilter::default().predicate(&predicate);

        match self.query_pipeline.cast_ray(
            &self.rigid_body_set,
            &self.collider_set,
            &ray,
            max_toi,
            true,
            filter,
        ) {
            Some((handle, toi)) => {
                let dist = (toi / max_toi).clamp(0.0, 1.0);
                let hit_type = match self.collider_tags.get(&handle) {
                    Some(ColliderTag::Wall) | Some(ColliderTag::Obstacle) => 0.33,
                    Some(ColliderTag::Agent { team }) => {
                        if *team == self_team { 0.66 } else { 1.0 }
                    }
                    Some(ColliderTag::Target) => 0.8,
                    None => 0.33,
                };
                (dist, hit_type)
            }
            None => (1.0, 0.0),
        }
    }

    pub fn has_line_of_sight(
        &self,
        from: Vec3,
        to: Vec3,
        exclude: Option<r3d::ColliderHandle>,
    ) -> bool {
        let delta = to - from;
        let dist = delta.length();
        if dist < 0.01 {
            return true;
        }
        let dir = delta / dist;
        match self.cast_ray(from, dir, dist, exclude) {
            Some((_, toi)) => toi >= dist - 0.1,
            None => true,
        }
    }

    pub fn obstacles_block_los(&self, from: Vec3, to: Vec3) -> bool {
        let delta = to - from;
        let dist = delta.length();
        if dist < 0.01 { return false; }
        let dir = delta / dist;

        let ray = r3d::Ray::new(
            point3r(from.x, from.y, from.z),
            vec3r(dir.x, dir.y, dir.z),
        );

        let tags = &self.collider_tags;
        let predicate = move |handle: r3d::ColliderHandle, _: &r3d::Collider| -> bool {
            matches!(tags.get(&handle), Some(ColliderTag::Obstacle))
        };
        let filter = r3d::QueryFilter::default().predicate(&predicate);

        self.query_pipeline
            .cast_ray(
                &self.rigid_body_set,
                &self.collider_set,
                &ray,
                dist,
                true,
                filter,
            )
            .is_some()
    }

    pub fn set_body_linvel(&mut self, handle: r3d::RigidBodyHandle, vel: Vec3) {
        if let Some(body) = self.rigid_body_set.get_mut(handle) {
            body.set_linvel(vec3r(vel.x, vel.y, vel.z), true);
        }
    }

    pub fn body_position(&self, handle: r3d::RigidBodyHandle) -> Option<Vec3> {
        self.rigid_body_set.get(handle).map(|b| {
            let t = b.translation();
            Vec3::new(t.x, t.y, t.z)
        })
    }

    pub fn body_velocity(&self, handle: r3d::RigidBodyHandle) -> Option<Vec3> {
        self.rigid_body_set.get(handle).map(|b| {
            let v = b.linvel();
            Vec3::new(v.x, v.y, v.z)
        })
    }

    pub fn teleport_body(&mut self, handle: r3d::RigidBodyHandle, pos: Vec3) {
        if let Some(body) = self.rigid_body_set.get_mut(handle) {
            body.set_translation(vec3r(pos.x, pos.y, pos.z), true);
            body.set_linvel(vec3r(0.0, 0.0, 0.0), true);
        }
    }
}

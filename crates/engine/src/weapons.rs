use bevy_ecs::prelude::*;
use glam::Vec2;

use crate::action_space::{ActionSpaceDef, RawActionBuffer};
use crate::ecs::components::*;
use crate::ecs::resources::*;
use crate::observation::{RewardBuffer, ShotEventBuffer};
use crate::physics::PhysicsState;
use crate::telemetry::TelemetryEvent;

/// Ticks down `cooldown_remaining` and `reload_remaining` on ALL weapon slots
/// (not just the active one) for every entity with an `Inventory` that is not dead.
///
/// When a reload completes (`reload_remaining` reaches 0 while `is_reloading` is true),
/// the weapon's ammo is refilled to `max_ammo` and `is_reloading` is set to false.
pub fn inventory_cooldown_system(
    mut query: Query<&mut Inventory, Without<Dead>>,
    tick: Res<TickState>,
) {
    for mut inv in &mut query {
        for slot in inv.weapons.iter_mut() {
            slot.cooldown_remaining = (slot.cooldown_remaining - tick.delta).max(0.0);

            if slot.is_reloading {
                slot.reload_remaining = (slot.reload_remaining - tick.delta).max(0.0);
                if slot.reload_remaining <= 0.0 {
                    slot.is_reloading = false;
                    slot.ammo = slot.max_ammo;
                }
            }
        }
    }
}

/// Reads the `weapon_select` action head (head index 3) and switches the active weapon
/// if the selected index differs from the current active slot and is a valid index.
///
/// Applies a switch penalty of 0.3s cooldown on the newly selected weapon to prevent
/// instant-switch-fire exploits.
pub fn weapon_switch_system(
    mut query: Query<(Entity, &mut Inventory), Without<Dead>>,
    raw_buffer: Res<RawActionBuffer>,
    action_space: Res<ActionSpaceDef>,
) {
    if action_space.heads.len() <= 3 {
        return;
    }

    for (entity, mut inv) in &mut query {
        let Some(raw) = raw_buffer.get(entity) else {
            continue;
        };
        if raw.len() < action_space.total_size {
            continue;
        }

        let select_slice = action_space.extract_head(raw, 3);
        if select_slice.is_empty() {
            continue;
        }

        let new_index = select_slice[0].round() as usize;
        if new_index < inv.weapons.len() && new_index != inv.active {
            inv.active = new_index;
            // Apply switch penalty
            inv.weapons[new_index].cooldown_remaining = 0.3;
        }
    }
}

/// Combat system that uses `Inventory` instead of the `Weapon` component.
///
/// Reads the `shoot` action head (head index 2). On fire:
/// - Checks cooldown, ammo, and reload state
/// - Decrements ammo, sets cooldown
/// - Auto-starts reload when ammo hits 0
/// - **Rifle**: single hitscan ray (same as standard combat)
/// - **Shotgun**: 5 pellets in a 15-degree cone, each doing `damage` independently
///
/// Emits `ShotFired` and `Damage` telemetry events and adjusts rewards.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn inventory_combat_system(
    mut commands: Commands,
    mut shooters: Query<
        (Entity, &Position, &Facing, &mut Inventory, &Team, &PhysicsHandle),
        Without<Dead>,
    >,
    mut targets: Query<(Entity, &Position, &mut Health, &Team, &PhysicsHandle), Without<Dead>>,
    raw_buffer: Res<RawActionBuffer>,
    action_space: Res<ActionSpaceDef>,
    physics: Res<PhysicsState>,
    tick: Res<TickState>,
    mut telemetry: ResMut<TelemetryBuffer>,
    mut rewards: ResMut<RewardBuffer>,
    mut shot_events: ResMut<ShotEventBuffer>,
    candidate_buffer: Res<CandidatePositionBuffer>,
    tactical_config: Res<crate::scenarios::tactical_deathmatch::TacticalConfig>,
) {
    use crate::scenarios::tactical_deathmatch::RewardMode;

    let mut hits: Vec<(Entity, f32, Entity, bool)> = Vec::new();

    for (shooter_entity, shooter_pos, facing, mut inv, shooter_team, shooter_ph) in &mut shooters {
        let mut wants_shoot = false;

        if let Some(raw) = raw_buffer.get(shooter_entity)
            && raw.len() >= action_space.total_size
            && action_space.heads.len() > 2
        {
            let shoot_slice = action_space.extract_head(raw, 2);
            if !shoot_slice.is_empty() && shoot_slice[0] > 0.5 {
                wants_shoot = true;
            }
        }

        let active_idx = inv.active;
        let weapon = match inv.weapons.get(active_idx) {
            Some(w) => w,
            None => continue,
        };

        if !wants_shoot || weapon.cooldown_remaining > 0.0 || weapon.ammo == 0 || weapon.is_reloading
        {
            continue;
        }

        let weapon_type = weapon.weapon_type;
        let range = weapon.range;
        let damage = weapon.damage;
        let fire_rate = weapon.fire_rate;

        // Consume ammo and set cooldown
        {
            let slot = &mut inv.weapons[active_idx];
            slot.ammo -= 1;
            slot.cooldown_remaining = fire_rate;

            // Auto-reload when ammo depleted
            if slot.ammo == 0 {
                slot.is_reloading = true;
                slot.reload_remaining = slot.reload_time;
            }
        }

        // Build ray directions based on weapon type
        let ray_angles: Vec<f32> = match weapon_type {
            WeaponType::Rifle => vec![facing.0],
            WeaponType::Shotgun => {
                // 5 pellets in a 15-degree (0.2618 rad) cone
                let half_spread = 7.5_f32.to_radians();
                let step = half_spread / 2.0;
                vec![
                    facing.0 - 2.0 * step,
                    facing.0 - step,
                    facing.0,
                    facing.0 + step,
                    facing.0 + 2.0 * step,
                ]
            }
        };

        // Check cover for reward bonus
        let in_cover = candidate_buffer
            .get(shooter_entity)
            .and_then(|cs| cs.positions.get(8).map(|stay| !stay.has_los_to_enemy))
            .unwrap_or(false);

        for &angle in &ray_angles {
            let dir = Vec2::new(angle.cos(), angle.sin());
            let mut best_hit: Option<(Entity, f32)> = None;

            for (target_entity, target_pos, _health, target_team, _target_ph) in &targets {
                if target_entity == shooter_entity || target_team.0 == shooter_team.0 {
                    continue;
                }

                let to_target = target_pos.0 - shooter_pos.0;
                let dist = to_target.length();
                if dist > range {
                    continue;
                }

                let proj = to_target.dot(dir);
                if proj < 0.0 {
                    continue;
                }
                let perp_dist = (to_target - dir * proj).length();
                let hitbox_radius = 15.0;

                if perp_dist <= hitbox_radius {
                    let occluded = if let Some((hit_collider, hit_toi)) =
                        physics.cast_ray(shooter_pos.0, dir, proj, Some(shooter_ph.collider))
                    {
                        let target_collider = targets
                            .get(target_entity)
                            .ok()
                            .and_then(|t| physics.collider_for_body(t.4.body));
                        target_collider.is_none_or(|tc| hit_collider != tc) && hit_toi < proj
                    } else {
                        false
                    };

                    if !occluded && (best_hit.is_none() || proj < best_hit.unwrap().1) {
                        best_hit = Some((target_entity, proj));
                    }
                }
            }

            telemetry.push(TelemetryEvent::ShotFired {
                tick: tick.tick,
                shooter: shooter_entity.to_bits(),
                origin: shooter_pos.0,
                direction: dir,
                hit_target: best_hit.map(|(e, _)| e.to_bits()),
            });

            if let Some((hit_entity, _)) = best_hit {
                hits.push((hit_entity, damage, shooter_entity, in_cover));
            }
        }

        // Only emit one shot event per fire action (not per pellet)
        shot_events.push(shooter_entity, shooter_pos.0);
    }

    for &(hit_entity, damage, shooter_entity, shooter_in_cover) in &hits {
        if let Ok((_entity, _pos, mut health, _team, _ph)) = targets.get_mut(hit_entity) {
            let max_hp = health.max;
            health.current -= damage;
            telemetry.push(TelemetryEvent::Damage {
                tick: tick.tick,
                source: shooter_entity.to_bits(),
                target: hit_entity.to_bits(),
                amount: damage,
            });
            let base_dmg_reward = 0.5 * damage / max_hp;
            let cover_bonus =
                if shooter_in_cover && tactical_config.reward_mode == RewardMode::Cover {
                    0.3 * damage / max_hp
                } else {
                    0.0
                };
            rewards.add(shooter_entity, base_dmg_reward + cover_bonus);
            rewards.add(hit_entity, -0.3 * damage / max_hp);
        }
        commands
            .entity(hit_entity)
            .insert(LastDamageSource(shooter_entity));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::resources::TickState;

    fn make_rifle() -> WeaponSlot {
        WeaponSlot {
            weapon_type: WeaponType::Rifle,
            damage: 34.0,
            fire_rate: 0.3,
            range: 400.0,
            cooldown_remaining: 0.0,
            ammo: 30,
            max_ammo: 30,
            reload_time: 2.0,
            reload_remaining: 0.0,
            is_reloading: false,
        }
    }

    fn make_shotgun() -> WeaponSlot {
        WeaponSlot {
            weapon_type: WeaponType::Shotgun,
            damage: 15.0,
            fire_rate: 0.8,
            range: 200.0,
            cooldown_remaining: 0.0,
            ammo: 8,
            max_ammo: 8,
            reload_time: 2.5,
            reload_remaining: 0.0,
            is_reloading: false,
        }
    }

    fn make_inventory() -> Inventory {
        Inventory {
            weapons: vec![make_rifle(), make_shotgun()],
            active: 0,
        }
    }

    // -------------------------------------------------------------------
    // inventory_cooldown_system tests
    // -------------------------------------------------------------------

    #[test]
    fn cooldown_ticks_down_for_all_slots() {
        let mut world = World::new();
        let tick = TickState { tick: 1, delta: 0.1 };
        world.insert_resource(tick);

        let mut inv = make_inventory();
        inv.weapons[0].cooldown_remaining = 0.3;
        inv.weapons[1].cooldown_remaining = 0.5;
        let entity = world.spawn(inv).id();

        let mut schedule = Schedule::default();
        schedule.add_systems(inventory_cooldown_system);
        schedule.run(&mut world);

        let inv = world.get::<Inventory>(entity).unwrap();
        assert!(
            (inv.weapons[0].cooldown_remaining - 0.2).abs() < 0.001,
            "rifle cooldown should tick down from 0.3 to 0.2, got {}",
            inv.weapons[0].cooldown_remaining
        );
        assert!(
            (inv.weapons[1].cooldown_remaining - 0.4).abs() < 0.001,
            "shotgun cooldown should tick down from 0.5 to 0.4, got {}",
            inv.weapons[1].cooldown_remaining
        );
    }

    #[test]
    fn cooldown_does_not_go_negative() {
        let mut world = World::new();
        let tick = TickState { tick: 1, delta: 0.5 };
        world.insert_resource(tick);

        let mut inv = make_inventory();
        inv.weapons[0].cooldown_remaining = 0.1;
        let entity = world.spawn(inv).id();

        let mut schedule = Schedule::default();
        schedule.add_systems(inventory_cooldown_system);
        schedule.run(&mut world);

        let inv = world.get::<Inventory>(entity).unwrap();
        assert_eq!(
            inv.weapons[0].cooldown_remaining, 0.0,
            "cooldown should clamp to 0, not go negative"
        );
    }

    #[test]
    fn reload_completes_and_refills_ammo() {
        let mut world = World::new();
        let tick = TickState { tick: 1, delta: 0.5 };
        world.insert_resource(tick);

        let mut inv = make_inventory();
        inv.weapons[0].ammo = 0;
        inv.weapons[0].is_reloading = true;
        inv.weapons[0].reload_remaining = 0.3;
        let entity = world.spawn(inv).id();

        let mut schedule = Schedule::default();
        schedule.add_systems(inventory_cooldown_system);
        schedule.run(&mut world);

        let inv = world.get::<Inventory>(entity).unwrap();
        assert!(!inv.weapons[0].is_reloading, "reload should complete");
        assert_eq!(
            inv.weapons[0].ammo, inv.weapons[0].max_ammo,
            "ammo should be refilled to max_ammo on reload complete"
        );
    }

    #[test]
    fn reload_ticks_down_but_not_complete() {
        let mut world = World::new();
        let tick = TickState { tick: 1, delta: 0.1 };
        world.insert_resource(tick);

        let mut inv = make_inventory();
        inv.weapons[0].ammo = 0;
        inv.weapons[0].is_reloading = true;
        inv.weapons[0].reload_remaining = 2.0;
        let entity = world.spawn(inv).id();

        let mut schedule = Schedule::default();
        schedule.add_systems(inventory_cooldown_system);
        schedule.run(&mut world);

        let inv = world.get::<Inventory>(entity).unwrap();
        assert!(inv.weapons[0].is_reloading, "reload should still be in progress");
        assert!(
            (inv.weapons[0].reload_remaining - 1.9).abs() < 0.001,
            "reload_remaining should tick down from 2.0 to 1.9, got {}",
            inv.weapons[0].reload_remaining
        );
        assert_eq!(inv.weapons[0].ammo, 0, "ammo should still be 0 during reload");
    }

    #[test]
    fn dead_entities_are_skipped() {
        let mut world = World::new();
        let tick = TickState { tick: 1, delta: 0.1 };
        world.insert_resource(tick);

        let mut inv = make_inventory();
        inv.weapons[0].cooldown_remaining = 1.0;
        let entity = world.spawn((inv, Dead)).id();

        let mut schedule = Schedule::default();
        schedule.add_systems(inventory_cooldown_system);
        schedule.run(&mut world);

        let inv = world.get::<Inventory>(entity).unwrap();
        assert_eq!(
            inv.weapons[0].cooldown_remaining, 1.0,
            "dead entity's cooldown should not tick down"
        );
    }

    // -------------------------------------------------------------------
    // weapon_switch_system tests
    // -------------------------------------------------------------------

    #[test]
    fn switch_changes_active_weapon() {
        let mut world = World::new();
        let action_space = ActionSpaceDef::new(vec![
            crate::action_space::ActionHead::Discrete { name: "move_target".into(), n: 12 },
            crate::action_space::ActionHead::Continuous { name: "aim_delta".into(), size: 1, low: vec![-1.0], high: vec![1.0] },
            crate::action_space::ActionHead::Discrete { name: "shoot".into(), n: 2 },
            crate::action_space::ActionHead::Discrete { name: "weapon_select".into(), n: 2 },
        ]);
        world.insert_resource(action_space);

        let inv = make_inventory(); // active = 0
        let entity = world.spawn(inv).id();

        let mut raw_buffer = RawActionBuffer::default();
        // Actions: [move_target=0, aim_delta=0, shoot=0, weapon_select=1]
        raw_buffer.insert(entity, vec![0.0, 0.0, 0.0, 1.0]);
        world.insert_resource(raw_buffer);

        let mut schedule = Schedule::default();
        schedule.add_systems(weapon_switch_system);
        schedule.run(&mut world);

        let inv = world.get::<Inventory>(entity).unwrap();
        assert_eq!(inv.active, 1, "should switch to weapon index 1 (shotgun)");
    }

    #[test]
    fn switch_applies_cooldown_penalty() {
        let mut world = World::new();
        let action_space = ActionSpaceDef::new(vec![
            crate::action_space::ActionHead::Discrete { name: "move_target".into(), n: 12 },
            crate::action_space::ActionHead::Continuous { name: "aim_delta".into(), size: 1, low: vec![-1.0], high: vec![1.0] },
            crate::action_space::ActionHead::Discrete { name: "shoot".into(), n: 2 },
            crate::action_space::ActionHead::Discrete { name: "weapon_select".into(), n: 2 },
        ]);
        world.insert_resource(action_space);

        let inv = make_inventory();
        let entity = world.spawn(inv).id();

        let mut raw_buffer = RawActionBuffer::default();
        raw_buffer.insert(entity, vec![0.0, 0.0, 0.0, 1.0]);
        world.insert_resource(raw_buffer);

        let mut schedule = Schedule::default();
        schedule.add_systems(weapon_switch_system);
        schedule.run(&mut world);

        let inv = world.get::<Inventory>(entity).unwrap();
        assert!(
            (inv.weapons[1].cooldown_remaining - 0.3).abs() < 0.001,
            "switched weapon should have 0.3s cooldown penalty, got {}",
            inv.weapons[1].cooldown_remaining
        );
    }

    #[test]
    fn no_switch_when_same_weapon_selected() {
        let mut world = World::new();
        let action_space = ActionSpaceDef::new(vec![
            crate::action_space::ActionHead::Discrete { name: "move_target".into(), n: 12 },
            crate::action_space::ActionHead::Continuous { name: "aim_delta".into(), size: 1, low: vec![-1.0], high: vec![1.0] },
            crate::action_space::ActionHead::Discrete { name: "shoot".into(), n: 2 },
            crate::action_space::ActionHead::Discrete { name: "weapon_select".into(), n: 2 },
        ]);
        world.insert_resource(action_space);

        let inv = make_inventory(); // active = 0
        let entity = world.spawn(inv).id();

        let mut raw_buffer = RawActionBuffer::default();
        // weapon_select=0 (same as active)
        raw_buffer.insert(entity, vec![0.0, 0.0, 0.0, 0.0]);
        world.insert_resource(raw_buffer);

        let mut schedule = Schedule::default();
        schedule.add_systems(weapon_switch_system);
        schedule.run(&mut world);

        let inv = world.get::<Inventory>(entity).unwrap();
        assert_eq!(inv.active, 0, "active should remain 0");
        assert_eq!(
            inv.weapons[0].cooldown_remaining, 0.0,
            "no penalty when not switching"
        );
    }

    #[test]
    fn no_switch_when_index_out_of_bounds() {
        let mut world = World::new();
        let action_space = ActionSpaceDef::new(vec![
            crate::action_space::ActionHead::Discrete { name: "move_target".into(), n: 12 },
            crate::action_space::ActionHead::Continuous { name: "aim_delta".into(), size: 1, low: vec![-1.0], high: vec![1.0] },
            crate::action_space::ActionHead::Discrete { name: "shoot".into(), n: 2 },
            crate::action_space::ActionHead::Discrete { name: "weapon_select".into(), n: 2 },
        ]);
        world.insert_resource(action_space);

        let inv = make_inventory(); // 2 weapons
        let entity = world.spawn(inv).id();

        let mut raw_buffer = RawActionBuffer::default();
        // weapon_select=5 (out of bounds)
        raw_buffer.insert(entity, vec![0.0, 0.0, 0.0, 5.0]);
        world.insert_resource(raw_buffer);

        let mut schedule = Schedule::default();
        schedule.add_systems(weapon_switch_system);
        schedule.run(&mut world);

        let inv = world.get::<Inventory>(entity).unwrap();
        assert_eq!(inv.active, 0, "should remain at index 0 for invalid select");
    }

    // -------------------------------------------------------------------
    // WeaponSlot helper tests
    // -------------------------------------------------------------------

    #[test]
    fn ammo_depletes_on_fire_and_auto_reload_triggers() {
        let mut slot = make_rifle();
        slot.ammo = 1; // last round

        // Simulate firing
        slot.ammo -= 1;
        slot.cooldown_remaining = slot.fire_rate;
        if slot.ammo == 0 {
            slot.is_reloading = true;
            slot.reload_remaining = slot.reload_time;
        }

        assert_eq!(slot.ammo, 0);
        assert!(slot.is_reloading, "should auto-reload when ammo hits 0");
        assert!(
            (slot.reload_remaining - 2.0).abs() < 0.001,
            "reload_remaining should be set to reload_time"
        );
    }

    #[test]
    fn cannot_fire_while_reloading() {
        let mut slot = make_rifle();
        slot.is_reloading = true;
        slot.reload_remaining = 1.0;
        slot.ammo = 0;

        // The combat system checks: ammo > 0 && !is_reloading && cooldown <= 0
        let can_fire = slot.ammo > 0 && !slot.is_reloading && slot.cooldown_remaining <= 0.0;
        assert!(!can_fire, "should not be able to fire while reloading");
    }

    #[test]
    fn cannot_fire_with_zero_ammo() {
        let mut slot = make_rifle();
        slot.ammo = 0;
        slot.is_reloading = false;

        let can_fire = slot.ammo > 0 && !slot.is_reloading && slot.cooldown_remaining <= 0.0;
        assert!(!can_fire, "should not be able to fire with 0 ammo");
    }
}

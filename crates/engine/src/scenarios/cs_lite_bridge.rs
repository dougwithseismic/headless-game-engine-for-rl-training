use bevy_ecs::prelude::*;
use glam::Vec3;

use crate::ecs::components::*;
use crate::strategy::bridge::StrategyBridge;
use crate::strategy::types::*;

use super::cs_lite::*;

#[derive(Component, Debug, Clone)]
pub struct StrategyControlled;

pub struct CsLiteBridge;

impl StrategyBridge for CsLiteBridge {
    fn snapshot(&self, world: &mut World) -> StateSnapshot {
        let tick = world.resource::<crate::ecs::resources::TickState>().tick;
        let round = world.resource::<CsRoundState>().clone();
        let bomb = world.resource::<BombState>().clone();
        let bomb_sites = world.resource::<BombSites>().clone();

        let mut agents = Vec::new();
        let mut query = world.query::<(
            &Agent,
            &Team,
            &Position3D,
            &Health,
            Option<&Dead>,
            Option<&BombCarrier>,
            Option<&CsSide>,
        )>();
        for (agent, team, pos, health, dead, carrier, side) in query.iter(world) {
            let mut custom = serde_json::Map::new();
            if carrier.is_some() {
                custom.insert("bomb_carrier".into(), serde_json::Value::Bool(true));
            }
            if let Some(s) = side {
                let side_str = match s {
                    CsSide::Terrorist => "terrorist",
                    CsSide::CounterTerrorist => "counter_terrorist",
                };
                custom.insert("side".into(), serde_json::Value::String(side_str.into()));
            }

            agents.push(AgentSnapshot {
                id: agent.source_id,
                team: team.0,
                position: [pos.0.x, pos.0.y, pos.0.z],
                health: health.current,
                alive: dead.is_none(),
                custom: serde_json::Value::Object(custom),
            });
        }
        agents.sort_by_key(|a| a.id);

        let phase_str = match round.phase {
            RoundPhase::BuyFreeze => "buy_freeze",
            RoundPhase::Active => "active",
            RoundPhase::RoundEnd => "round_end",
        };

        let bomb_carrier_id = bomb.carrier.and_then(|e| {
            world.get::<Agent>(e).map(|a| a.source_id)
        });

        let structured = serde_json::json!({
            "round": round.round_number,
            "phase": phase_str,
            "phase_timer": round.phase_timer,
            "t_score": round.t_score,
            "ct_score": round.ct_score,
            "t_alive": round.t_alive,
            "ct_alive": round.ct_alive,
            "bomb_planted": bomb.planted,
            "bomb_defused": bomb.defused,
            "bomb_detonated": bomb.detonated,
            "bomb_carrier": bomb_carrier_id,
            "bomb_plant_position": bomb.plant_position.map(|p| [p.x, p.y, p.z]),
            "bomb_plant_site": bomb.plant_site,
            "site_a_center": [bomb_sites.site_a_center.x, bomb_sites.site_a_center.y, bomb_sites.site_a_center.z],
            "site_b_center": [bomb_sites.site_b_center.x, bomb_sites.site_b_center.y, bomb_sites.site_b_center.z],
        });

        let t_alive = agents.iter().filter(|a| a.team == 0 && a.alive).count();
        let ct_alive = agents.iter().filter(|a| a.team == 1 && a.alive).count();
        let summary = format!(
            "Round {} ({}) | T {}-{} CT | Alive: T {} CT {} | Bomb: {}",
            round.round_number,
            phase_str,
            round.t_score,
            round.ct_score,
            t_alive,
            ct_alive,
            if bomb.planted {
                "planted"
            } else if bomb.carrier.is_some() {
                "carried"
            } else {
                "dropped"
            }
        );

        StateSnapshot {
            tick,
            scenario: "cs_lite".into(),
            summary,
            structured,
            agents,
        }
    }

    fn available_intents(&self, world: &mut World) -> Vec<IntentSpec> {
        let bomb = world.resource::<BombState>().clone();
        let round = world.resource::<CsRoundState>().clone();

        let mut intents = vec![
            IntentSpec {
                name: "eliminate".into(),
                description: "Aggressively seek and frag enemies".into(),
                params: vec![ParamSpec {
                    name: "posture".into(),
                    param_type: ParamType::Enum(vec![
                        "aggressive".into(),
                        "default".into(),
                        "passive".into(),
                    ]),
                    description: "Engagement posture".into(),
                }],
                scope: IntentScope::Agent,
            },
            IntentSpec {
                name: "hold_position".into(),
                description: "Hold current position and watch angles".into(),
                params: vec![
                    ParamSpec {
                        name: "position".into(),
                        param_type: ParamType::Position,
                        description: "Position to hold (optional, defaults to current)".into(),
                    },
                    ParamSpec {
                        name: "posture".into(),
                        param_type: ParamType::Enum(vec![
                            "aggressive".into(),
                            "default".into(),
                            "passive".into(),
                        ]),
                        description: "Engagement posture".into(),
                    },
                ],
                scope: IntentScope::Agent,
            },
            IntentSpec {
                name: "rotate".into(),
                description: "Rotate to a new position on the map".into(),
                params: vec![
                    ParamSpec {
                        name: "position".into(),
                        param_type: ParamType::Position,
                        description: "Target position to rotate to".into(),
                    },
                    ParamSpec {
                        name: "posture".into(),
                        param_type: ParamType::Enum(vec![
                            "aggressive".into(),
                            "default".into(),
                            "passive".into(),
                        ]),
                        description: "Movement posture".into(),
                    },
                ],
                scope: IntentScope::Agent,
            },
        ];

        if round.phase == RoundPhase::Active {
            if !bomb.planted {
                intents.push(IntentSpec {
                    name: "plant_bomb".into(),
                    description: "Plant the bomb at a site".into(),
                    params: vec![ParamSpec {
                        name: "site".into(),
                        param_type: ParamType::Enum(vec!["A".into(), "B".into()]),
                        description: "Bomb site to plant at".into(),
                    }],
                    scope: IntentScope::Agent,
                });
            }
            if bomb.planted {
                intents.push(IntentSpec {
                    name: "defuse_bomb".into(),
                    description: "Go to bomb and defuse it".into(),
                    params: vec![ParamSpec {
                        name: "position".into(),
                        param_type: ParamType::Position,
                        description: "Bomb position (auto-filled if omitted)".into(),
                    }],
                    scope: IntentScope::Agent,
                });
            }
        }

        intents
    }

    fn apply_directive(&self, world: &mut World, directive: &Directive) -> u32 {
        let bomb_sites = world.resource::<BombSites>().clone();
        let bomb = world.resource::<BombState>().clone();

        let objective = match directive.intent.as_str() {
            "plant_bomb" => ObjectiveType::PlantBomb,
            "defuse_bomb" => ObjectiveType::DefuseBomb,
            "hold_position" => ObjectiveType::HoldPosition,
            "eliminate" => ObjectiveType::Eliminate,
            "rotate" => ObjectiveType::Rotate,
            _ => return 0,
        };

        let posture = match directive
            .params
            .get("posture")
            .and_then(|v| v.as_str())
        {
            Some("aggressive") => Posture::Aggressive,
            Some("passive") => Posture::Passive,
            _ => Posture::Default,
        };

        let position = extract_vec3_param(&directive.params, "position");
        let site = directive.params.get("site").and_then(|v| v.as_str());

        let target_position = match objective {
            ObjectiveType::PlantBomb => match site {
                Some("B") => bomb_sites.site_b_center,
                _ => bomb_sites.site_a_center,
            },
            ObjectiveType::DefuseBomb => {
                position.unwrap_or_else(|| {
                    bomb.plant_position.unwrap_or(bomb_sites.site_a_center)
                })
            }
            ObjectiveType::Rotate | ObjectiveType::HoldPosition => {
                position.unwrap_or(Vec3::ZERO)
            }
            ObjectiveType::Eliminate => position.unwrap_or(Vec3::ZERO),
        };

        let goal = AgentGoal {
            objective,
            target_position,
            posture,
        };

        let mut applied = 0u32;

        // Find matching entities by source_id
        let mut matches: Vec<Entity> = Vec::new();
        let mut query = world.query::<(Entity, &Agent)>();
        for (entity, agent) in query.iter(world) {
            if directive.target_agents.is_empty()
                || directive.target_agents.contains(&agent.source_id)
            {
                matches.push(entity);
            }
        }

        for entity in matches {
            world.entity_mut(entity).insert(goal.clone());
            world.entity_mut(entity).insert(StrategyControlled);
            applied += 1;
        }

        applied
    }
}

fn extract_vec3_param(params: &serde_json::Value, key: &str) -> Option<Vec3> {
    let arr = params.get(key)?.as_array()?;
    if arr.len() >= 3 {
        Some(Vec3::new(
            arr[0].as_f64()? as f32,
            arr[1].as_f64()? as f32,
            arr[2].as_f64()? as f32,
        ))
    } else {
        None
    }
}

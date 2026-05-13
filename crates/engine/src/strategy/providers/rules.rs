use crate::strategy::*;

pub struct RulesProvider;

impl StrategyProvider for RulesProvider {
    fn decide(&self, snapshot: &StateSnapshot, _intents: &[IntentSpec]) -> Vec<Directive> {
        let mut directives = Vec::new();

        let bomb_planted = snapshot
            .structured
            .get("bomb_planted")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let bomb_carrier = snapshot
            .structured
            .get("bomb_carrier")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let bomb_site_a = extract_position(&snapshot.structured, "site_a_center");
        let bomb_plant_pos = extract_position(&snapshot.structured, "bomb_plant_position");

        for agent in &snapshot.agents {
            if !agent.alive {
                continue;
            }

            let is_t_side = agent.team == 0;

            let directive = if is_t_side {
                if bomb_planted {
                    Directive {
                        intent: "hold_position".into(),
                        target_agents: vec![agent.id],
                        params: serde_json::json!({}),
                        reasoning: Some("Bomb planted, hold and protect".into()),
                    }
                } else if bomb_carrier == Some(agent.id) {
                    Directive {
                        intent: "plant_bomb".into(),
                        target_agents: vec![agent.id],
                        params: serde_json::json!({ "site": "A" }),
                        reasoning: Some("Carrier should plant at nearest site".into()),
                    }
                } else {
                    Directive {
                        intent: "eliminate".into(),
                        target_agents: vec![agent.id],
                        params: serde_json::json!({}),
                        reasoning: Some("Support push by fragging".into()),
                    }
                }
            } else if bomb_planted {
                let target = bomb_plant_pos.unwrap_or(bomb_site_a.unwrap_or([0.0; 3]));
                Directive {
                    intent: "defuse_bomb".into(),
                    target_agents: vec![agent.id],
                    params: serde_json::json!({ "position": target }),
                    reasoning: Some("Bomb planted, rush to defuse".into()),
                }
            } else {
                Directive {
                    intent: "hold_position".into(),
                    target_agents: vec![agent.id],
                    params: serde_json::json!({}),
                    reasoning: Some("Hold site and wait for T push".into()),
                }
            };

            directives.push(directive);
        }

        directives
    }

    fn decision_interval(&self) -> u64 {
        64
    }

    fn name(&self) -> &str {
        "rules"
    }
}

fn extract_position(value: &serde_json::Value, key: &str) -> Option<[f32; 3]> {
    let arr = value.get(key)?.as_array()?;
    if arr.len() >= 3 {
        Some([
            arr[0].as_f64()? as f32,
            arr[1].as_f64()? as f32,
            arr[2].as_f64()? as f32,
        ])
    } else {
        None
    }
}

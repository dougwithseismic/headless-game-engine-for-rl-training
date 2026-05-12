# Eureka-Style LLM-Guided Reward Search for FPS Agents

Applying NVIDIA Eureka's framework to GhostLobby: an LLM generates reward function variants, we run them in parallel, behavioral metrics determine winners, the LLM iterates. Nobody has done this for game AI — all existing work is robotics/locomotion.

## References

- [Eureka](https://arxiv.org/abs/2310.12931) (NVIDIA, ICLR 2024) — GPT-4 writes reward code, evolutionary search. Beat human rewards on 83% of tasks. [Code](https://github.com/eureka-research/Eureka)
- [DrEureka](https://eureka-research.github.io/dr-eureka/) — extends to physics randomization configs
- [Text2Reward](https://arxiv.org/abs/2309.11489) (ICLR 2024 Spotlight) — natural language → reward code
- [Language to Rewards](https://arxiv.org/abs/2306.08647) (DeepMind) — LLM → reward parameters, interactive corrections
- [CARD](https://arxiv.org/abs/2410.14660) — trajectory preference evaluation, faster iteration
- [LEARN-Opt](https://arxiv.org/abs/2511.19355) — fully autonomous, model-agnostic

## What We Have Already

| Component | Status | How Eureka Uses It |
|-----------|--------|-------------------|
| Headless game engine (19K TPS) | Built | Eureka needs a fast simulator |
| Parallel envs (32 via SubprocVecEnv) | Built | Eureka runs candidates in parallel |
| Behavioral metrics (accuracy, kills, damage, shoot rate) | Built | Eureka's fitness function |
| Reward breakdown (per-component tracking) | Built | Eureka evaluates which reward terms help |
| BC warm-start pipeline | Built | Gives every candidate a reasonable starting policy |
| PPO training with SB3 | Built | Eureka uses PPO for each candidate |
| Replay recording | Built | For trajectory-based evaluation (CARD approach) |
| TensorBoard logging | Built | Monitoring and comparison |
| Config-driven rewards (JSON) | Partially — rewards are hardcoded in Rust | Needs to be configurable |

## What We Need to Build

### 1. Config-Driven Reward Function (Rust)

Currently rewards are hardcoded in `cs_lite.rs` across 4 systems. Need to make them data-driven so the LLM can generate reward configs without recompiling.

**Approach:** Add a `RewardConfig` resource loaded from the JSON config's `extra` block:

```json
{
  "extra": {
    "rewards": {
      "kill": 3.0,
      "death": 0.0,
      "damage_dealt_per_hp": 1.0,
      "damage_taken_per_hp": 0.0,
      "round_win": 5.0,
      "round_loss": 0.0,
      "near_miss": 0.03,
      "friendly_fire": -2.0,
      "aim_dot_bonus": 0.0,
      "proximity_bonus": 0.0,
      "cover_bonus": 0.0,
      "exposure_penalty": 0.0,
      "idle_penalty": 0.0,
      "exploration_bonus": 0.0
    }
  }
}
```

Each `rewards.add()` call in the Rust code reads from this config instead of hardcoded values. Zero recompilation between candidates. The LLM generates JSON, not Rust code.

**Files to change:**
- `crates/engine/src/scenarios/cs_lite.rs` — add `RewardConfig` struct, load from GameConfig.extra, replace hardcoded values
- Config JSON files — add `rewards` block

### 2. Parallel Candidate Runner (Python)

A script that launches N training runs simultaneously, each with a different reward config.

```python
# eureka_search.py
class EurekaSearch:
    def __init__(self, base_config, n_candidates=8, steps_per_round=500_000):
        self.base_config = base_config
        self.n_candidates = n_candidates
        self.steps_per_round = steps_per_round
        self.llm = ClaudeClient()  # or OpenAI
        self.history = []

    def run_round(self, reward_configs: list[dict]) -> list[dict]:
        """Run N candidates in parallel, return behavioral metrics for each."""
        results = []
        # Launch N training processes (can use subprocess or multiprocessing)
        # Each gets a modified config JSON with different reward values
        # All share the same BC warm-start model
        # Run for steps_per_round steps
        # Collect eval metrics from TensorBoard/evaluations.npz
        return results

    def ask_llm(self, round_history: list) -> list[dict]:
        """LLM sees previous results, generates next batch of reward configs."""
        prompt = f"""
        You are designing reward functions for an FPS game agent.
        
        Here are the results from previous rounds:
        {format_history(round_history)}
        
        The behavioral metrics that matter:
        - accuracy: % of shots that hit (higher = better aim)
        - kills_per_ep: kills per episode (higher = more lethal)
        - shoot_rate: % of steps where agent fires (0.1-0.3 is healthy)
        - damage_dealt_per_ep: total damage output
        - deaths_per_ep: times agent died (lower = better survival)
        
        Generate {self.n_candidates} reward config variants as JSON.
        Each config has these keys with float values:
        kill, death, damage_dealt_per_hp, damage_taken_per_hp,
        round_win, round_loss, near_miss, friendly_fire,
        aim_dot_bonus, proximity_bonus, cover_bonus,
        exposure_penalty, idle_penalty, exploration_bonus
        
        Explore diverse strategies. Mutate the best performers.
        Avoid configs similar to ones that produced 0% accuracy.
        """
        return self.llm.generate(prompt)

    def search(self, n_rounds=10):
        """Full Eureka search loop."""
        # Round 0: LLM generates initial diverse candidates
        configs = self.ask_llm([])
        
        for round in range(n_rounds):
            results = self.run_round(configs)
            self.history.append({"configs": configs, "results": results})
            
            # Log best candidate
            best = max(results, key=lambda r: r["accuracy"] + r["kills_per_ep"])
            print(f"Round {round}: best accuracy={best['accuracy']:.1%}, kills={best['kills_per_ep']:.1f}")
            
            # LLM generates next round
            configs = self.ask_llm(self.history)
        
        return self.history
```

**Key design decisions:**
- N candidates share the same BC warm-start (fair comparison)
- 500K steps per round (~2.5 minutes at 3300 FPS per candidate)
- 8 candidates = 4 envs each from our 32 pool, or run sequentially on all 32
- Sequential is simpler and still fast: 8 × 2.5min = 20 minutes per round
- 10 rounds = ~3.5 hours total to search 80 reward configurations

### 3. Results Aggregator

Reads eval metrics from each candidate's run directory and formats them for the LLM.

```python
def aggregate_results(run_dir: str) -> dict:
    """Read behavioral metrics from a training run."""
    # Read from TensorBoard event files or evaluations.npz
    # Return dict with accuracy, kills_per_ep, shoot_rate, etc.
```

Already partially built — the `BehaviorEvalCallback` writes all this data. Just needs a reader function.

### 4. LLM Prompt Engineering

The prompt to the LLM needs to include:
- The reward config schema (which knobs exist)
- Results from all previous rounds (what worked, what didn't)
- Behavioral metric definitions (what each number means)
- Constraints (e.g., "near_miss should be small, kill should be positive")
- Instructions to explore diverse strategies, not just mutate the best

Eureka's key insight: the LLM generates CODE (reward functions), not just parameters. For us, generating JSON parameter configs is simpler and sufficient — the reward function structure is fixed, only the weights change.

But we could also let the LLM generate conditional reward logic:
```json
{
  "kill": 3.0,
  "kill_from_cover": 5.0,
  "kill_while_moving": 4.0,
  "headshot_bonus": 2.0
}
```

This requires the Rust reward code to support more granular conditions, but each condition is just a flag check — straightforward to add.

### 5. Curriculum Integration

After Eureka finds the best Phase 1 reward config, use it as the starting point for Phase 2 search:

```
Phase 1 (vs dummy AI):
  Eureka searches reward configs
  Finds: kill=5, damage=2, near_miss=0.1, everything else 0
  Agent reaches 30% accuracy, 80% win rate

Phase 2 (vs scripted AI):
  Start from Phase 1 best config
  Re-enable death penalty, add cover/positioning rewards
  Eureka searches new configs building on Phase 1 winner
  Agent learns tactical play on top of combat fundamentals

Phase 3 (self-play):
  Start from Phase 2 best config
  Eureka fine-tunes reward weights for self-play stability
```

Each phase inherits the best model AND best reward config from the previous phase.

## Implementation Order

### Phase A: Config-driven rewards (Rust, 2-3 hours)
1. Define `RewardConfig` struct with all reward weights
2. Load from `GameConfig.extra.rewards` with defaults
3. Replace hardcoded values in cs_combat_system, cs_death_system, cs_round_state_system, cs_bomb_system
4. Verify: change config JSON, see different rewards in behavioral metrics without recompiling

### Phase B: Candidate runner (Python, 3-4 hours)
1. Script that takes N reward config JSONs
2. Launches N sequential training runs (each 500K steps)
3. Collects behavioral metrics from each
4. Outputs a comparison table

### Phase C: LLM loop (Python, 2-3 hours)
1. Prompt template for reward generation
2. Claude/GPT-4 API integration
3. Parse LLM output → JSON configs
4. Wire into candidate runner
5. Full loop: LLM → configs → train → metrics → LLM

### Phase D: Curriculum chaining (Python, 1-2 hours)
1. Best config from Phase N becomes starting point for Phase N+1
2. Best model checkpoint carries forward
3. LLM receives phase context ("now the opponent is harder, you may need death penalties")

## Expected Outcomes

- **Time to find good reward config:** ~3.5 hours (10 rounds × 8 candidates × 500K steps)
- **Compared to manual tuning:** We spent 12+ hours and tried ~8 reward configs manually
- **Quality:** Eureka outperformed human-designed rewards on 83% of robotics tasks. Similar improvement plausible for game AI.
- **Novelty:** First application of Eureka-style reward search to FPS game agents. All existing work is robotics.

## Risks

- **Reward hacking at scale:** LLM might generate configs that look good on metrics but produce degenerate behavior. Mitigated by recording replays for manual review of top candidates.
- **LLM hallucination:** The LLM might suggest nonsensical reward values. Mitigated by clamping all values to sane ranges and validating JSON schema.
- **Compute cost:** 80 training runs × 500K steps = 40M total steps = ~3.5 hours on our hardware. Acceptable.
- **LLM API cost:** ~10 rounds × ~2K tokens per prompt × ~4K tokens per response = ~60K tokens total. At Claude Sonnet pricing: ~$0.30. Negligible.

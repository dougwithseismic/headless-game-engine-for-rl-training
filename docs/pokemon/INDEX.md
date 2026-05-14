# Pokemon Gold RL: Project State & Direction

## Current State (Honest)

A PPO agent that navigates Pokemon Gold's overworld. It walks west from the player's house in New Bark Town to Route 29. It does this consistently in honest stochastic eval. It does not play Pokemon in any strategic sense -- it cannot battle, use menus, or make progress through the story.

### What works

- **Overworld movement**: The agent learns to move efficiently and explore new tiles using count-based exploration rewards (1/sqrt(N) diminishing returns).
- **Route 29 navigation**: At 20K steps with milestone reward, the agent reaches Route 29 in all eval checkpoints. Verified via stochastic policy eval (not deterministic argmax, which collapses).
- **Honest eval numbers**: 150-208 unique tiles, 3-4 maps, 47-93% movement rate, Route 29 reached consistently.
- **CNN + RAM observations**: MultiInputPolicy processes screen (36x40 grayscale) + RAM (33 features including 8 temporal deltas). The agent can see walls and paths.
- **6-button action space**: Removing Start/Select eliminated the menu trap problem entirely (was 95% stuck in menus with 8 buttons).
- **Telemetry**: High-water-mark tracking for level/badge/party events filters out GBC RAM glitches. No more false level-up reports.
- **Save state management**: Fast resets (~5ms), backup system, A-press burst to clear dialogue on load.

### What doesn't work

- **No strategic play**: The agent moves around but doesn't understand game objectives, NPCs, or progression.
- **No battle capability**: Battles are entered by accident (walking into tall grass). The agent presses random buttons until the battle ends.
- **No menu access**: Start/Select removed from action space means no items, no HMs, no saving, no Pokemon management.
- **Exploration collapse**: With older reward functions, the agent found small efficient loops after 2M steps. The milestone reward mitigates this with large one-time bonuses for reaching new maps, but the underlying problem remains for long training runs.
- **Throughput ceiling**: ~300 steps/sec with CNN+RAM (1 env), ~700 steps/sec RAM-only. This is fine for the current scope but slow for million-step runs.

## Architecture

```
train_gold.py
  |
  +-- make_pokemon_gold_bridge()
  |     +-- PyBoyHost              (GBC emulator, cgb=True)
  |     +-- PyBoyActionSink        (MultiBinary(6): up/down/left/right/A/B)
  |     +-- PyBoyObservationSource (25 RAM features + optional 36x40 screen)
  |     +-- PyBoyReset             (save state resets, ~5ms)
  |
  +-- ExternalGameGym              (Gymnasium wrapper)
  |     +-- reward_fn              (exploration / progress / milestone / battle)
  |
  +-- TemporalObsWrapper           (+8 delta features)
  +-- AntiLoopWrapper              (soft penalties for spam/cycling/idle)
  +-- PokemonTelemetryWrapper      (game event detection, stats tracking)
  +-- MultiInputWrapper            (splits flat obs -> Dict{screen, ram} for CNN+MLP)
```

## Reward Functions (4 options)

| Reward | Purpose | Best for |
|--------|---------|----------|
| `exploration` | Count-based 1/sqrt(N) per tile | General exploration, no collapse |
| `progress` | Exploration + badge/level/party bonuses | Longer runs with game milestones |
| `milestone` | Map-based progression (+50 Route 29) + exploration + dialogue + level/badge | Curriculum phases with specific map goals |
| `battle` | Win/loss/HP preserved | Battle training from encounter save states |

The **milestone** reward is the current best. It gives large one-time bonuses for reaching specific maps, diminishing-return exploration between milestones, dialogue advancement bonus, and level/badge rewards -- all with high-water-mark tracking.

## The Two Questions

### Question A: "Can an agent learn to play Pokemon from scratch?"

No, not with current methods. A flat PPO with 33-dim RAM + screen observations learns to walk around but can't learn the multi-context game (overworld movement vs battle vs menu vs dialogue all use the same buttons for different purposes). This would need 10M-100M+ steps, recurrent memory, and much richer reward shaping. PokemonRedExperiments-style projects take weeks of GPU training.

### Question B: "Can we build an AI system that plays Pokemon competently?"

This is the right question and the product-relevant one. It requires a hierarchical system:

```
+-----------------------------------------------+
|            VLM Strategy Layer                  |
|  Runs every ~200 steps                        |
|  Input: screenshot + RAM state summary        |
|  Output: goal for the RL motor                |
+--------------------+--------------------------+
                     | goal
+--------------------v--------------------------+
|    State Detector (RAM-based)                  |
|  battle_mode > 0 -> BATTLE                    |
|  textbox_flags > 0 -> DIALOGUE                |
|  else -> OVERWORLD                            |
+------+----------+----------+-----------+------+
       |          |          |           |
  +----+---+ +----+----+ +--+------+ +--+-----+
  |Overworld| |  Battle | |Dialogue| |  Menu   |
  |Explorer | |   AI    | |Handler | |Navigator|
  | (RL)    | |(RL/MCTS)| |(Script)| |(Script) |
  +---------+ +---------+ +--------+ +--------+
```

The overworld explorer is what we have now. The rest needs building.

## The Start Button Problem

Removing Start/Select was essential for training (95% menu trap with 8 buttons) but creates a hard ceiling:

- Can't use items (potions, repels, TMs)
- Can't teach HMs (Cut, Surf, Fly) -- required for game progression
- Can't check Pokemon status or manage party
- Can't save the game

This is the strongest argument for the hierarchical approach: a high-level planner decides WHEN to open menus and WHAT to do in them (via scripted subroutines), while the RL motor handles overworld navigation.

## Curriculum Plan

| Phase | Goal | Start state | Reward | Steps | Status |
|-------|------|-------------|--------|-------|--------|
| 0 | Movement + dialogue | New Bark Town | milestone | 500K | Validated (20K steps reaches Route 29) |
| 1 | Leave town, reach Route 29 | New Bark Town | milestone | 1M | Ready |
| 2 | Navigate Route 29, reach Cherrygrove | Route 29 save state | milestone | 1M | Needs Cherrygrove map values |
| 3 | Battle training | Wild encounter save state | battle | 500K | Needs save state |
| 4 | Full route New Bark -> Cherrygrove | New Bark Town | all milestones | 2M | Needs phases 1-3 done |

## Known Map Values

Discovered via manual play and RAM reading:

| Location | (map_group, map_number) | Notes |
|----------|------------------------|-------|
| New Bark Town | (24, 4) | Starting town |
| Route 29 | (24, 3) | Entry at x=59 (west edge of map) |
| Elm's Lab | (24, 5) | |
| Player's House 1F | (24, 6) | |
| Player's House 2F | (24, 7) | |
| Cherrygrove City | TBD | Need to discover via manual play |
| Route 30 | TBD | |

## Reward Design Lessons

### What failed

1. **Revisit penalties (-0.01 per revisited tile)**: Net reward was negative because revisits outnumber discoveries. The rational strategy was "don't move."
2. **Binary visited set**: Once nearby tiles were all visited, zero reward for movement. Agent stopped.
3. **Harsh anti-loop penalties (-0.1 to -0.3)**: Combined with revisit penalties, the agent was punished for doing anything.
4. **Battle/HP penalties**: Discouraged engagement with core game mechanics.
5. **Deterministic eval**: Collapsed to wall-walking. Stochastic eval shows the actual learned policy distribution.

### What works

1. **Positive-only rewards**: Never punish, only reward. The agent always has incentive to try things.
2. **Count-based exploration (1/sqrt(N))**: Diminishing returns but never zero. The 100th visit to a tile still gives 2.0/sqrt(100) = 0.2 reward.
3. **Anti-bounce detection**: Don't reward returning to the previous position. Prevents A-B oscillation exploit.
4. **Soft anti-loop**: Very mild penalties (-0.02 to -0.05 as configured in train_gold.py) that nudge without dominating.
5. **High-water-mark tracking**: Level/badge/party events only fire when value exceeds previous max. Filters GBC RAM glitches (5->0->5 doesn't trigger).
6. **Milestone bonuses**: +50 for reaching Route 29 gives a clear directional signal. The agent learns to go west.
7. **Dialogue bonus**: +0.1 for pressing A when textbox active prevents the agent from getting stuck on NPC dialogue.

## Connection to GhostLobby Platform

The Pokemon Gold work validates three things:

1. **The game bridge framework works end-to-end**: ROM -> PyBoy -> banked WRAM reads -> observation protocol -> wrappers -> Gymnasium -> PPO -> TensorBoard. Same architecture plugs into SNES, N64, GBA emulators, and eventually Assetto Corsa/Unity/Unreal.

2. **Reward design is the hard problem**: The bridge, wrappers, and training loop were built quickly. Getting the reward function to produce useful behavior took multiple iterations and several failed approaches. This is where Eureka-style LLM reward search has the most value.

3. **Hierarchical is the right architecture for complex games**: A flat PPO can learn to walk around but can't play Pokemon. The same applies to FPS (flat PPO can aim but can't plan). The hierarchical agent framework (VLM strategy + skills + RL motor) maps directly onto Pokemon (planner + battle AI + overworld explorer).

## Documents

- [README.md](README.md) -- Quick start, CLI reference, observation/action space docs
- [TRAINING_RESULTS.md](TRAINING_RESULTS.md) -- Training run results, bugs found and fixed, honest eval numbers
- [VLM_DIRECTION.md](VLM_DIRECTION.md) -- Vision-language model strategy layer (tested: Gemma 4 locally, 12s/call)

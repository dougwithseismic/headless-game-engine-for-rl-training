# Vision-Language Model Direction for Pokemon Gold

## The Idea

Instead of training a CNN from scratch to understand Pokemon screenshots, use a pre-trained vision-language model (VLM) to interpret what's on screen. The VLM already understands Pokemon — it's been trained on millions of screenshots, walkthroughs, and wikis.

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  VLM Strategy Layer               │
│  Runs every ~100-500 steps                       │
│  Input: full-res screenshot + game state summary │
│  Output: structured goal for the RL motor        │
│                                                  │
│  "You're in New Bark Town. Prof. Elm's lab is    │
│   to the right. Go talk to him to get your       │
│   starter Pokemon."                              │
│                                                  │
│  → goal: "walk_to" (7, 3) on map (24, 5)        │
└──────────────┬───────────────────────────────────┘
               │ goal every ~100 steps
┌──────────────▼───────────────────────────────────┐
│              RL Motor (PPO)                       │
│  Runs every step (~24 emulator frames)           │
│  Input: RAM features + screen + current goal     │
│  Output: button press                            │
│                                                  │
│  Reward: distance to goal + exploration bonus    │
└──────────────────────────────────────────────────┘
```

The VLM doesn't run every frame — it's too slow for that. It runs periodically (every 100-500 agent steps ≈ every 40-200 seconds of game time) and sets a high-level goal. The RL policy handles low-level execution (navigating menus, walking, pressing A to advance text).

## Why This Could Work

1. **VLMs already understand Pokemon.** A model like Gemma 4 or Claude can identify locations, NPCs, menu states, battle screens, and even read in-game text from a screenshot. Zero training needed for scene understanding.

2. **Eliminates the hardest RL problem.** The agent's biggest failure is not understanding context — it doesn't know that "pressing A on this NPC advances the story" or "walking into tall grass triggers battles." A VLM knows this instantly from a single screenshot.

3. **Separates strategic thinking from motor control.** The VLM decides WHAT to do (go to Elm's lab, pick Cyndaquil, walk to Route 29). The RL policy decides HOW (navigate around obstacles, dismiss text, walk the path). Each component does what it's good at.

4. **Works with the existing bridge framework.** The VLM is just another wrapper — it reads the screen observation, calls the model periodically, and injects the goal into the observation or reward function.

## Model Options

### Local Models (M4 Max)

| Model | Size | Speed (M4 Max) | Vision | Notes |
|-------|------|-----------------|--------|-------|
| **Gemma 3** | 4B-27B | ~5-20 tok/s | Yes (PaliGemma) | Google, open weights, good at visual QA |
| **Llama 3.2 Vision** | 11B-90B | ~3-15 tok/s | Yes | Meta, strong visual understanding |
| **Qwen2.5-VL** | 3B-72B | ~5-25 tok/s | Yes | Alibaba, strong on structured visual tasks |
| **Phi-4-multimodal** | 5.6B | ~10-20 tok/s | Yes | Microsoft, optimized for edge |
| **moondream** | 2B | ~30 tok/s | Yes | Tiny, fast, good enough for game screenshots |

**Recommendation:** Start with **moondream** (2B) for speed — it can process a screenshot in <500ms on M4 Max. Upgrade to Gemma 3 12B if more reasoning is needed.

### API Models

| Model | Speed | Quality | Cost |
|-------|-------|---------|------|
| **Claude Sonnet** | ~1-2s | Excellent | ~$0.003/screenshot |
| **GPT-4o mini** | ~1-2s | Good | ~$0.001/screenshot |
| **Gemini Flash** | ~0.5s | Good | Free tier available |

At 1 call per 500 steps (every ~3 min of game time), 1 hour of training = 20 VLM calls = ~$0.06. Negligible cost.

## Implementation Plan

### Phase 1: Screenshot Interpreter (no RL changes)

A standalone tool that takes a PyBoy screenshot and returns structured game state:

```python
class PokemonVLMInterpreter:
    def __init__(self, model="moondream"):
        self.model = load_model(model)
    
    def interpret(self, screenshot: np.ndarray, ram_state: dict) -> GameContext:
        prompt = f"""Look at this Pokemon Gold screenshot.
        Player is at position ({ram_state['x']}, {ram_state['y']}) on map {ram_state['map']}.
        Party: {ram_state['party_size']} Pokemon, lead is Lv {ram_state['level']}.
        
        Describe:
        1. What location is this? (town name, route, building interior)
        2. What's visible on screen? (NPCs, obstacles, paths, items)
        3. Is there a textbox/menu active? What does it say?
        4. What should the player do next to progress in the game?
        
        Respond as JSON."""
        
        return self.model.generate(screenshot, prompt)
```

This is useful as a debug tool even without RL integration — run it during headed eval to see what the VLM thinks is happening.

### Phase 2: Goal-Conditioned Wrapper

A gym wrapper that periodically queries the VLM and injects a goal into the observation:

```python
class VLMGoalWrapper(gym.Wrapper):
    def __init__(self, env, vlm, goal_interval=200):
        self.vlm = vlm
        self.goal_interval = goal_interval
        self._current_goal = None
    
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        
        if self._step_count % self.goal_interval == 0:
            screenshot = self._get_screenshot()
            self._current_goal = self.vlm.get_goal(screenshot, obs)
        
        # Add goal to observation
        obs_with_goal = self._inject_goal(obs, self._current_goal)
        
        # Add goal-progress reward
        reward += self._goal_reward(obs, self._current_goal)
        
        return obs_with_goal, reward, term, trunc, info
```

### Phase 3: Adaptive Strategy

The VLM adjusts its guidance based on what's happening:

- **Overworld:** "Walk right to reach Route 29. Avoid tall grass until you level up."
- **Dialogue:** "Press A to advance this text. Prof. Elm is asking you to run an errand."
- **Battle:** "Use Tackle (first move). Your Cyndaquil is strong against this Sentret."
- **Menu:** "Press B to close this menu."

The VLM can also detect when the agent is stuck and issue corrective goals: "You've been walking into this wall for 30 seconds. Turn around and go south."

## Why Not Just Use the VLM for Everything?

A VLM-only agent (no RL) could work — call the VLM every step and ask "what button should I press?" But:

1. **Too slow.** Even the fastest local VLMs need 100-500ms per inference. At 60fps game speed, you need <16ms per decision. The VLM can set strategy; the RL motor executes at speed.

2. **VLMs are bad at precise control.** "Walk to coordinates (7,3)" is easy for an RL policy, but a VLM doesn't have the fine-grained spatial reasoning to output optimal button sequences.

3. **Cost at scale.** Calling an API model every frame during training (millions of steps) would cost thousands of dollars. Periodic calls (every 100-500 steps) keep costs negligible.

4. **The research angle is the hybrid.** "VLM strategy + RL motor" is novel and publishable. "VLM plays Pokemon" is a YouTube video.

## Connection to Existing Work

- **Hierarchical agent framework:** The VLM IS the LLM strategy layer from the 3-layer architecture. The RL motor is the skill execution layer.
- **World model direction:** The VLM provides a form of "world knowledge" — it knows Pokemon's rules, map layout (from training data), and optimal strategies without learning them from scratch.
- **Eureka reward search:** The VLM could generate reward functions dynamically: "You just entered Violet City — reward for reaching the gym."
- **Game bridge framework:** The VLM wrapper sits between the bridge and the training loop, same as the existing wrappers (temporal, anti-loop, telemetry).

## Proof of Concept Results (Gemma 4, May 2025)

Tested locally on M4 Max using Ollama with `gemma4` (e4B, 9.6GB) and `gemma3:4b` (3.3GB). The VLM receives a 2x-upscaled Pokemon Gold screenshot (320×288) plus RAM context and returns structured game analysis.

### Sample Output (Gemma 4)

**Input:** Screenshot of New Bark Town + "Position: (6,6), Party: 1 Pokemon Lv 5, Battle: none"

**Output (12s):**
```
1. LOCATION: Town street or residential area.
2. VISIBLE: Player character, one NPC, a two-story building, a paved path, and surrounding grass.
3. STATE: Overworld exploration (no menu, textbox, or battle active).
4. ACTION: Walk toward the NPC to initiate conversation or continue walking down the path.
```

### Model Comparison

| | Gemma 3 (4B) | Gemma 4 (e4B) |
|---|---|---|
| **Speed** | 1-4s | 11-16s |
| **NPC detection** | Missed NPC entirely | "one NPC (man)" |
| **Scene detail** | "building" | "two-story building, paved path, surrounding grass" |
| **Action quality** | "walk forward" | "Walk toward the NPC to initiate conversation" |
| **State detection** | Correct | Correct + specific ("overworld exploration") |
| **Location accuracy** | Wrong ("Route 9") | Correct type ("town/residential"), wrong name |

Neither model identifies New Bark Town by name — the GBC pixel art is too generic at 160×144. But Gemma 4 correctly reads the scene structure: buildings, NPCs, paths, grass. Combined with RAM data providing exact coordinates, the VLM provides the *semantic understanding* that RAM alone cannot.

### Key Findings

1. **Scene understanding works.** Gemma 4 distinguishes overworld from menus from battles, identifies NPCs, and recognizes building entrances — all from a tiny GBC screenshot.

2. **Actionable suggestions.** "Walk toward the NPC to initiate conversation" is a real game-relevant instruction that a goal-conditioned RL policy could execute.

3. **12 seconds is fine.** At 1 call per 200 agent steps (~80 seconds of game time at 24 ticks/step), VLM overhead is <15% of wall time. For periodic strategy, this is acceptable.

4. **RAM + VLM complement each other.** RAM knows exact coordinates, HP, level, badges — quantitative state. VLM knows "there's a building entrance to the right and an NPC blocking the path" — spatial/semantic state. Together they cover everything.

5. **Runs fully local.** No API calls, no cost, no latency beyond inference time. Gemma 4 at 9.6GB fits comfortably in 64GB unified memory alongside PyBoy and PPO training.

### Running the Test

```bash
# Install ollama and pull Gemma 4
brew install ollama
brew services start ollama
ollama pull gemma4

# Run the VLM test on Pokemon Gold screenshots
cd python
python test_vlm_pokemon.py --model gemma4

# Try with smaller/faster model
python test_vlm_pokemon.py --model gemma3:4b
```

## References

- **PokéAI** (arXiv:2506.23689) — LLM multi-agent system for Pokemon battles. 80.8% win rate with Planning/Execution/Critique agents. Not RL, but validates that LLMs understand Pokemon.
- **Voyager** (NVIDIA, 2023) — GPT-4 plays Minecraft using code generation. Discovers new skills autonomously via LLM + execution loop.
- **SPRING** (CMU, 2023) — LLM reads game manual, generates situation-aware prompts for game agents. Zero-shot play via knowledge extraction.
- **SayCan** (Google, 2022) — LLM says what to do, RL says what it can do. The "affordance grounding" pattern for combining language understanding with physical capabilities.

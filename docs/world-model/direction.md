# World Model Direction: Deep Research & Implementation Guide

## Table of Contents

1. [The Core Thesis](#the-core-thesis)
2. [What World Models Are (Mechanically)](#what-world-models-are-mechanically)
3. [Architecture Landscape](#architecture-landscape)
4. [Competitive Landscape](#competitive-landscape)
5. [What This Means for GhostLobby](#what-this-means-for-ghostlobby)
6. [The Spectrum of Agent Intelligence](#the-spectrum-of-agent-intelligence)
7. [Implementation Plan](#implementation-plan)
8. [Research Paper Roadmap](#research-paper-roadmap)
9. [Open-Source Tools & Libraries](#open-source-tools--libraries)
10. [References](#references)

---

## The Core Thesis

A world model is a neural network that learns the transition function of an environment -- given state S and action A, predict next state S'. Once you have that, you can "dream" rollouts without running the real simulator.

**GhostLobby's position is unusual.** Most researchers want learned world models because they either (a) don't have a simulator, (b) their simulator is too slow, or (c) they want to transfer from real-world video. None of those apply here -- GhostLobby IS a ground-truth simulator running at 19K+ FPS.

The play isn't to replace the simulator with a learned model. It's to use a learned model **alongside** a perfect simulator:
- The model for **fast approximate planning at inference time** (the agent carries its world model in its weights -- no sim needed at deployment)
- The simulator for **ground-truth training and benchmarking** (measure exactly how good the learned model is)
- Synthetic data generation for **targeted curriculum on rare events**

This combination is genuinely underexplored. Almost no one has a fast, deterministic, controlled simulator AND a learned world model side-by-side.

### The Gun Cleric Analogy

Think Equilibrium's Grammaton Clerics. They've studied "gun kata" -- a martial art based on statistical analysis of gunfight dynamics. They've mapped probability distributions of where shots come from in combat and designed movement to minimize their profile while maximizing offensive coverage.

That is literally model-based reinforcement learning:
- "Statistical analysis of gunfight dynamics" = learning a world model from combat data
- "Probability distributions of where shots come from" = the stochastic transition model's predictions
- "Designed movement to minimize profile while maximizing coverage" = policy optimization in imagination space

The clerics aren't fast. They're **pre-computed**. They've done all the rollouts in training and compressed optimal responses into muscle memory (a policy network). The fight is already over before it starts because they've already simulated it.

---

## What World Models Are (Mechanically)

When you play an FPS, you see an enemy disappear behind a wall. Your brain runs a simulation: "They were moving right at this speed, so in 2 seconds they'll be around that corner." You pre-aim based on that imagination.

In Dreamer's architecture (the most relevant for GhostLobby), this works as:

1. **Encoder**: Takes raw game state, compresses to latent vector `z` -- a compact representation of "what's happening now"
2. **Transition model (RSSM)**: Given latent state `z_t` and action `a_t`, predicts next latent state `z_t+1`. This IS the world model. It's a recurrent network maintaining a belief about the world.
3. **Reward predictor**: Given a latent state, predicts expected reward
4. **Decoder** (optional): Reconstructs full state from latent -- mostly for training the encoder

The key insight is the **latent space**. The model doesn't predict "enemy at position (4.2, 7.1, 3.0) with velocity (1.2, 0, -0.5)." It learns compressed representations that capture whatever structure matters for decisions. It might represent "enemy is pushing aggressively" or "this is a 2v1 disadvantage" as smooth regions in latent space, even though nobody told it those concepts exist.

### Three reasons a learned model matters even when you have a fast simulator

1. **The learned model is differentiable.** You can backpropagate through imagined trajectories -- directly optimize "what action sequence leads to the best outcome?" via gradient descent. The real sim is a black box. Dreamer trains its policy entirely in imagination using gradient ascent on imagined returns.

2. **The learned model runs in the agent's head.** At deployment, no sim needed. The agent carries its world model as part of its weights. For real-world applications (Unity/Unreal injection), the agent imagines outcomes internally because it can't pause the real game to run 1000 rollouts.

3. **Abstraction and generalization.** A learned model captures dynamics at a higher abstraction than raw physics. Instead of simulating every rigid body, it might learn "close-range AWP fights favor the AWP 80% of the time." This compressed knowledge enables faster, more strategic planning.

---

## Architecture Landscape

### The Dreamer Family (Hafner et al.)

The most relevant architecture lineage for GhostLobby.

#### Dreamer V1 (2020)
- Gaussian latent states, reparameterization gradients
- Three components: world model (RSSM), critic (value), actor (policy)
- First to show competitive model-based RL on continuous control

#### Dreamer V2 (2021, ICLR)
- **Categorical latents** replacing Gaussians -- more expressive, avoids posterior collapse
- Straight-through gradients for discrete latents and actions
- First model-based agent to achieve human-level on Atari 55-game benchmark

#### Dreamer V3 (2023, published Nature 2025)
- General algorithm outperforming specialized methods across **150+ diverse tasks** with a single configuration
- First algorithm to collect diamonds in Minecraft from scratch without human data or curricula
- Key innovations: symlog predictions, KL balancing, percentile return normalization, 1% unimix
- **RSSM**: 32 categorical distributions x 32 classes = 1024 discrete latent dimensions + 4096-unit GRU deterministic state

**RSSM Architecture Detail:**

| Component | Function | Architecture |
|-----------|----------|-------------|
| Sequence model | `h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})` | GRU, 4096 units (8 blocks x 512) |
| Encoder (Posterior) | `z_t ~ q(z_t \| h_t, x_t)` | MLP (state vectors) or CNN (pixels) |
| Dynamics (Prior) | `z_hat_t ~ p(z_t \| h_t)` | MLP predicting 32x32 categorical logits |
| Reward predictor | `r_t = f(h_t, z_t)` | MLP with twohot symlog output |
| Continue predictor | `c_t = f(h_t, z_t)` | Bernoulli output |

**For state-vector input (GhostLobby's case):** Replace CNN encoder/decoder with 2-layer MLPs. Everything else is identical. DreamerV3 explicitly supports this -- the paper evaluates on both pixel and proprioceptive environments.

**Default Hyperparameters (XL size):**

| Parameter | Value |
|-----------|-------|
| Model dim | 1024 |
| Deterministic state (h) | 4096 |
| Stochastic categories | 32 x 32 |
| MLP layers | 2 x 1024 |
| Actor/Critic layers | 4 x 1024 |
| Batch size | 16 sequences |
| Sequence length | 64 steps |
| Imagination horizon | 15 steps |
| Learning rate | 1e-4 |
| Discount (gamma) | 0.997 |
| Train ratio | 512 (model updates per env step) |
| Replay buffer | 1M transitions |

#### Dreamer V4 (2025)
- Replaces RSSM with a **diffusion transformer** (MAE + DiT architecture)
- **Shortcut Forcing**: 4 denoising steps match 64-step quality -- 16x speedup
- Real-time: 21 FPS on a single H100
- Scaling: 12M to 400M parameters, with direct performance gains from scale
- First agent to obtain diamonds purely from offline video-action data without environment interaction
- 1000%+ data efficiency gain over IMPALA/R2D2+ in 100M frames

### MuZero Family (Schrittwieser et al.)

Learns three connected networks:
- **Representation**: observation -> latent state
- **Dynamics**: (latent state, action) -> (next latent state, reward)
- **Prediction**: latent state -> (policy prior, value)

Plans via Monte Carlo Tree Search (MCTS) in learned latent space.

| Variant | Year | Key Innovation |
|---------|------|---------------|
| MuZero | 2020 | Superhuman Go/chess/shogi without knowing rules, SOTA Atari |
| Sampled MuZero | 2021 | Sampling-based MCTS for continuous action spaces |
| Stochastic MuZero | 2021 | Handles stochastic environments |
| Gumbel MuZero | 2022 | Reduces search complexity in vast action spaces |
| EfficientZero | 2021 | Superhuman Atari with only 2 hours of gameplay |
| EfficientZero V2 | 2024, ICML | Handles continuous + discrete. HNS 2.428 on Atari 100k. Surpasses DreamerV3 on 50/66 tasks |

**Latency analysis for real-time FPS:** Each MCTS simulation = ~0.1-0.3ms on GPU. At 60Hz you have 16.6ms per decision. 50 simulations barely fits; 200+ does not. MuZero is optimized for sample efficiency, not throughput. **Poor fit for GhostLobby's training pipeline** where environment steps are cheap (19K FPS). Better suited for deployment-time planning only.

### Other Key Architectures

**IRIS** (Micheli et al., 2023, ICLR notable top 5%)
- Transformer-based world model for Atari
- VQ-VAE tokenizer converts frames to discrete tokens; autoregressive transformer predicts sequences
- HNS 1.046 on Atari 100k

**STORM** (Zhang et al., 2023, NeurIPS)
- Stochastic Transformer + VAE world model
- 126.7% mean human performance on Atari 100k
- Trains in 4.3 hours on a single RTX 3090

**DIAMOND** (Alonso et al., 2024, NeurIPS Spotlight)
- Diffusion-based world model operating directly on pixel frames
- **HNS 1.46 on Atari 100k** -- best for world-model-trained agents
- Can simulate CS:GO on a single RTX 3090
- Diffusion maintains visual consistency where discrete-bottleneck models hallucinate

**TD-MPC2** (Hansen et al., 2024, ICLR)
- Implicit decoder-free world model for continuous control
- Plans via trajectory optimization (MPC) in latent space
- Single 317M parameter agent handles 80 tasks across multiple domains

**GameNGen** (Google, 2024, ICLR 2025)
- First game engine powered entirely by neural inference
- Playable DOOM at 20+ FPS on a single TPU
- PSNR 29.4 (comparable to lossy JPEG); human raters barely distinguish real vs. simulated

**Genie Series** (DeepMind):
- Genie 1 (2024): 11B params, trained on 200K+ hours of 2D platformer video, learns controls from unlabeled video
- Genie 2 (Dec 2024): 3D action-controllable environments from single prompt images
- Genie 3 (Aug 2025): Real-time (24 FPS, 720p), general-purpose, text-prompted. Released publicly as "Project Genie" (Jan 2026)

**Oasis** (Decart + Etched, Oct 2024)
- Interactive Minecraft-like world model
- ViT spatial autoencoder + DiT latent diffusion backbone
- 20 FPS, 500M open-weight model released
- 1M+ users in 3 days at launch

### Benchmarks: State of the Art

**Atari 100k (mean human-normalized score):**

| Method | Type | HNS | Year |
|--------|------|-----|------|
| EfficientZero V2 | Tree search + WM | 2.428 | 2024 |
| DIAMOND | Diffusion WM | 1.46 | 2024 |
| STORM | Stochastic transformer WM | 1.266 | 2023 |
| DreamerV3 | RSSM WM | 1.097 | 2023 |
| IRIS | Transformer + VQ-VAE WM | 1.046 | 2023 |
| Human | - | 1.0 | - |

**Where model-based beats model-free:**
- Sample efficiency: consistently **10-100x better** (HVAC research: 10x; DreamerV4 vs VPT: 100x less data)
- Planning in latent space: ~5000% more data efficient on average vs model-free
- Low-data regimes: EfficientZero is superhuman with 2 hours of gameplay

**Where model-free still wins:**
- Asymptotic performance in some domains (model bias limits ceiling)
- Simplicity and stability of training
- Settings where dynamics are too complex/stochastic to model accurately

### Multi-Agent World Models

This is the frontier most relevant to GhostLobby's adversarial FPS setting.

**GAWM** (Shi et al., Jan 2025): Global-Aware World Model for multi-agent RL. Extra Transformer fuses local observations across agents. Outperforms model-free/model-based baselines on StarCraft SMAC.

**CoDreamer** (Toledo, 2024): Extends DreamerV3 for multi-agent environments using Graph Neural Networks for agent-to-agent communication. Addresses partial observability and cooperation.

**World-Model-Assisted CEM-MARL** (2025): Infers opponents' mental states, predicts/evaluates future trajectories. Accelerates learning **40x** vs model-free MARL in multi-UAV scenarios.

**Theory of Mind in RL:**
- CMU Dissertation (2025): Uses opponent ToM modeling error to induce deceptive behavior. Extends to higher-order ToM (beliefs about beliefs)
- LG-TOM: Online RL with Theory of Mind + language grounding for partially observable, decentralized multi-agent settings
- Emergent evidence of collaborative behaviors and high-order ToM among LLM-based agents

**Key challenge in adversarial settings:** The opponent's policy is non-stationary and part of the dynamics model itself. The world model must account for changing opponent behavior, not just physics.

### Video Generation as World Models

**The "world simulator" framing:** OpenAI framed Sora as a "world simulator." It can model some physics (paint strokes persist, bite marks on food) but fails at basic interactions (glass shattering).

**Runway GWM-1** (Dec 2025): First explicit world model product from Runway. Autoregressive, 24 FPS, 720p. Controllable via camera pose and robot commands.

**V-JEPA 2** (Meta, Jun 2025): LeCun's Joint Embedding Predictive Architecture. 1.2B params, trained on 1M+ hours of video. Predicts in abstract representation space, not pixels. Zero-shot robot control with 65-80% success using only 62 hours of robot data. 30x faster than NVIDIA Cosmos.

**Pixel vs state-level world models for GhostLobby:** State-level is the clear choice. We have structured observations (feature vectors), not pixels. Latent/state-level models are 5000% more data efficient, lower compute, faster rollouts. Pixel models are for when you don't have access to the game's internal state.

---

## Competitive Landscape

### The World Model Space Is Bifurcated

There are two distinct categories:
1. **Generative visual world models** (Genie, Oasis, Runway GWM) -- dream up pixels. Targeting entertainment, film, content creation.
2. **Decision-making world models** (MuZero, DreamerV3, EfficientZero) -- learn dynamics for planning/RL. Targeting robotics, games, autonomous systems.

GhostLobby sits firmly in category (b) -- using a real physics engine as ground truth, not hallucinated pixels.

### Direct Competitors: Game QA / Testing AI

| Company | Raised | Approach | Can Do FPS? | GhostLobby Advantage |
|---------|--------|----------|-------------|---------------------|
| **Filuta AI** | $6.7M (seed, Jun 2025) | PDDL-based automated planning. Tests specify goals, planner generates paths. Roots in space/deep-sea missions. | No -- limited to games that can be formalized in PDDL. No RL, no continuous actions. | Continuous-action RL agents, FPS-native, world model for planning |
| **modl.ai** | $25.8M ($15M Series A, Q1 2025) | AI-driven playtesting and QA bots. Learned agent behavior for exploration and bug detection. | Limited | Raw throughput (19K FPS), config-driven scenarios, world model direction |
| **Spirit AI** | Acquired by Twitch (Aug 2022) | NPC behavior (Character Engine) + community safety (Ally). Not QA-focused. | N/A | Different market segment |
| **ManaMind** | $1.1M (seed) | AI-powered game testing agents. Early stage. | Unknown | Maturity, architecture, throughput |

**Market size:** Game QA testing services estimated at **$5B in 2025**, projected **$15B by 2033**. Outsourcing segment alone $2.5B growing at 15% CAGR.

**Major studio in-house efforts:**
- Square Enix: Plans to automate 70% of QA by 2027
- Ubisoft: AI bots autonomously explore open worlds using behavior trees + ML
- CD Projekt Red: AI-driven regression testing for Cyberpunk
- EA, Microsoft, Tencent: All building internal AI testing pipelines

### "AI Game Engine" Companies

| Company | Raised | Valuation | What They Do |
|---------|--------|-----------|-------------|
| **AMI Labs** (LeCun) | $1.03B seed (Mar 2026) | $3.5B | JEPA-based world models. Largest European seed ever. |
| **World Labs** (Fei-Fei Li) | $1.23B total | ~$5B | Marble: generates persistent 3D environments from text/images |
| **Decart** | $153M ($100M Series B) | $3.1B | Oasis: real-time AI Minecraft. 1M+ users in 3 days |
| **Runway** | $315M (Feb 2026) | Undisclosed | GWM-1: general world model, 24 FPS, controllable |
| **Odyssey** | $27M+ | Undisclosed | Photorealistic 3D world streaming, 40ms per frame |
| **Altera** | $44M ($33M Series A) | Undisclosed | AI agents for multiple verticals |
| **Voyage/Latitude** | Undisclosed (Google AI Futures Fund) | Undisclosed | AI-native RPG platform, autonomous NPC agents |
| **Lucid** (YC) | Undisclosed | Undisclosed | Fastest action-conditioned diffusion, 20+ FPS on 4090 |
| **Origin Lab** | $8M seed (May 2026) | Undisclosed | Helps game companies license data to world model builders |

**Macro funding picture:** Over **$2.5B** flowed into world model companies in the last 12 months. PitchBook projects the world model market in gaming could grow from $1.2B (2022-2025) to **$276B by 2030**. The funding window is open.

### Key Takeaways for GhostLobby's Positioning

1. **Almost no one is building headless, config-driven simulation engines optimized for RL training throughput.** The big players chase photorealism and pixel generation. GhostLobby's headless ECS approach is complementary, not competing.

2. **Game QA is underserved.** Filuta can't do FPS (PDDL-only). modl.ai is broader but smaller scale. Neither has world model capabilities.

3. **Origin Lab's $8M raise to broker game data to world model labs validates that structured game simulation data has commercial value** -- which is exactly what GhostLobby produces.

4. **The combination of ground-truth sim + learned world model + adversarial FPS focus is a unique niche** no funded competitor occupies.

5. **Research labs (DeepMind, BAIR, FAIR) are pushing world models forward but not building products.** The translation from papers to usable game AI products is wide open.

---

## What This Means for GhostLobby

### Three Areas Where World Models Get Interesting

#### 1. Model-Based RL for Planning (Gun Cleric Mode)

Instead of pure reactive policy (see state -> output action), the agent "imagines" 5-10 steps ahead before choosing. For FPS mechanics:

| Mechanic | Reactive (current) | With World Model |
|----------|-------------------|-----------------|
| Aim duels | React to enemy appearing | Pre-aim at predicted position |
| Peek timing | Peek when "feels right" | Simulate outcomes, peek when favorable |
| Trading | React after teammate dies | Predict the trade, pre-position |
| Site executes | Follow trained pattern | Simulate enemy response, adapt |
| Information denial | N/A | Model what opponent knows, deny info |

The agent doesn't react to the enemy appearing -- it predicts where they'll be and pre-positions. Every movement accounts for where opponents **will be**, not where they **are**.

#### 2. Synthetic Data for Rare Events

Normal training generates data at the distribution current agents produce. But the interesting learning happens in rare situations: clutch 1v3s, spray transfers, creative positioning.

With a trained dynamics model:
1. **State mining**: Scan replay buffer for interesting starting states (low HP, outnumbered, caught in the open)
2. **Branching**: From each state, imagine 1000 different action sequences
3. **Filtering**: Keep trajectories where something interesting happens (comeback, creative positioning)
4. **Training**: Mix imagined trajectories into PPO buffer

The model doesn't need to be perfect. Sutton's Dyna architecture (1991) showed that even a fairly inaccurate model speeds up learning because imperfect plans are still better than no plans.

#### 3. The Research Angle (Ground-Truth Benchmarking)

GhostLobby provides a **ground-truth oracle** to benchmark against. We can:
- Measure exactly how much a learned world model's predictions diverge from reality
- Quantify how prediction error affects agent performance
- Decompose the value of planning vs the cost of model error

Most world model papers can't do this cleanly (Atari pixels, real robotics). We have a deterministic, fast, controlled environment. **That's a paper.**

### Connection to the Hierarchical Agent Framework

The existing 3-layer architecture (LLM strategy + skills + RL motor) maps perfectly onto world models:

- **LLM strategy layer**: Uses the world model to simulate entire round-level plans. "What if we send two A and one lurk B?" Run 100 imagined rollouts, evaluate outcomes, pick best strategy. Doesn't need to run at 60Hz.
- **Skills layer**: Each skill (peek, spray, rotate) has its own planning horizon. Short rollouts (0.5s) for aim, longer (5-10s) for rotations.
- **RL motor**: Executes in real time. The world model is the bridge between strategic thinking and tactical execution.

---

## The Spectrum of Agent Intelligence

| Level | Name | Mechanism | FPS Capability | GhostLobby Status |
|-------|------|-----------|---------------|-------------------|
| 0 | Reactive | See state -> output action | Aim, basic movement | Current PPO agents |
| 1 | Recurrent | Hidden state remembers past | Track unseen enemies, patterns | GRU branch (in progress) |
| 2 | World Model | Explicit dynamics model, imagination rollouts | Pre-aim, trade prediction, peek timing | **Target** |
| 3 | Theory of Mind | Models opponent's decision-making | Deception, baiting, information denial | Future research |
| 4 | Hierarchical + WM | LLM strategy + WM planning + RL execution | Emergent macro-tactics (splits, fakes, rotations) | Future research |

**Level 2 is the target.** The jump from reactive to predictive is where "good aimbot" becomes "plays like it understands the game."

---

## Implementation Plan

### Phase 1: Dyna-MLP Augmentation (1-2 weeks)

The simplest starting point. Train a dynamics model alongside existing PPO, generate synthetic transitions, mix into training.

**Architecture: MLP Ensemble**

```python
class DynamicsEnsemble(nn.Module):
    """Ensemble of N probabilistic MLPs predicting (next_state_delta, reward)."""
    def __init__(self, obs_dim, act_dim, hidden=256, n_models=5):
        super().__init__()
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim + act_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, (obs_dim + 1) * 2),  # mean + log_var for delta + reward
            )
            for _ in range(n_models)
        ])
```

**Training procedure:**
1. Collect real transitions with PPO, store (s, a, r, s', done) in replay buffer
2. Train dynamics model every ~1000 env steps: sample mini-batches, each ensemble member on bootstrapped 80% subset, Gaussian NLL loss on state deltas
3. Generate synthetic transitions: sample real states as starting points, roll out 1-3 steps using model + current policy, randomly select ensemble member per step
4. Mix into PPO: augment on-policy buffer with 10-25% synthetic transitions

**Integration via SB3 callback:**

```python
class DynaCallback(BaseCallback):
    """Trains dynamics model and injects synthetic data."""
    def _on_step(self):
        self.replay.add(obs, actions, rewards, new_obs, dones)
        if self.num_timesteps % self.model_train_freq == 0:
            self._train_model()
        return True
    
    def _on_rollout_end(self):
        """Inject synthetic transitions into PPO's buffer."""
        starts = self.replay.sample(n_synth)
        synth = self._generate_rollouts(starts)
        self._inject_into_buffer(self.model.rollout_buffer, synth)
```

**Warning:** PPO is on-policy. Injecting off-policy synthetic data violates its assumptions. Keep synthetic ratio low (10-25%), use short rollouts (1-3 steps). A 2024 paper ("Stealing That Free Lunch") showed naive Dyna-PPO can hurt on some benchmarks. Test carefully.

**Expected outcome:** 2-5x sample efficiency improvement. MBPO reports 10-20x on continuous control; FPS dynamics are more complex, so expect less initially.

**Computational overhead on M4 Max:** <5% of training time.

### Phase 2: RSSM Dynamics Model (2-4 weeks)

Replace MLP ensemble with Dreamer-style RSSM. This mirrors the existing `_GruAsLstm` architecture -- the RSSM is essentially the GRU policy's backbone repurposed for environment prediction.

**What changes:**
- GRU recurrent state maintains belief about unobserved game state (enemy behind wall)
- Categorical stochastic state handles multi-modal futures (opponent might go left OR right)
- Extend rollouts from 1-3 steps to 5-15 steps
- Add ensemble disagreement-based rollout truncation (discard when models disagree too much)

**Key training tricks from DreamerV3:**
- **Symlog**: `sign(x) * log(1 + |x|)` -- compresses magnitudes so same network works across reward scales
- **Twohot predictions**: Softmax over 255 symlog-scaled bins instead of single-point regression
- **KL balancing**: 80% gradient to prior (dynamics predictor), 20% to posterior
- **Free nats**: KL clipped below 1.0 nat to prevent over-compression
- **1% Unimix**: All categorical distributions mix in 1% uniform probability to prevent mode collapse

**Important:** These tricks do NOT transfer to model-free PPO. A NeurIPS 2023 paper tested this explicitly -- symlog, twohot, percentile normalization "generally do not enhance the performance of PPO." They're synergistic with the world model training loop specifically.

### Phase 3: Full DreamerV3 (4-8 weeks)

Replace PPO entirely with DreamerV3's actor-critic-in-imagination.

**Training loop:**
```
for each environment step:
    1. Act with actor network
    2. Store transition in replay buffer

every train_ratio steps (512 model updates per env step):
    1. Sample 16 sequences (length 64) from replay
    2. Train world model (RSSM + reward + continue heads)
    3. Imagine 15-step trajectories from encoded replay states
    4. Train actor and critic on imagined trajectories only
```

**Actor-critic in imagination:**
1. Sample 16 real sequences from replay, encode through RSSM to get starting latent states
2. From each start, unroll 15 imagined steps using dynamics predictor + actor (no real observations)
3. Compute lambda-returns over imagined trajectories using critic's value estimates
4. Actor loss: REINFORCE with baseline + entropy regularizer
5. Critic loss: twohot symlog regression on lambda-returns

**Throughput implications:** DreamerV3's train ratio of 512 means training is GPU-bound, not sim-bound. Your 19K FPS throughput fills the replay buffer instantly. The bottleneck shifts to GPU training. This is actually a good fit -- fast env + compute-heavy training.

**Expected outcome:** 5-20x sample efficiency improvement over model-free PPO.

**Implementation path:** Fork `NM512/dreamerv3-torch` or use SheepRL. Adapt for `GhostLobbyEnv` observation/action space.

### Phase 4: Synthetic Curriculum + Theory of Mind (8+ weeks)

**Synthetic curriculum:**
- Mine replay buffer for interesting starting states
- Condition the dynamics model on specific scenarios
- Dream targeted rollouts (low HP, outnumbered, post-plant)
- Mix back into training for accelerated learning on rare events

**Opponent modeling:**
- Extend world model to predict opponent actions separately
- Enable the agent to simulate "what does the enemy think I'm doing?"
- Hypothesis: agents with opponent models develop emergent deception (faking, baiting, information denial)

**Skip MuZero** unless moving to a setting where environment steps are expensive (e.g., real-world deployment on physical hardware).

---

### Architecture Choices for FPS World Models

**State-space models vs transformers vs MLPs (for state-vector observations):**

| Architecture | Best For | Compute | Recommendation |
|-------------|----------|---------|---------------|
| MLP ensemble (5x 3-layer 256) | 1-3 step rollouts, Dyna augmentation | Lowest | Phase 1 starting point |
| GRU/RSSM (DreamerV3-style) | 5-15 step rollouts, partial observability | Medium | Phase 2-3 target |
| Transformer (MATWM-style) | Multi-agent modeling, 5+ agents | 3-5x more than GRU | Overkill unless 5+ agents |

**Handling multi-agent dynamics:**

Start centralized (concatenate all observable agent states into one vector, predict full next-state). Your observation already concatenates entity features. Upgrade to factored (per-agent dynamics with cross-attention) only if needed for scaling.

**Partial observability (fog of war, occlusion):**

FPS is inherently partially observable. The RSSM's recurrent state naturally maintains belief about unobserved state. The GRU remembers "enemy was last seen heading north 2 seconds ago." Variable-length entity lists handled via attention or fixed-size slots with presence masks.

**Deterministic vs stochastic models:**

For FPS combat, **stochastic is important**:
- Opponent actions are unpredictable from ego perspective
- Damage has variance (spread, headshot probability)
- Multiple plausible futures exist (opponent might peek left or right)

The ensemble provides implicit stochasticity (different members predict different futures). RSSM's explicit categorical state is better for multi-modal prediction.

---

### Evaluation and Debugging World Models

**Prediction accuracy metrics:**

```python
def eval_1step(model, test_buffer, n_samples=1000):
    states, actions, rewards, next_states, _ = test_buffer.sample(n_samples)
    pred_delta, pred_reward = model.predict(states, actions)
    pred_next = states + pred_delta
    state_mse = F.mse_loss(pred_next, next_states)
    reward_mse = F.mse_loss(pred_reward, rewards)
    per_feat_mse = (pred_next - next_states).pow(2).mean(0)
    return state_mse, reward_mse, per_feat_mse

def eval_nstep(model, policy, real_trajectories, horizon=10):
    """Compare model rollouts vs real trajectories (self-feeding)."""
    for traj in real_trajectories:
        s = traj.states[0]
        for t in range(horizon):
            a = policy.predict(s)
            s = s + model.predict(s, a)[0]  # self-feeding divergence
            mse = F.mse_loss(s, traj.states[t+1])
```

**"Good enough" thresholds:**
- 1-step state prediction R-squared > 0.95 per feature
- 3-step rollout error < 2x the 1-step error (sublinear compounding)
- Ensemble disagreement on in-distribution data < 10% of feature variance
- Policy trained on 25% synthetic data performs within 5% of pure real-data PPO

**Common failure modes:**

| Failure | Symptom | Fix |
|---------|---------|-----|
| Model exploitation | Reward climbs in imagination, falls in real env | Shorter rollouts, ensemble penalty |
| Compounding drift | N-step error grows exponentially | Reduce horizon, add regularization |
| Reward hacking | Agent finds unreachable high-reward states | Truncate on ensemble disagreement |
| Catastrophic forgetting | Model degrades as policy changes | Bigger replay buffer, keep old data |
| Feature scaling | Some features predicted well, others poorly | Normalize features, use symlog |

---

## Research Paper Roadmap

### Paper 1: "Do FPS Agents Need Imagination?"

Use GhostLobby to compare model-free PPO vs Dreamer-style model-based RL on specific FPS mechanics:

| Mechanic | Hypothesis |
|----------|-----------|
| Aim duels | World model adds little (reactive is sufficient) |
| Peek timing | Prediction helps significantly |
| Trade setups | Planning is critical |
| Site executes | Strategy-level planning required |

**Novel contribution:** Ground-truth sim enables clean decomposition of prediction error vs planning benefit. Nobody else can do this in a controlled FPS environment.

### Paper 2: "Theory of Mind in Tactical Combat"

Extend world model to include opponent modeling. Train agents with and without explicit opponent models.

**Hypothesis:** Agents with opponent models develop emergent deception (faking, baiting, information denial) because they can simulate "what does the enemy think I'm doing?"

**Connection to:** Game theory literature on information asymmetry, CMU ToM dissertation (2025), CEM-MARL opponent modeling.

### Paper 3: "Hierarchical Planning with Learned World Models"

LLM strategy + world model + RL motor stack. Strategy layer uses world model to simulate round-level outcomes.

**Hypothesis:** Hierarchy produces emergent macro-tactics (splits, fakes, rotations) that flat model-based RL can't discover because its planning horizon is too short.

---

## Open-Source Tools & Libraries

### DreamerV3 Implementations (PyTorch)

| Repo | Quality | Notes |
|------|---------|-------|
| [NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch) | Best standalone | Clean PyTorch port, good configs |
| [SheepRL](https://github.com/Eclectic-Sheep/sheeprl) | Best framework | Distributed via Lightning Fabric, Hydra configs, many algorithms |
| [burchim/DreamerV3-PyTorch](https://github.com/burchim/DreamerV3-PyTorch) | Good | Complete port |
| [DrunkJin/dreamer-from-scratch](https://github.com/DrunkJin/dreamer-from-scratch) | Educational | Minimal, for learning |

### MuZero/MCTS

| Repo | Notes |
|------|-------|
| [opendilab/LightZero](https://github.com/opendilab/LightZero) | NeurIPS 2023 Spotlight. MuZero, EfficientZero, Sampled/Gumbel/UniZero |
| [EfficientZeroV2](https://github.com/shengjiewang-jason/efficientzerov2) | ICML 2024 Spotlight. Continuous + discrete |
| [muzero-general](https://github.com/werner-duvaud/muzero-general) | Community, easily adaptable |

### Model-Based RL Frameworks

| Repo | Notes |
|------|-------|
| [facebookresearch/mbrl-lib](https://github.com/facebookresearch/mbrl-lib) | Ensemble dynamics, MBPO, PETS. Modular. Works with Gymnasium. |
| [jannerm/mbpo](https://github.com/jannerm/mbpo) | Official MBPO reference |

### World Model Resources

| Resource | Notes |
|----------|-------|
| [DIAMOND](https://github.com/eloialonso/diamond) | Diffusion world model. Can simulate CS:GO. NeurIPS 2024 |
| [Skywork Matrix-Game](https://github.com/SkyworkAI/Matrix-Game) | Open-source interactive world model, MIT license |
| [Awesome-World-Models](https://github.com/leofan90/Awesome-World-Models) | Curated paper/repo list |
| [awesome-model-based-RL](https://github.com/opendilab/awesome-model-based-RL) | Comprehensive MBRL index |

### Integration Path for GhostLobby

All libraries work with Gymnasium environments. For `BaseGhostLobbyGym` subclasses:

- **MBRL-lib**: Use `mbrl.models.GaussianMLP` as dynamics model, `mbrl.util.common.train` for training loop. Minimal integration effort.
- **SheepRL/DreamerV3**: Register gym env via Hydra config pointing to `CsLiteGym`. Handles distributed training.
- **LightZero**: Wrap env in `LightZeroEnv` wrapper with their observation format.

---

## References

### Core Papers

- Ha & Schmidhuber, "World Models" (2018) -- [arXiv 1803.10122](https://arxiv.org/abs/1803.10122)
- Hafner et al., "PlaNet / RSSM" (2019) -- [arXiv 1811.04551](https://arxiv.org/abs/1811.04551)
- Hafner et al., "Dreamer V2" (2021) -- [ICLR 2021](https://openreview.net/pdf?id=0oabwyZbOu)
- Hafner et al., "Dreamer V3" (2023) -- [arXiv 2301.04104](https://arxiv.org/abs/2301.04104), Nature 2025
- Hafner et al., "Dreamer V4" (2025) -- [arXiv 2509.24527](https://arxiv.org/abs/2509.24527)
- Schrittwieser et al., "MuZero" (2020) -- [arXiv 1911.08265](https://arxiv.org/abs/1911.08265), Nature 2020
- Ye et al., "EfficientZero" (2021) -- NeurIPS 2021
- Wang et al., "EfficientZero V2" (2024) -- [arXiv 2403.00564](https://arxiv.org/abs/2403.00564), ICML 2024
- Micheli et al., "IRIS" (2023) -- [arXiv 2209.00588](https://arxiv.org/abs/2209.00588), ICLR 2023
- Alonso et al., "DIAMOND" (2024) -- [arXiv 2405.12399](https://arxiv.org/abs/2405.12399), NeurIPS 2024
- Zhang et al., "STORM" (2023) -- [arXiv 2310.09615](https://arxiv.org/abs/2310.09615), NeurIPS 2023
- Hansen et al., "TD-MPC2" (2024) -- [arXiv 2310.16828](https://arxiv.org/abs/2310.16828), ICLR 2024

### Game-Specific World Models

- Google, "GameNGen" (2024) -- [arXiv 2408.14837](https://arxiv.org/abs/2408.14837), ICLR 2025
- DeepMind, "Genie 1" (2024) -- [arXiv 2402.15391](https://arxiv.org/abs/2402.15391)
- DeepMind, "Genie 2" (2024) -- [deepmind.google/blog/genie-2](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/)
- DeepMind, "Genie 3" (2025) -- [deepmind.google/blog/genie-3](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)
- Decart, "Oasis" (2024) -- [oasis-model.github.io](https://oasis-model.github.io/)
- Microsoft, "Muse/WHAM" (2024) -- [microsoft.com/research/blog/introducing-muse](https://www.microsoft.com/en-us/research/blog/introducing-muse-our-first-generative-ai-model-designed-for-gameplay-ideation/)

### Multi-Agent World Models

- Shi et al., "GAWM" (2025) -- [arXiv 2501.10116](https://arxiv.org/abs/2501.10116)
- Toledo, "CoDreamer" (2024) -- [arXiv 2406.13600](https://arxiv.org/abs/2406.13600)
- Meta, "V-JEPA 2" (2025) -- [arXiv 2506.09985](https://arxiv.org/abs/2506.09985)
- NVIDIA, "Cosmos" (2025) -- [arXiv 2501.03575](https://arxiv.org/abs/2501.03575)

### Surveys

- "Understanding World or Predicting Future?" -- ACM Computing Surveys (2025), [arXiv 2411.14499](https://arxiv.org/abs/2411.14499)
- "A Comprehensive Survey on World Models for Embodied AI" (2025) -- [arXiv 2510.16732](https://arxiv.org/abs/2510.16732)

### Key Blog Posts and Guides

- Dyna architecture: Sutton, "Dyna, an Integrated Architecture for Learning, Planning, and Reacting" (1991)
- MBPO: [BAIR Blog: Model-Based RL Theory and Practice](https://bair.berkeley.edu/blog/2019/12/12/mbpo/)
- SheepRL DreamerV3 guide: [eclecticsheep.ai/2023/08/10/dreamer_v3](https://eclecticsheep.ai/2023/08/10/dreamer_v3.html)
- NeurIPS 2023: "DreamerV3 Tricks Do NOT Help PPO" -- [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2023/file/04f61ec02d1b3a025a59d978269ce437-Paper-Conference.pdf)

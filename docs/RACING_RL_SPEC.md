# Project Sophy-One: Superhuman Sim Racing AI

**Goal:** Solo-replicate GT Sophy's results — train an RL agent to beat human lap records in a commercial racing simulator, using consumer hardware (3090 + M4 Max).

**Headline:** "One developer. One GPU. Superhuman lap times."

---

## 1. Why This Matters

Sony needed 1,000 PS4 consoles, a team of PhD researchers, and a privileged internal API from Polyphony Digital to train GT Sophy. The agent beat world champions and was published in Nature (Feb 2022).

This project proves the same result is achievable by a single engineer with:
- Off-the-shelf racing sim telemetry (shared memory, not privileged APIs)
- Consumer hardware (RTX 3090)
- A smarter training pipeline (BC warm-start + distributional RL + curriculum)

If this works, it's an undeniable credential — and a foundation for consulting, product work, or a startup in simulation-based RL.

---

## 2. Target Sim Selection

### Primary: Assetto Corsa (AC)

| Factor | Detail |
|--------|--------|
| Physics fidelity | Laser-scanned tracks, tire model validated against real telemetry |
| Telemetry API | Shared memory at 100Hz: full inputs (gas, brake, steerAngle), dynamics (speed, velocity[3], accG[3]), wheels (slip, load, temp, camber, suspension), environment (airTemp, roadTemp) |
| Control interface | vgamepad (virtual Xbox 360) or vJoy |
| Modding | Extensive — Content Manager, custom apps, Python plugins |
| Community | Massive. Active modding, racing leagues, content creation |
| Existing RL work | AC Gym (116 stars, one maintainer, basic SAC) — weak baseline to surpass |
| Platform | Windows (shared memory is Win32 mmap) |
| Cost | ~$10 on Steam, most DLC available cheap |

### Secondary consideration: rFactor 2

| Factor | Detail |
|--------|--------|
| Unique advantage | **Only sim that exposes full inputs (throttle/brake/steer) for ALL cars on track** via shared memory |
| BC collection | Can spectate races and collect expert (state, action) pairs from every car simultaneously — up to 128 cars |
| Telemetry | rF2SharedMemoryMapPlugin: position, velocity, accel, RPM, gear, filtered+unfiltered inputs per car |
| Downside | Smaller community, less visual appeal for demos, less name recognition |

### Recommendation

Start with AC for the demo (bigger community, more recognizable, existing AC Gym to benchmark against). Use rFactor 2 for large-scale BC data collection if needed (spectate multi-car races, harvest expert telemetry from all cars simultaneously).

### iRacing note

60Hz telemetry, good input data when driving, but **throttle/brake are stripped when spectating** (only steering exposed for other cars). Not viable for passive BC collection from other drivers. Could be a future target for the trained agent to race in (biggest competitive sim racing platform).

---

## 3. Technical Architecture

### 3.1 Overview

```
Phase 1: BC Data Collection
  Spectate/record races --> dump (telemetry, inputs) to disk at 100Hz
  Downsample to 20Hz for training
  Target: 50-100 laps of competent driving per track

Phase 2: BC Pretraining
  Supervised learning: MLP maps telemetry --> steering/throttle/brake
  Offline on 3090, takes hours
  Output: competent baseline policy + reference weights for KL anchor

Phase 3: RL Fine-tuning (TQC/QR-SAC)
  Agent drives AC in real-time via vgamepad
  TQC (sb3-contrib) with KL anchor to BC policy
  Replay buffer seeded with BC demonstrations
  Target: surpass BC ceiling, approach/beat human records

Phase 4: Curriculum & Multi-track
  Start on wide/slow tracks, advance to narrow/fast
  Transfer policy across tracks via domain randomization
  Optional: multi-car racing with opponent modeling
```

### 3.2 Telemetry Interface (AC Shared Memory)

Three memory-mapped pages, read via Python `mmap` + `ctypes`:

**SPageFilePhysics (~100Hz) — the main one:**
- Inputs: `gas`, `brake`, `steerAngle`, `clutch`
- Dynamics: `speedKmh`, `velocity[3]`, `accG[3]`, `localVelocity[3]`, `localAngularVel[3]`
- Orientation: `heading`, `pitch`, `roll`
- Wheels (x4): `wheelSlip`, `wheelLoad`, `wheelsPressure`, `wheelAngularSpeed`, `tyreCoreTemperature`, `suspensionTravel`, `camberRAD`, `brakeTemp`
- Systems: `gear`, `rpms`, `fuel`, `drs`, `tc`, `abs`
- Damage: `carDamage[5]`, `numberOfTyresOut`

**SPageFileGraphics (~configurable, default 0.1Hz — reconfigure for RL):**
- `normalizedCarPosition` (0-1 track progress — critical for reward)
- `completedLaps`, `currentTime`, `lastTime`, `bestTime`
- `carCoordinates[3]`
- `currentSectorIndex`, `lastSectorTime`
- `status` (AC_OFF/AC_REPLAY/AC_LIVE/AC_PAUSE)

**SPageFileStatic (once per session):**
- `trackSPlineLength` (total track length in meters)
- `carModel`, `track`, `trackConfiguration`
- `maxTorque`, `maxPower`, `maxRpm`
- `tyreRadius[4]`, `suspensionMaxTravel[4]`

**Gotcha:** Check `packetId` to detect stale data — only process when it changes. Graphics page updates at 10s by default; must reconfigure or read from Physics page primarily.

**Python access:** Use `sim_info.py` pattern — `mmap.mmap(-1, size, "acpmf_physics")` mapped to ctypes structures.

### 3.3 Control Output

**vgamepad** (Python library, emulates Xbox 360 via ViGEmBus driver):
```python
import vgamepad as vg
gamepad = vg.VX360Gamepad()
gamepad.left_joystick_float(x_value=steering, y_value=0.0)  # steering
gamepad.right_trigger_float(value=throttle)                   # gas
gamepad.left_trigger_float(value=brake)                       # brake
gamepad.update()
```

AC sees a real controller. No injection, no hacking, no anti-cheat concerns.

### 3.4 Timing

- Read telemetry at 100Hz (physics page rate)
- Policy decision at **20Hz** (every 5th physics tick)
- vgamepad output immediately after policy inference
- Total inference budget per step: **50ms** (20Hz). MLP inference is <1ms. Plenty of headroom.

**Why 20Hz:** GT Sophy ran at 10Hz. ETH F1TENTH at 20-40Hz. At 200kph, 20Hz = one decision every 2.7m of track. Human drivers react at ~10-15Hz. No useful information is lost.

---

## 4. Observation Space

### 4.1 Design Principles

- **Track-relative (Frenet frame)** coordinates, not world coordinates
- **Ego-centric** — everything relative to the car's current position/heading
- **Normalize to [-1, 1]** using known physical bounds (not learned running stats)
- **Include previous actions** (last 2-3 steps) for the network to learn smooth control
- **No frame stacking if velocity/accel are explicit** — stacking is only needed when you can't observe dynamics directly. With telemetry, you can.

### 4.2 Feature Vector

Based on GT Sophy, ETH F1TENTH, and AC Gym research:

**Car state (14 features):**
| Feature | Source | Normalization |
|---------|--------|---------------|
| Speed (m/s) | `speedKmh / 3.6` | `/ max_speed` |
| Local velocity X (lateral) | `localVelocity[0]` | `/ max_speed` |
| Local velocity Z (longitudinal) | `localVelocity[2]` | `/ max_speed` |
| Acceleration X (lateral G) | `accG[0]` | `/ max_g` (~4.0) |
| Acceleration Z (longitudinal G) | `accG[2]` | `/ max_g` |
| Yaw rate | `localAngularVel[1]` | `/ max_yaw_rate` |
| Steering angle | `steerAngle` | `/ max_steer` |
| Throttle | `gas` | already [0,1] |
| Brake | `brake` | already [0,1] |
| Gear | `gear` | `/ max_gear` |
| RPM | `rpms` | `/ max_rpm` |
| Slip angle front avg | `mean(wheelSlip[0:2])` | `/ max_slip` |
| Slip angle rear avg | `mean(wheelSlip[2:4])` | `/ max_slip` |
| Tyre grip estimate | derived from slip + load | [0,1] |

**Track-relative state (6 features):**
| Feature | Source | Normalization |
|---------|--------|---------------|
| Track progress (s-coordinate) | `normalizedCarPosition` | already [0,1] |
| Cross-track error (lateral offset from centerline) | derived from car position vs track spline | `/ track_width` |
| Heading error (angle to track direction) | derived from heading vs track tangent | `/ pi` |
| Distance to left edge | derived or from range-finders | `/ track_width` |
| Distance to right edge | derived or from range-finders | `/ track_width` |
| Track curvature at current position | from track spline | normalize by max curvature |

**Lookahead features (N points ahead, ~20 features):**

Inspired by GT Sophy's 60-point lookahead. Sample N points along the track centerline ahead of the car, spaced by time-to-reach (not fixed distance — so they spread further at high speed):

| Feature per point | Count |
|-------------------|-------|
| Curvature at point | N |
| Track width at point | N |
| Relative heading change | N |
| Distance to point | N |

With N=5 lookahead points: 20 features. Covers ~3-5 seconds of track ahead.

**Previous actions (4 features):**
| Feature | Source |
|---------|--------|
| Previous steering (t-1) | buffer |
| Previous throttle/brake (t-1) | buffer |
| Previous steering (t-2) | buffer |
| Previous throttle/brake (t-2) | buffer |

**Total observation dimension: ~44 features**

This is intentionally compact. GT Sophy used more (especially the 180-feature track geometry encoding), but they had 1,000 PS4s worth of compute. A smaller obs space trains faster on one GPU.

### 4.3 Track Geometry Extraction

AC doesn't expose track spline data through shared memory. Options:

1. **Pre-extract from track files** — AC stores track data in `ai/fast_lane.ai` and `ai/ideal_line.ai` files. Parse these offline to get centerline, boundaries, and curvature. Tools exist in the AC modding community.
2. **Record one manual lap** — drive around slowly, log position at each physics tick, fit a spline. Compute curvature and width from the spline.
3. **Use `normalizedCarPosition`** — this gives 0-1 progress along the track. Combined with `trackSPlineLength`, you get absolute distance. Build up the track map incrementally.

Recommendation: pre-extract from `fast_lane.ai` files. One-time effort per track, gives the highest quality reference line.

---

## 5. Action Space

### 5.1 Design: 2D Combined

Following GT Sophy's approach:

| Action | Range | Meaning |
|--------|-------|---------|
| Steering | [-1, 1] | Full left to full right |
| Acceleration | [-1, 1] | -1 = full brake, 0 = coast, +1 = full throttle |

**Why 2D combined over 3D separate:**
- Simpler for the policy to learn (no need to coordinate throttle and brake independently)
- GT Sophy validated this design at champion level
- The "trail braking" technique (brake + steer simultaneously) is naturally represented
- Fewer action dimensions = faster convergence

**Mapping to vgamepad:**
```python
if action[1] >= 0:
    throttle = action[1]
    brake = 0.0
else:
    throttle = 0.0
    brake = -action[1]
steering = action[0]
```

### 5.2 Action Smoothing

**Critical gotcha:** RL policies oscillate between actions by default. An un-smoothed policy will jitter the steering 60 times per second, which looks insane and exploits physics.

Options (use at least one):

1. **Rate-limit in the environment** — clamp `|action[t] - action[t-1]| < max_delta`. Hard constraint, can't be hacked. Recommended as a starting point.
2. **CAPS regularization** — add `lambda * ||a(s) - a(s')||` to the loss for nearby states. Produces inherently smooth policies (paper: arXiv 2012.06644).
3. **Reward penalty for jerk** — `reward -= alpha * |steering[t] - steering[t-1]|`. Soft constraint, agent can still choose to jerk if the reward is worth it.
4. **Low-pass filter on output** — exponential moving average on actions before sending to vgamepad. Simple but hides the real policy behavior.

**Recommendation:** Start with rate-limiting (max_delta = 0.1 per step at 20Hz) + reward penalty for steering jerk. Add CAPS if needed.

### 5.3 Absolute vs Delta Actions

- **Absolute** (GT Sophy): "set steering to 0.3" — simpler, what the controller actually does
- **Delta** (ETH recommendation): "adjust steering by +0.02" — naturally smooth, but needs dynamic masking near actuator limits

Start with absolute + rate-limiting. It's what GT Sophy used and simpler to debug.

---

## 6. Reward Function

### 6.1 Core Reward: Velocity Along Track

The research consensus (Evans et al., GT Sophy, Nature 2025 reward design paper):

```
r_progress = v_along_track / v_max
```

Where `v_along_track` is the component of velocity along the track centerline direction (not total speed — penalizes sideways motion naturally).

Alternatively, use track-progress-per-step:
```
r_progress = (normalizedCarPosition[t] - normalizedCarPosition[t-1]) * track_length
```

This is simpler and what TMRL uses (waypoint counting). Both work. The velocity version is smoother.

### 6.2 Penalty Terms

| Penalty | Formula | Weight | Purpose |
|---------|---------|--------|---------|
| Off-track | `-1.0` if `numberOfTyresOut >= 4` | 1.0 | Hard boundary |
| Wall contact | `-0.5` on `carDamage` increase | 0.5 | Prevent wall-riding |
| Steering jerk | `-alpha * abs(steer[t] - steer[t-1])` | 0.02 | Smooth driving |
| Excessive slip | `-beta * max(0, abs(slip) - slip_threshold)` | 0.1 | Prevent drifting (slower than grip driving) |
| Wrong direction | `-2.0` if `v_along_track < -1.0` | 1.0 | Prevent reversing |

### 6.3 Terminal Conditions

Episode ends (with penalty) if:
- Car is fully off track for > 2 seconds
- Car is stationary for > 5 seconds (stuck)
- Car is going backwards for > 2 seconds
- Significant collision/damage
- Lap completed (positive terminal bonus)

Episode ends (no penalty) if:
- Maximum steps reached (timeout)

### 6.4 Reward Shaping Gotchas

From the research:

1. **Don't reward the centerline — reward the minimum-curvature line.** The optimal racing line is NOT the center of the track. It cuts corners and uses the full track width. Rewarding centerline adherence produces slow, cautious agents. Either don't penalize lateral position at all (let the agent find its own line) or compute the minimum-curvature reference line and reward proximity to that.

2. **Positive-heavy, not penalty-heavy.** Penalty-only shaping produces agents that creep around tracks avoiding all risk. The progress reward should dominate. Penalties should be small corrections.

3. **Wall-riding exploit.** Without wall-contact penalties, agents discover that scraping walls at certain angles gives free speed boosts (physics engine artifact). Always include a wall-contact penalty.

4. **Reward components fight each other.** Speed reward says "go fast." Slip penalty says "don't slide." These conflict in fast corners where some slip is optimal. Tune carefully — the agent should learn that controlled slip at high speed is better than no slip at low speed.

5. **One bad demo trajectory is enough for waypoint-based rewards.** TMRL's approach — drive one lap (even badly), place waypoints, reward progress past them. This is the lowest-effort reward setup and works surprisingly well as a starting point.

---

## 7. Training Pipeline

### 7.1 Phase 1: Behavioral Cloning

**Data collection:**
- Record AC's built-in AI driving 50-100 laps on the target track
- Or: record yourself driving (if you're fast, your data is better than the AI)
- Or: use rFactor 2 to spectate multiplayer and collect all cars' telemetry + inputs simultaneously
- Dump to disk as numpy arrays: `observations.npy`, `actions.npy`
- Downsample from 100Hz to 20Hz
- Target dataset: ~50K-200K (observation, action) pairs per track

**Training:**
```python
# Simple supervised learning
model = MLP(input_dim=44, hidden=[256, 256, 256], output_dim=2)
optimizer = Adam(lr=3e-4)
loss = MSE(model(obs), actions)
# Train for 50-100 epochs, takes minutes on 3090
```

**Output:**
- `bc_policy.pt` — the behavioral cloning weights
- `bc_reference.pt` — frozen copy for KL anchor during RL

**Expected result:** Agent that can complete laps cleanly, follows a reasonable line, but drives conservatively. Probably 5-15% slower than the demonstration source.

### 7.2 Phase 2: RL Fine-tuning with TQC

**Algorithm: TQC (Truncated Quantile Critics) from sb3-contrib**

TQC is the closest available implementation to GT Sophy's QR-SAC. Drop-in replacement for SAC with distributional critics.

**Starting hyperparameters (from GT Sophy + TQC defaults + racing-specific tuning):**

```python
from sb3_contrib import TQC

model = TQC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,          # actor + critic
    buffer_size=2_000_000,       # large replay buffer
    batch_size=512,              # GT Sophy used 1024, scale to GPU
    tau=0.005,                   # soft target update
    gamma=0.99,                  # discount factor
    train_freq=1,                # update every step
    gradient_steps=32,           # high UTD ratio for sample efficiency
    policy_kwargs=dict(
        net_arch=[512, 512, 512],  # 3 hidden layers
        n_critics=2,               # start with 2, try 5 later
        n_quantiles=25,            # TQC default, 25 quantiles per critic
    ),
    top_quantiles_to_drop_per_net=2,  # truncation for overestimation control
    ent_coef=0.01,               # GT Sophy used fixed 0.01
    verbose=1,
)
```

**KL Anchor (prevent catastrophic forgetting of BC policy):**

```python
# Custom callback that adds KL penalty to the loss
# KL(pi_current || pi_BC) penalizes deviation from BC policy
# Beta = 0.01-0.05, decay over training
# Same approach as GhostLobby's KLAnchorCallback
```

**Replay buffer seeding:**
Load BC demonstration data directly into the replay buffer at the start of training. This gives the critic expert transitions to learn from immediately ("demo-augmented RL").

**Adaptive demo sampling (ARB approach):**
Start with 50% demo / 50% online data in each batch. Decay demo ratio as online data accumulates. The ARB paper (2025) suggests measuring "on-policyness" of demo transitions and fading out the most off-policy ones first.

### 7.3 Phase 3: Curriculum

**Track difficulty progression:**
1. Wide tracks, slow cars (e.g., AC's Mugello with a road car)
2. Wide tracks, fast cars (Mugello with GT3)
3. Narrow tracks, slow cars (Nordschleife sections with road car)
4. Narrow tracks, fast cars (Nordschleife with GT3)

**Speed curriculum (within a track):**
1. Start with speed-limited car (cap at 60% throttle)
2. Gradually increase throttle cap as agent masters current level
3. Full speed when stable at 80% cap

**Advancement criteria:** Complete 10 consecutive laps without off-track or collision, with lap time within 5% of target.

### 7.4 Phase 4: Pushing to Superhuman

Once the agent is consistently fast and clean:

1. **Remove the KL anchor.** Let the policy diverge freely from BC. It no longer needs the safety net.
2. **Increase exploration.** Raise entropy coefficient temporarily to discover new lines.
3. **Optimize for lap time directly.** Switch reward from velocity-along-track to pure lap-time minimization (sparse but now safe because the agent is already competent).
4. **Residual policy (BeTAIL approach).** Freeze the current policy, train a small residual network on top that learns corrections. This is what Sony AI did in their 2024 paper — it avoids catastrophic forgetting entirely while still improving.

---

## 8. Network Architecture

### 8.1 Policy (Actor)

```
Input: obs_dim (44)
  --> Linear(44, 512) + ReLU
  --> Linear(512, 512) + ReLU
  --> Linear(512, 512) + ReLU
  --> Linear(512, 2 * action_dim)  # mean + log_std for each action
Output: SquashedGaussian(mean, std) --> action_dim (2)
```

Squashed Gaussian (tanh on output) ensures actions stay in [-1, 1].

### 8.2 Critic (TQC)

```
Input: obs_dim + action_dim (46)
  --> Linear(46, 512) + ReLU
  --> Linear(512, 512) + ReLU
  --> Linear(512, 512) + ReLU
  --> Linear(512, n_quantiles)  # 25 quantile estimates
```

With `n_critics=2`: two independent copies of this network. The truncation step drops the top `2 * 2 = 4` quantile values from the combined `2 * 25 = 50` estimates, then averages the remaining 46 for the target Q-value. This controls overestimation bias.

### 8.3 Why Not RNN/LSTM?

GT Sophy used recurrent networks because their observation didn't include explicit velocity/acceleration in early versions. With full telemetry (speed, accel, slip angles all observed directly), an MLP with previous-action features captures sufficient temporal context. RNNs add complexity and training instability for minimal gain in this setting.

If multi-car racing is added later, consider an attention mechanism over opponent features (variable number of opponents).

---

## 9. Infrastructure

### 9.1 Software Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| RL algorithm | sb3-contrib TQC | Distributional SAC, drop-in |
| BC pretraining | PyTorch (manual) | Simple supervised learning |
| Environment wrapper | Gymnasium | Standard RL interface |
| Telemetry reader | Custom (ctypes + mmap) | Read AC shared memory |
| Controller output | vgamepad | Emulate Xbox 360 |
| Real-time timing | Custom (based on TMRL's rtgym pattern) | Enforce 20Hz step timing |
| Logging | TensorBoard + Weights & Biases | Training curves, metrics |
| Track data | Custom parser for AC's fast_lane.ai | Centerline, curvature, width |

### 9.2 Gymnasium Environment Wrapper

```python
class AssettoCorsaEnv(gymnasium.Env):
    observation_space = Box(low=-1, high=1, shape=(44,))
    action_space = Box(low=-1, high=1, shape=(2,))

    def __init__(self, track, car, config):
        self.telemetry = ACSharedMemory()
        self.gamepad = vgamepad.VX360Gamepad()
        self.track_data = load_track(track)  # pre-extracted centerline/curvature
        self.step_freq = 20  # Hz

    def step(self, action):
        self._send_action(action)
        self._wait_for_next_tick()  # enforce 20Hz timing
        obs = self._read_telemetry()
        reward = self._compute_reward(obs)
        terminated = self._check_terminal(obs)
        return obs, reward, terminated, False, {}

    def reset(self):
        # Teleport car to start (AC console command or restart session)
        # Wait for car to be stationary
        return self._read_telemetry(), {}
```

### 9.3 Reset Mechanism

This is a known pain point for real-game RL. Options:

1. **AC console commands** — AC has a developer console that can restart sessions. Scriptable.
2. **Content Manager API** — Content Manager (popular AC launcher) has automation features.
3. **Keyboard macro** — simulate Escape > Restart > Go. Crude but reliable.
4. **No reset / continuous training** — don't reset on crash. Apply large penalty, let the agent recover and continue. This is what TMRL does for some configurations. Removes the reset infrastructure entirely but makes reward attribution harder.

Recommendation: Start with keyboard macro resets (simplest). Move to Content Manager API if the macro proves unreliable.

### 9.4 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3090 (24GB) | Same — more than enough for MLP-based policy |
| CPU | Any modern 4+ core | M4 Max for development, Windows desktop for training |
| RAM | 16GB | 32GB (replay buffer lives in RAM) |
| OS | Windows 10/11 (AC requirement) | Same |
| Storage | 10GB for AC + data | SSD for telemetry logging throughput |

**Note:** Training runs on the Windows desktop with the 3090 (AC must run there). Development, analysis, and BC pretraining can happen on the M4 Max.

### 9.5 Training Time Estimates

Based on comparable projects:

| Phase | Estimated wall-clock time |
|-------|--------------------------|
| Environment setup + telemetry integration | 1-2 weeks |
| BC data collection (50 laps) | 2-3 hours |
| BC pretraining | 30 minutes |
| RL fine-tuning to "competent" (matches BC) | 2-5 days (continuous training) |
| RL fine-tuning to "fast" (approaches human) | 1-3 weeks |
| RL fine-tuning to "superhuman" (beats record) | Unknown — this is the research question |

**Comparison points:**
- TMRL (TrackMania, vision-based, SAC from scratch): months, still 30% slower than humans
- GT Sophy (1000 PS4s, privileged API): days of wall-clock (massive parallelism)
- ETH F1TENTH (telemetry, PPO, small-scale car): hours
- Linesight (TrackMania, from scratch): 169 days, never reached superhuman

The BC warm-start is the key differentiator. Starting from competent BC policy vs from scratch should compress the RL phase by 10-100x (Doug's own GhostLobby data showed 61% improvement with BC warm-start at same step count).

---

## 10. Evaluation & Metrics

### 10.1 Primary Metric

**Lap time** on the target track. Compare to:
- AC's built-in AI lap time (easy target)
- Your personal best (medium target)
- Track record from online leaderboards (hard target)
- World record (the headline)

### 10.2 Secondary Metrics

| Metric | Purpose |
|--------|---------|
| Consistency (std of lap times over 20 laps) | Reliability — superhuman isn't useful if it crashes every 3rd lap |
| Sector times | Identifies which parts of the track are strong/weak |
| Off-track frequency | Safety metric |
| Steering smoothness (mean |d_steer/dt|) | Realism — jittery steering = physics exploit |
| Average tire slip angle | Driving style — too high means drifting (suboptimal), too low means not pushing |
| Throttle/brake transition smoothness | Trail-braking quality |

### 10.3 Benchmark Tracks

Start with one, expand:

1. **Monza** (primary) — fast, iconic, well-known lap records. Long straights + heavy braking zones + fast chicanes. Good mix of skills. Every sim racer knows Monza times.
2. **Spa-Francorchamps** (secondary) — technically demanding, elevation changes, blind corners. Tests generalization.
3. **Nordschleife** (stretch goal) — 20+ km, 170+ corners. If the agent can do this, it can do anything.

### 10.4 Demo / Content Plan

The demo that gets attention:

1. **Training timelapse** — 60-second video showing the agent's progress from BC (cautious) through RL (fast) to superhuman (alien). Show the racing line evolving, lap time dropping. Overlay the reward curve.
2. **Side-by-side comparison** — split screen: agent's lap vs human record holder's lap. Same track, same car. Show where the agent gains time.
3. **Telemetry overlay** — show steering, throttle, brake traces alongside the video. Sim racers live for telemetry comparisons. This is the "proof it's real" content.
4. **Open-source the pipeline** — GitHub repo with clean code, documentation, and instructions to reproduce. This is the credential.

---

## 11. Known Risks & Mitigations

### 11.1 Technical Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| AC shared memory timing issues (stale data) | Medium | Check `packetId` changes, skip stale reads |
| vgamepad latency adds delay to control loop | Low | Measured at <1ms, negligible at 20Hz |
| Agent exploits physics engine glitches | High | Wall-contact penalty, steering smoothness constraint, visual inspection of trained laps |
| Training diverges after BC → RL transition | High | KL anchor (beta=0.01-0.05), residual policy (BeTAIL), demo replay buffer seeding |
| Reset mechanism unreliable | Medium | Start with continuous training (no reset), add macro resets later |
| Graphics page updates too slow (0.1Hz default) | Medium | Compute track progress from physics data + pre-extracted track spline instead |
| Sample efficiency insufficient on single GPU | Medium | High UTD ratio (gradient_steps=32), demo replay seeding, n-step returns |

### 11.2 Scope Risks

| Risk | Mitigation |
|------|------------|
| "Superhuman" might not be achievable on single GPU | Define success tiers: (1) beats AC AI, (2) beats your PB, (3) top 10% leaderboard, (4) record. Each tier is a valid demo. |
| Project takes months instead of weeks | BC + basic RL should produce a compelling demo in 2-3 weeks even if not superhuman. The video of training progress is content regardless of final performance. |
| Scope creep into multi-car racing | Stay time-trial only until time-trial is solved. Multi-car is a separate project. |

---

## 12. Research References

### Core Papers

| Paper | Year | Key Contribution | Relevance |
|-------|------|-------------------|-----------|
| [Outracing champion GT drivers with deep RL](https://www.nature.com/articles/s41586-021-04357-7) (Nature) | 2022 | QR-SAC, beat world champions in GT Sport | Architecture and training methodology baseline |
| [Champion-level Vision-based Agent in GT7](https://arxiv.org/abs/2504.09021) | 2025 | Asymmetric actor-critic, vision-only competitive racing | Vision approach reference (not our path but useful for comparison) |
| [On Learning Racing Policies with RL](https://arxiv.org/abs/2504.02420) | 2025 | Zero-shot sim-to-real, beat humans on F1TENTH | Obs/action design, delta vs absolute actions |
| [Assetto Corsa Gym Benchmark](https://arxiv.org/abs/2407.16680) | 2024 | AC as RL benchmark, 125-dim obs, SAC baseline | Direct comparison target — beat their results |
| [BeTAIL: Behavior Transformer Adversarial IL](https://arxiv.org/abs/2402.14194) | 2024 | Residual policy on frozen BC, cross-track transfer | BC-to-RL transition without forgetting |

### Training Methodology

| Paper | Year | Key Contribution |
|-------|------|-------------------|
| [Hybrid IL+RL Racing](https://arxiv.org/abs/2110.05437) | 2021 | BC pretrain → RL fine-tune beat human by 0.96s in <20 hours |
| [Imitation Is Not Enough](https://arxiv.org/abs/2212.11419) | 2022 | (Waymo) BC alone fails on edge cases, RL fine-tune reduces failures 38% |
| [Reward Design for Autonomous Racing](https://arxiv.org/abs/2103.10098) | 2021 | Minimum-curvature reference line beats centerline for rewards |
| [Reward Design for Generalizable Racing Agents](https://www.nature.com/articles/s41598-025-27702-6) | 2025 | Systematic reward component evaluation, Nature Scientific Reports |
| [Automated Reward Design for GT](https://arxiv.org/abs/2511.02094) | 2025 | LLM-based reward search for racing |
| [CAPS: Smooth Control via Regularization](https://arxiv.org/abs/2012.06644) | 2020 | Action smoothness regularization |
| [Adaptive Replay Buffer for O2O RL](https://arxiv.org/abs/2512.10510) | 2025 | Dynamic demo/online sampling ratio |

### Algorithm

| Paper | Year | Key Contribution |
|-------|------|-------------------|
| [TQC: Truncated Quantile Critics](https://proceedings.mlr.press/v119/kuznetsov20a.html) | 2020 | Distributional SAC with truncation, sb3-contrib implementation |
| [REDQ: Randomized Ensemble Double Q](https://arxiv.org/abs/2101.05982) | 2021 | Alternative to TQC: ensemble critics, high UTD ratio |
| [Physics-Informed RL for Map-Free Racing](https://arxiv.org/abs/2604.09499) | 2026 | No map needed, physics-informed rewards |

### Practical / Engineering

| Resource | Key Contribution |
|----------|-------------------|
| [TMRL (TrackMania RL)](https://github.com/trackmania-rl/tmrl) | Proven vgamepad + screen capture architecture, distributed training |
| [Assetto Corsa Gym](https://github.com/dasGringuen/assetto_corsa_gym) | AC shared memory integration, Gymnasium wrapper |
| [Linesight (TrackMania)](https://github.com/Linesight-RL/linesight) | 169-day training log, gotchas documented |
| [SAC Tuning for Isaac Sim](https://araffin.github.io/post/tune-sac-isaac-sim/) | Practical SAC hyperparameter tuning (Raffin/SB3 author) |
| [SB3 Tips and Tricks](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) | General RL training advice from SB3 maintainers |

---

## 13. Success Tiers

| Tier | Definition | Timeline | Demo Value |
|------|-----------|----------|------------|
| **Bronze** | Agent completes laps cleanly, beats AC's built-in AI | Week 2-3 | "It works" — proof of concept |
| **Silver** | Agent matches or beats a competent human (your PB or online average) | Week 4-6 | "It's good" — technically impressive |
| **Gold** | Agent is in top 10% of online leaderboard times | Month 2-3 | "It's better than most humans" — gets attention |
| **Platinum** | Agent beats the track record | Month 3-6+ | "Solo dev replicates GT Sophy" — the headline |

Each tier is a valid stopping point with its own demo/content value. Bronze alone is a better portfolio piece than most ML engineers have. Gold is exceptional. Platinum makes the news.

---

## 14. Open Questions

1. **Which car class to train on?** GT3 cars are popular and well-modeled. Road cars are simpler (lower grip, lower speed, more forgiving). Start with a road car for faster iteration, then transfer to GT3.

2. **Can the policy transfer across cars on the same track?** Unknown. GT Sophy trained per-car. Transfer learning across cars would be impressive but might not work due to different tire models and aero.

3. **How to handle the Graphics page 0.1Hz limitation?** Compute track progress entirely from Physics page data + pre-extracted track spline. Don't rely on `normalizedCarPosition` during training.

4. **Is 20Hz sufficient for fast chicanes?** At 300kph through Monza's first chicane, 20Hz = one decision every 4.2m. This might be too coarse. Consider 30Hz or adaptive frequency (higher in corners, lower on straights).

5. **Should we use n-step returns?** GT Sophy switched from 5-step to 7-step returns. Multi-step returns improve sample efficiency but increase bias. Start with 1-step, experiment with 3 and 7.

6. **rFactor 2 for multi-agent BC data?** rF2 exposes all cars' inputs. Could collect BC data from 20+ cars simultaneously during a race, giving massive expert datasets with diverse driving styles. Worth exploring for Phase 2.

7. **Headless AC?** AC doesn't have a headless mode. Training requires the game to be rendering. Could potentially minimize the window or reduce graphics quality to reduce GPU load (the 3090 needs to run both the game and the training). Test whether AC runs stable at minimum graphics while TQC trains on the same GPU.

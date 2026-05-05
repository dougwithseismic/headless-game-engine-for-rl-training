# Headless FPS Engine — Product Spec

A configurable, zero-dependency C++ FPS simulation engine built for training AI agents at 100K+ ticks/sec. No rendering, no game client, no bloat — pure mechanics exposed via a Python Gym API.

Configure it to match CS2, Valorant, Rainbow Six, or any FPS via a JSON file. Train RL agents, generate synthetic gameplay data, or study game mechanics in isolation.

---

## What It Is

A standalone C++ library that simulates the core mechanics of any first-person shooter:

- Character movement (walking, running, jumping, crouching, air control)
- Hitscan and projectile weapon ballistics
- Recoil and spray patterns
- Health, armor, and damage models
- Line-of-sight and visibility checks
- Simple 3D map geometry (navmesh + collision volumes)
- Economy system (buy phase, money per round/kill)
- Round/match state machine

Everything is driven by a JSON config file. Swap the config, swap the game.

---

## What It Is NOT

- Not a game engine (no rendering, no audio, no asset pipeline)
- Not a cheat or a hack (no game integration required)
- Not a perfect replica of any specific game (accurate enough for training, not frame-perfect)
- Not a product for sale (open source portfolio piece / research tool)

---

## Architecture

```
┌─────────────────────────────────────┐
│           Python Training           │
│   (Gymnasium / SB3 / CleanRL)       │
└──────────────┬──────────────────────┘
               │ step() / reset()
               │ via pybind11
┌──────────────▼──────────────────────┐
│          Gym Wrapper (C++)          │
│  - Observation space builder        │
│  - Action space mapper              │
│  - Reward function (configurable)   │
│  - Episode management               │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│        Simulation Core (C++)        │
│                                     │
│  ┌───────────┐  ┌────────────────┐  │
│  │ Character  │  │    Weapons     │  │
│  │ Controller │  │  - Hitscan     │  │
│  │ - Movement │  │  - Projectile  │  │
│  │ - Collision│  │  - Recoil      │  │
│  │ - Jumping  │  │  - Spray       │  │
│  └───────────┘  └────────────────┘  │
│                                     │
│  ┌───────────┐  ┌────────────────┐  │
│  │   Map     │  │   Game State   │  │
│  │ - Navmesh │  │  - Rounds      │  │
│  │ - Collider│  │  - Economy     │  │
│  │ - Spawns  │  │  - Score       │  │
│  │ - Cover   │  │  - Buy phase   │  │
│  └───────────┘  └────────────────┘  │
│                                     │
│  ┌───────────┐  ┌────────────────┐  │
│  │ Visibility│  │   Damage       │  │
│  │ - Raycasts│  │  - HP/Armor    │  │
│  │ - FOV     │  │  - Headshot    │  │
│  │ - Occlusion│ │  - Falloff     │  │
│  └───────────┘  └────────────────┘  │
│                                     │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│     Config Loader (JSON)            │
│  - Movement params                  │
│  - Weapon definitions               │
│  - Map geometry                     │
│  - Economy rules                    │
│  - Game mode rules                  │
└─────────────────────────────────────┘

               ┌──────────────────────┐
               │  Web Viewer (opt.)   │
               │  - WebSocket stream  │
               │  - React + Canvas    │
               │  - 2D top-down view  │
               │  - Agent POV overlay │
               └──────────────────────┘
```

---

## Config System

Every game-specific behavior is driven by a JSON config. The engine has no hardcoded game knowledge.

```json
{
  "title": "cs2-like",
  "tick_rate": 128,

  "movement": {
    "walk_speed": 250.0,
    "run_speed": 250.0,
    "crouch_speed": 85.0,
    "acceleration": 5.5,
    "deceleration": 5.0,
    "air_acceleration": 100.0,
    "air_speed_cap": 30.0,
    "gravity": 800.0,
    "jump_velocity": 301.99,
    "crouch_height": 54.0,
    "stand_height": 72.0,
    "enable_air_strafe": true,
    "enable_counter_strafe": true
  },

  "weapons": [
    {
      "name": "rifle",
      "type": "hitscan",
      "damage": 27,
      "headshot_multiplier": 4.0,
      "fire_rate": 0.1,
      "range": 4096.0,
      "armor_penetration": 0.775,
      "spread_base": 0.6,
      "spread_moving": 5.0,
      "recoil_pattern": [
        [0.0, -0.8],
        [0.1, -0.9],
        [-0.2, -1.0],
        [0.3, -0.7],
        [-0.4, -0.6]
      ],
      "spray_reset_time": 0.55,
      "magazine_size": 30,
      "reload_time": 3.1,
      "switch_time": 0.5,
      "cost": 2700
    }
  ],

  "damage": {
    "max_health": 100,
    "max_armor": 100,
    "armor_damage_reduction": 0.5,
    "friendly_fire": false,
    "headshot_lethal_range": 4096.0
  },

  "economy": {
    "start_money": 800,
    "max_money": 16000,
    "kill_reward": 300,
    "round_win_bonus": 3250,
    "round_loss_base": 1400,
    "round_loss_increment": 500,
    "round_loss_max": 3400
  },

  "rules": {
    "round_time": 115.0,
    "freeze_time": 15.0,
    "rounds_to_win": 13,
    "players_per_team": 5,
    "respawn": false,
    "bomb_plant_time": 3.2,
    "bomb_defuse_time": 10.0,
    "bomb_explode_time": 40.0
  }
}
```

Switch `enable_air_strafe` to false, slow the movement, add ability cooldowns — it's Valorant. Change TTK, add lean mechanics — it's Rainbow Six.

---

## Simulation Core — Module Breakdown

### 1. Character Controller

Handles all player movement and collision.

**Inputs:** directional movement (forward/back/strafe), jump, crouch, sprint
**Outputs:** position (vec3), velocity (vec3), view angles (pitch/yaw), stance (stand/crouch)

Physics:

- Ground movement: acceleration toward desired velocity, capped at max speed per stance
- Air movement: source-engine-style air strafing (configurable) — accelerate in wish direction, capped by air_speed_cap
- Counter-strafing: optional instant velocity kill on opposite input
- Jump: instant vertical velocity impulse
- Crouch: height change, speed reduction
- Collision: AABB vs map geometry (walls, floors, ceilings, ramps)

No skeletal animation, no footstep sounds, no viewmodel. Just a capsule with position, velocity, and a viewangle.

### 2. Weapons

Each weapon is defined in the config. The engine supports:

**Hitscan:** Instant raycast from eye position along view direction + spread offset. Check for intersections with enemy player AABBs. Apply damage based on hit region (head/body/legs), distance falloff, and armor.

**Projectile:** Spawn entity at muzzle with velocity vector. Update position per tick. Check for collision. Apply splash damage within radius. (For grenades, molotovs, etc.)

**Recoil:** Per-shot offset applied to view angles. Pattern is an array of (x, y) deltas. Resets after spray_reset_time of no firing. Agent must learn to compensate.

**Spread:** Base inaccuracy + movement penalty. Samples a random offset within a cone. Rewards standing still.

### 3. Map Geometry

Maps are not full 3D level designs. They are simplified representations:

- **Navmesh:** Walkable surfaces as a polygon mesh. Loaded from file (Recast/Detour format or custom). Used for pathfinding and spatial queries.
- **Collision volumes:** AABBs or convex hulls for walls, cover objects, ramps. Used for movement collision and hitscan occlusion.
- **Spawn points:** Per-team spawn positions.
- **Objectives:** Bomb sites (positions + radii), hostage locations, or flag positions.
- **Cover annotations:** Positions marked as "cover" with facing direction. Optional — agents can learn cover from geometry, but annotations speed up training.

Map format: JSON or a simple binary format. Can be hand-authored, procedurally generated, or extracted from real game navmeshes.

### 4. Visibility / Line of Sight

**Raycasting:** Line segment vs AABB intersection (Slab method). Used for:

- Hitscan weapon fire
- Agent visibility checks (can I see enemy X?)
- Cover evaluation

**FOV cone:** Agents have a configurable field of view. Enemies outside FOV are not visible even if unoccluded.

**Occlusion:** Raycast from agent eye position to enemy center. If any collision volume intersects, enemy is not visible.

No smoke, no flash, no fog-of-war in v1. Add these as configurable modules later.

### 5. Damage Model

Simple and configurable:

```
effective_damage = base_damage * region_multiplier
if target has armor:
  armor_absorbed = effective_damage * armor_reduction * armor_penetration
  health_damage = effective_damage - armor_absorbed
  armor_damage = armor_absorbed * 2
else:
  health_damage = effective_damage

target.health -= health_damage
target.armor -= armor_damage
```

Hit regions: head (configurable multiplier, typically 4x), body (1x), legs (0.75x). Region determined by raycast intersection height relative to player AABB.

### 6. Game State Machine

```
WARMUP → FREEZE_TIME → LIVE → ROUND_END → FREEZE_TIME → ...
                                    ↓
                              MATCH_END (first to rounds_to_win)
```

- FREEZE_TIME: buy phase, no movement. Economy resolved.
- LIVE: players can move, shoot, plant/defuse bomb.
- ROUND_END: triggered by elimination, bomb explosion, bomb defuse, or time expiry.
- Economy updates between rounds based on win/loss streaks.

### 7. Economy

Per-player money tracking:

- Start money per half
- Kill rewards (per weapon type, configurable)
- Round win/loss bonuses (with loss streak escalation)
- Buy menu: spend money on weapons/armor during freeze time

Agent must learn economic decisions: when to save, when to force buy, when to full buy.

---

## Observation Space

What the agent sees each tick:

```python
observation = {
    # Self
    "position": [x, y, z],            # vec3
    "velocity": [vx, vy, vz],         # vec3
    "view_angles": [pitch, yaw],       # vec2
    "health": 100,                     # int
    "armor": 100,                      # int
    "money": 4750,                     # int
    "current_weapon": 2,               # int (weapon index)
    "ammo_clip": 30,                   # int
    "ammo_reserve": 90,                # int
    "stance": 0,                       # 0=stand, 1=crouch
    "on_ground": True,                 # bool
    "is_reloading": False,             # bool

    # Visible enemies (variable length, padded)
    "visible_enemies": [
        {"position": [x,y,z], "velocity": [vx,vy,vz], "health_estimate": 0.8},
    ],

    # Audible events (footsteps, gunfire — directional)
    "audio_events": [
        {"direction": [dx, dy, dz], "type": "footstep", "distance": 450.0},
    ],

    # Team state
    "alive_teammates": 4,              # int
    "alive_enemies": 5,                # int

    # Round state
    "round_time_remaining": 87.5,      # float
    "bomb_planted": False,             # bool
    "bomb_time_remaining": 0.0,        # float

    # Spatial awareness
    "nearest_cover_direction": [dx, dy, dz],  # vec3
    "nearest_cover_distance": 120.0,          # float
}
```

Configurable: can expose more or less information depending on what you're training for. Training with perfect information (wallhack mode) vs realistic information produces different agent behaviors — useful for anti-cheat data generation.

---

## Action Space

```python
action = {
    "move_forward": [-1.0, 1.0],      # continuous
    "move_right": [-1.0, 1.0],        # continuous
    "look_pitch": [-5.0, 5.0],        # continuous (degrees per tick)
    "look_yaw": [-5.0, 5.0],          # continuous (degrees per tick)
    "shoot": 0 or 1,                   # discrete
    "reload": 0 or 1,                  # discrete
    "jump": 0 or 1,                    # discrete
    "crouch": 0 or 1,                  # discrete
    "weapon_switch": 0-4,              # discrete (weapon slot)
    "buy": 0-N,                        # discrete (item index, only during freeze)
}
```

Can be flattened to a single vector for standard RL algorithms. Continuous actions (movement, look) can be discretized if needed (like Nexto's 90-action approach).

---

## Reward Function (Configurable)

Default reward function for general FPS training:

```python
reward = (
    # Combat
    + 10.0 * kills_this_tick
    + 3.0  * damage_dealt_this_tick / 100.0
    - 5.0  * deaths_this_tick
    - 1.0  * damage_taken_this_tick / 100.0

    # Objectives
    + 15.0 * bomb_planted_this_tick
    + 15.0 * bomb_defused_this_tick
    + 20.0 * round_won_this_tick
    - 10.0 * round_lost_this_tick

    # Behavioral shaping (small, decay over training)
    + 0.01 * distance_moved          # don't camp spawn
    - 0.005 * time_not_visible_to_enemies  # encourage engagement
    + 0.1  * crosshair_on_enemy_time       # reward aim tracking
)
```

Fully overridable via config or Python callback. Different reward functions for different training objectives:

- **QA testing:** reward exploration and finding map boundaries
- **Anti-cheat data:** reward natural-looking play patterns
- **Competitive training:** reward winning and objective play
- **Exploit hunting:** reward unusual positions and unintended damage

---

## System Identification Pipeline

Fitting engine parameters to match a real game's mechanics:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Real Game       │     │  Parameter        │     │  Engine Config  │
│  Telemetry       │────▶│  Fitting          │────▶│  (JSON)         │
│  (demos/replays) │     │  (scipy.optimize) │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

**Movement parameters:**

1. Parse real game demos (awpy for CS2, community parsers for others)
2. Extract: player positions over time, input states, stance changes
3. Fit acceleration/deceleration/max_speed/air_control curves to observed trajectories
4. Validate: simulate same inputs in engine, compare position traces

**Weapon parameters:**

1. Extract: damage events, hit regions, distances, weapon types
2. Fit: base damage, falloff curves, headshot multipliers, fire rates
3. Spray patterns: extract recoil from consecutive shot deltas in demos
4. Validate: simulate same engagement, compare damage output

**Economy parameters:**

1. Extract: money values per player per round from demos
2. Reverse: kill rewards, round bonuses, loss streak values
3. These are usually documented on community wikis anyway

**Map geometry:**

1. Extract navmesh from game files (if accessible) or
2. Build from player position data (convex hull of all observed positions = walkable area)
3. Extract collision from raycasts that failed in demos (player aimed at X, bullet stopped at Y = wall at Y)

Output: a JSON config file that makes the engine behave like the target game within acceptable tolerance.

---

## Web Viewer

Optional real-time visualization for debugging and demos. NOT part of the training loop.

- WebSocket server in C++ (lightweight, e.g., uWebSockets or custom)
- Streams game state as JSON at configurable rate (30-60 fps for viewing)
- React + Canvas 2D client:
  - Top-down map view with player positions, FOV cones, bullet traces
  - Health/armor bars per player
  - Kill feed
  - Round state, score, economy
  - Heatmap overlay (accumulate positions over time)
- Optional: Three.js 3D view for first-person perspective

The viewer is what makes demo videos compelling. Training runs headless. You open the viewer when you want to watch.

---

## Build Plan

### Phase 1: Core Loop (Week 1-2)

- [ ] Project scaffold: CMake, pybind11 bindings, basic Gym wrapper
- [ ] Vec3 math (position, velocity, direction — no external deps)
- [ ] Character controller: ground movement, jumping, gravity, AABB collision
- [ ] Box room map: four walls, floor, ceiling. Hardcoded geometry.
- [ ] JSON config loader: movement params only
- [ ] Gym wrapper: step/reset, observation (position + velocity), action (move + look)
- [ ] Verify: PPO agent learns to move to a target position in the box room
- [ ] Benchmark: measure ticks/sec, target 100K+

Milestone: agent running around a box room, training at 100K+ ticks/sec.

### Phase 2: Combat (Week 3-4)

- [ ] Hitscan weapon system: raycast, damage, hit regions
- [ ] Recoil and spread
- [ ] Health and armor
- [ ] Second agent (opponent): stationary target, then moving target, then fighting back
- [ ] Kill/death detection, respawn logic
- [ ] Reward function: damage dealt, kills, deaths
- [ ] Verify: agent learns to aim and shoot at moving targets

Milestone: two agents fighting in a box room. One visibly learning to shoot the other.

### Phase 3: Maps & Navigation (Week 5-6)

- [ ] Navmesh loader (Recast/Detour or custom simple format)
- [ ] Collision volumes: walls, cover objects, ramps
- [ ] Raycasting for line-of-sight / visibility
- [ ] FOV cone visibility checks
- [ ] Spawn points, basic round flow (spawn → fight → reset)
- [ ] Simple map: two rooms connected by a corridor with cover objects
- [ ] Verify: agent learns to use cover and peek corners

Milestone: agents fighting on a real-ish map with cover and visibility.

### Phase 4: Game Rules & Economy (Week 7-8)

- [ ] Round state machine: freeze time → live → round end → repeat
- [ ] Economy: money, buy phase, weapon purchasing
- [ ] Bomb plant/defuse mechanics (objective mode)
- [ ] Team logic: 5v5, alive counts, team-based rewards
- [ ] Round/match win conditions
- [ ] Verify: agents learn economic decisions (save vs buy)

Milestone: full 5v5 round-based matches running at scale.

### Phase 5: System Identification (Week 9-10)

- [ ] CS2 demo parser integration (awpy or custom)
- [ ] Movement parameter fitting from demo data
- [ ] Weapon parameter fitting from damage events
- [ ] Economy parameter extraction
- [ ] Map geometry extraction from position data
- [ ] Output: cs2.json config file
- [ ] Validate: compare simulated trajectories to real demo trajectories

Milestone: engine configured as CS2-like, validated against real data.

### Phase 6: Viewer & Polish (Week 11-12)

- [ ] WebSocket state streaming from engine
- [ ] React + Canvas 2D top-down viewer
- [ ] Kill feed, health bars, round state display
- [ ] Heatmap overlay
- [ ] Recording: output state log for replay
- [ ] README, docs, example configs, example training scripts
- [ ] Demo video: agent learning to play CS2-like from scratch, time-lapse

Milestone: ship it. GitHub, tweet, demo video.

---

## Performance Targets

| Metric                     | Target                                             |
| -------------------------- | -------------------------------------------------- |
| Tick rate                  | 100,000+ ticks/sec (single-threaded, no rendering) |
| Memory per instance        | < 10 MB                                            |
| Parallel instances         | Limited only by CPU cores                          |
| Observation build time     | < 1 μs                                             |
| Step time (full tick)      | < 10 μs                                            |
| Python overhead (pybind11) | < 5% of total step time                            |
| Config load time           | < 100 ms                                           |
| Map load time              | < 1 sec for typical FPS map navmesh                |

---

## Anti-Cheat Data Generation Mode

A key secondary use case. The engine can generate perfectly-labeled gameplay data for training anti-cheat ML models.

**Legit agents:** Trained via RL with human-like constraints:

- Reaction time floor (200ms minimum response to new visual stimuli)
- Aim noise (gaussian jitter on crosshair tracking)
- Decision variance (don't always make the optimal play)
- Fatigue modeling (accuracy degrades over long sessions)

**Cheat agents:** Same base agent with synthetic enhancements:

- Aimbot: snap to nearest enemy head position within N ms
- Wallhack: full enemy position information regardless of visibility
- Triggerbot: auto-fire when crosshair intersects enemy hitbox
- Speedhack: movement speed multiplied by configurable factor
- Recoil compensation: perfect counter-strafe recoil pattern

**Output format:**

```json
{
  "tick": 14523,
  "player_id": 7,
  "label": "aimbot",
  "confidence": 1.0,
  "state": {
    "position": [1024.0, -512.0, 64.0],
    "view_angles": [2.1, -45.7],
    "view_delta": [0.3, -12.4],
    "target_angle_to_nearest_enemy_head": [2.4, -45.9],
    "time_since_last_target_switch": 0.05,
    "shots_fired": 3,
    "shots_hit": 3,
    "is_moving": false,
    "weapon": "rifle"
  }
}
```

Every sample is ground-truth labeled. No manual annotation. Generate millions per hour.

---

## Tech Stack Summary

| Layer           | Tech                          | Why                             |
| --------------- | ----------------------------- | ------------------------------- |
| Simulation core | C++17, no external deps       | Speed, portability, zero bloat  |
| Linear algebra  | Header-only (custom or Eigen) | No build complexity             |
| Navmesh         | Recast/Detour (optional)      | Industry standard, MIT licensed |
| Python bindings | pybind11                      | Cleanest C++ ↔ Python bridge    |
| Gym API         | Gymnasium (Farama)            | Universal RL interface          |
| RL training     | Stable Baselines 3 or CleanRL | Proven PPO/SAC implementations  |
| Config          | nlohmann/json (header-only)   | Single-header JSON parser       |
| Web viewer      | uWebSockets + React + Canvas  | Lightweight, no framework bloat |
| Build system    | CMake                         | Cross-platform, standard        |
| CI              | GitHub Actions                | Free for open source            |

---

## What Success Looks Like

1. An agent visibly learns to play from scratch — a time-lapse from spinning in circles to getting kills
2. The same engine configured as two different games (CS2-like and Valorant-like) producing agents with visibly different playstyles
3. Anti-cheat data generation demo: side-by-side visualization of legit agent vs aimbot agent, with the behavioral differences highlighted
4. System identification demo: real CS2 telemetry in, configured engine out, simulated trajectories matching real ones
5. Training benchmark: 1 million ticks in < 10 seconds, agent reaches competent play in < 1 hour
6. Clean open-source repo that someone can clone, pip install, and start training in under 5 minutes

---

## Name Ideas

- **FragSim**
- **HollowPoint**
- **DryFire** (practice without live ammo)
- **BlankRound**
- **Tickrate**
- **GhostLobby**

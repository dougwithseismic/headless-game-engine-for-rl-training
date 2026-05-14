# GhostLobby Virtual Gamepad

Cross-platform virtual Xbox 360 controller for driving games, RL agents, or anything that reads gamepad input. On Windows with ViGEmBus installed, it creates a real system-level Xbox 360 controller that Steam and all games see. On Mac/Linux it falls back to a mock backend for development and testing.

## File layout

```
bridges/gamepad/
  protocol.py          # GamepadBackend protocol, Axis/Button enums, GamepadState, apply_deadzone()
  mock_backend.py      # MockGamepad — no OS deps, history ring buffer, ASCII debug viz
  vgamepad_backend.py  # VGamepadBackend — real Xbox 360 via ViGEmBus (Windows only)
  pid.py               # PIDController + GamepadPIDController — smooth ramp per axis
  factory.py           # make_gamepad() and make_pid_gamepad() — auto-detect OS
  visualizer.py        # Browser GUI server (WebSocket + HTTP on localhost:8765)
  visualizer.html      # Tailwind-based controller visualization
  test_gamepad.py      # 30 pytest tests
```

## Windows setup

This is where the gamepad does real work. Three steps:

1. **Install ViGEmBus driver** — download the latest `.msi` from
   https://github.com/nefarius/ViGEmBus/releases and run it. Reboot if prompted.

2. **Install the Python library** in your project venv:
   ```bash
   pip install vgamepad
   ```

3. **Use it** — `make_gamepad()` on Windows auto-selects the real backend:
   ```python
   from bridges.gamepad.factory import make_gamepad

   pad = make_gamepad()          # auto: VGamepadBackend on Windows
   pad.connect()
   pad.set_axis("left_stick_x", 0.5)
   pad.update()
   # Windows now sees a real Xbox 360 controller with left stick pushed right
   pad.disconnect()
   ```

After `connect()`, the controller appears in Windows Game Controllers (`joy.cpl`), Steam Big Picture, and any game that reads XInput.

## Mac / Linux

On non-Windows platforms, `make_gamepad()` returns a `MockGamepadBackend`. It implements the same API but does not create an OS-level device. All axis/button state is tracked in memory with a history ring buffer, which is useful for testing and development.

```python
from bridges.gamepad.factory import make_gamepad

pad = make_gamepad()  # auto: MockGamepadBackend on Mac/Linux
pad.connect()
pad.set_axis("left_stick_x", 0.7)
pad.update()
print(pad.get_state().axes)   # {Axis.LEFT_STICK_X: 0.68421..., ...}
pad.print_state()              # ASCII controller diagram
pad.disconnect()
```

If you explicitly request `backend="vgamepad"` on Mac, it raises `RuntimeError`.

## Usage

### Basic: set axes and buttons

```python
from bridges.gamepad.factory import make_gamepad

pad = make_gamepad(backend="auto")
pad.connect()

# Sticks: range [-1.0, 1.0]
pad.set_axis("left_stick_x", 0.5)
pad.set_axis("left_stick_y", -0.3)

# Triggers: range [0.0, 1.0]
pad.set_axis("right_trigger", 0.8)

# Buttons
pad.set_button("a", True)

# Flush all pending state to the virtual controller
pad.update()

# Release everything
pad.reset()

pad.disconnect()
```

String axis/button names auto-coerce to enums. These are equivalent:

```python
from bridges.gamepad.protocol import Axis, Button

pad.set_axis(Axis.LEFT_STICK_X, 0.5)
pad.set_axis("left_stick_x", 0.5)       # same thing

pad.set_button(Button.A, True)
pad.set_button("a", True)                # same thing
```

### Available axes and buttons

**Axes:**
`left_stick_x`, `left_stick_y`, `right_stick_x`, `right_stick_y` (range: -1 to 1)
`left_trigger`, `right_trigger` (range: 0 to 1)

**Buttons:**
`a`, `b`, `x`, `y`, `lb`, `rb`, `start`, `back`, `l3`, `r3`,
`dpad_up`, `dpad_down`, `dpad_left`, `dpad_right`

### PID-controlled gamepad

The PID controller gives smooth, human-like transitions between axis values. Instead of jumping instantly to a target, it ramps smoothly.

```python
from bridges.gamepad.factory import make_pid_gamepad
from bridges.gamepad.protocol import AXIS_NAMES

pad, pid = make_pid_gamepad(backend="auto", dt=1/60)
pad.connect()

# Set desired axis targets
pid.set_targets({
    "left_stick_x": 0.3,    # steer slightly right
    "right_trigger": 0.8,   # mostly full throttle
})

# Run the control loop
current_axes = {name: 0.0 for name in AXIS_NAMES}
dt = 1 / 60

for _ in range(200):
    outputs = pid.update(current_axes)

    for axis, rate in outputs.items():
        # PID output is a rate signal — integrate it
        new_val = current_axes[axis] + rate * dt

        # Clamp to valid range
        if axis in ("left_trigger", "right_trigger"):
            new_val = max(0.0, min(1.0, new_val))
        else:
            new_val = max(-1.0, min(1.0, new_val))

        current_axes[axis] = new_val
        pad.set_axis(axis, new_val)

    pad.update()
    # time.sleep(dt)  # if running in real time

pad.disconnect()
```

The PID output is a **rate signal** (units/sec), not a position. You integrate it with `current + output * dt`. The gamepad layer clamps values to valid ranges, so overshooting the integration is safe.

### Racing example: steering + throttle/brake

```python
import time
from bridges.gamepad.factory import make_pid_gamepad
from bridges.gamepad.protocol import AXIS_NAMES

pad, pid = make_pid_gamepad(backend="auto", dt=1/60)
pad.connect()

current = {name: 0.0 for name in AXIS_NAMES}
dt = 1 / 60

# Phase 1: accelerate straight
pid.set_targets({"right_trigger": 1.0, "left_stick_x": 0.0})
for _ in range(120):
    outputs = pid.update(current)
    for axis, rate in outputs.items():
        v = current[axis] + rate * dt
        if axis in ("left_trigger", "right_trigger"):
            v = max(0.0, min(1.0, v))
        else:
            v = max(-1.0, min(1.0, v))
        current[axis] = v
        pad.set_axis(axis, v)
    pad.update()
    time.sleep(dt)

# Phase 2: brake and steer into a corner
pid.set_targets({
    "right_trigger": 0.3,
    "left_trigger": 0.5,
    "left_stick_x": -0.7,
})
for _ in range(90):
    outputs = pid.update(current)
    for axis, rate in outputs.items():
        v = current[axis] + rate * dt
        if axis in ("left_trigger", "right_trigger"):
            v = max(0.0, min(1.0, v))
        else:
            v = max(-1.0, min(1.0, v))
        current[axis] = v
        pad.set_axis(axis, v)
    pad.update()
    time.sleep(dt)

# Phase 3: straighten and accelerate out
pid.set_targets({
    "left_trigger": 0.0,
    "right_trigger": 1.0,
    "left_stick_x": 0.0,
})
for _ in range(120):
    outputs = pid.update(current)
    for axis, rate in outputs.items():
        v = current[axis] + rate * dt
        if axis in ("left_trigger", "right_trigger"):
            v = max(0.0, min(1.0, v))
        else:
            v = max(-1.0, min(1.0, v))
        current[axis] = v
        pad.set_axis(axis, v)
    pad.update()
    time.sleep(dt)

pad.disconnect()
```

### Configuring deadzone

Deadzone defaults to 0.05. Values within the deadzone snap to zero, and the remaining range is rescaled so the full output range (-1 to 1 for sticks, 0 to 1 for triggers) is still reachable.

```python
from bridges.gamepad.factory import make_gamepad

# Wider deadzone
pad = make_gamepad(backend="mock", deadzone=0.15)
pad.connect()
pad.set_axis("left_stick_x", 0.1)   # inside deadzone -> snaps to 0.0
pad.update()
print(pad.get_state().axes)          # LEFT_STICK_X == 0.0

# No deadzone
pad2 = make_gamepad(backend="mock", deadzone=0.0)
pad2.connect()
pad2.set_axis("left_stick_x", 0.01)  # no snap
pad2.update()
print(pad2.get_state().axes)          # LEFT_STICK_X == 0.01
```

The `deadzone` kwarg passes through `make_gamepad()` and `make_pid_gamepad()` to the backend constructor.

### Custom PID gains

Override the default presets per axis:

```python
from bridges.gamepad.factory import make_pid_gamepad
from bridges.gamepad.pid import PIDPreset

# Slower, smoother steering; snappier triggers
pad, pid = make_pid_gamepad(
    backend="auto",
    presets={
        "left_stick_x": PIDPreset(kp=5.0, ki=1.0, kd=0.5),
        "right_trigger": PIDPreset(kp=20.0, ki=4.0, kd=0.1),
    },
    dt=1/60,
)
```

Default presets:
- **Sticks** (`STICK_PRESET`): kp=10, ki=2, kd=0.3
- **Triggers** (`TRIGGER_PRESET`): kp=14, ki=3, kd=0.15

## Visualizer

Browser-based GUI that shows live controller state with interactive sticks, triggers, buttons, PID target indicators, and a telemetry feed.

### Running the visualizer

```bash
cd python
python bridges/gamepad/visualizer.py
```

Opens `http://localhost:8765` in your default browser. Change the port with `--port`:

```bash
python bridges/gamepad/visualizer.py --port 9000
```

The visualizer runs a `MockGamepad` with a `GamepadPIDController` at 60 Hz. It ships with several demo modes you can switch between in the UI:

- **sine_steer** — sinusoidal steering with steady throttle
- **figure_eight** — both sticks trace figure-eight patterns
- **race_lap** — simulated lap: accelerate, corner, brake zones
- **button_hold** — holds each face button for 2 seconds in sequence
- **button_sweep** — cycles through all 14 buttons rapidly
- **pid_step_response** — sharp step changes to show PID tracking behavior
- **manual_idle** — all zeros, for manual control testing

### Keyboard controls in the visualizer

You can drive the virtual controller from your keyboard while the visualizer is open. User input overrides the demo mode for the active axes.

| Key | Control |
|-----|---------|
| W / S | Left stick Y (up/down) |
| A / D | Left stick X (left/right) |
| Arrow Up / Down | Right stick Y |
| Arrow Left / Right | Right stick X |
| Left Shift | Left trigger (full press) |
| Right Shift | Right trigger (full press) |
| Space | A button |
| E | B button |
| Q | X button |
| R | Y button |
| Tab | LB bumper |
| Backslash | RB bumper |

Mouse/touch also works: drag the sticks, drag the trigger bars, click/tap buttons directly on the canvas.

### How the visualizer works

The server (`visualizer.py`) runs a WebSocket + HTTP server. The HTML page connects via WebSocket and receives state updates at 60 fps. User input (keyboard, mouse) is sent back over the same WebSocket as `{"type": "input", "axes": {...}, "buttons": {...}}` messages. The server applies user input directly (bypassing PID) for axes the user is actively controlling, then hands control back to the demo mode when the user releases.

## Running the tests

From the `python/` directory:

```bash
pytest bridges/gamepad/test_gamepad.py -v
```

The test suite covers:
- Deadzone snapping and rescaling (sticks and triggers)
- Deadzone at zero (passthrough)
- Deadzone propagation through mock backend and factory
- Mock connect/disconnect lifecycle
- Axis clamping (out-of-range values)
- Invalid axis/button name rejection
- History ring buffer overflow
- Reset zeroing
- Button press/release tracking
- All 6 axes and all 14 buttons
- PID convergence to target
- PID output clamping
- PID reset
- GamepadPIDController multi-axis convergence
- Batch target setting
- Factory backend selection (mock, auto)
- Factory make_pid_gamepad pairing
- Full integration loop (PID + mock gamepad, 100 ticks)
- PID target change tracking
- Axis/Button string-to-enum equivalence

## Architecture notes

- `GamepadBackend` is a `typing.Protocol` (structural subtyping). Any class with `connect()`, `disconnect()`, `set_axis()`, `set_button()`, `update()`, and `reset()` satisfies it.
- `MockGamepadBackend` is verified against the protocol at import time with `isinstance()`.
- `vgamepad_backend.py` import-guards the `vgamepad` dependency so the module loads cleanly on Mac. It only raises at `connect()` time if the library is missing.
- The mock backend records every `update()` call into a timestamped ring buffer (default 1000 entries). Access it with `get_history()`.
- `MockGamepad` is an alias for `MockGamepadBackend`.

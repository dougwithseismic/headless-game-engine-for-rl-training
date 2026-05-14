from __future__ import annotations

import pytest

from bridges.gamepad.protocol import Axis, Button, AXIS_NAMES, BUTTON_NAMES, apply_deadzone


# -----------------------------------------------------------------------
# Deadzone
# -----------------------------------------------------------------------

def test_deadzone_stick_snaps_to_zero():
    assert apply_deadzone(0.03, 0.05, Axis.LEFT_STICK_X) == 0.0
    assert apply_deadzone(-0.03, 0.05, Axis.LEFT_STICK_X) == 0.0

def test_deadzone_stick_rescales():
    val = apply_deadzone(0.525, 0.05, Axis.LEFT_STICK_X)
    assert abs(val - 0.5) < 0.001

def test_deadzone_stick_reaches_extremes():
    assert apply_deadzone(1.0, 0.05, Axis.LEFT_STICK_X) == 1.0
    assert apply_deadzone(-1.0, 0.05, Axis.LEFT_STICK_X) == -1.0

def test_deadzone_trigger_snaps_to_zero():
    assert apply_deadzone(0.03, 0.05, Axis.LEFT_TRIGGER) == 0.0

def test_deadzone_trigger_rescales():
    val = apply_deadzone(0.525, 0.05, Axis.LEFT_TRIGGER)
    assert abs(val - 0.5) < 0.001

def test_deadzone_trigger_reaches_one():
    assert apply_deadzone(1.0, 0.05, Axis.LEFT_TRIGGER) == 1.0

def test_deadzone_zero_means_no_deadzone():
    assert apply_deadzone(0.01, 0.0, Axis.LEFT_STICK_X) == 0.01

def test_mock_deadzone_applied():
    from bridges.gamepad.mock_backend import MockGamepad
    gp = MockGamepad(deadzone=0.1)
    gp.connect()
    gp.set_axis("left_stick_x", 0.05)
    gp.update()
    assert gp.get_state().axes[Axis.LEFT_STICK_X] == 0.0
    gp.set_axis("left_stick_x", 0.5)
    gp.update()
    assert gp.get_state().axes[Axis.LEFT_STICK_X] > 0.0
    gp.disconnect()

def test_mock_no_deadzone():
    from bridges.gamepad.mock_backend import MockGamepad
    gp = MockGamepad(deadzone=0.0)
    gp.connect()
    gp.set_axis("left_stick_x", 0.01)
    gp.update()
    assert gp.get_state().axes[Axis.LEFT_STICK_X] == 0.01
    gp.disconnect()

def test_factory_deadzone_passthrough():
    from bridges.gamepad.factory import make_gamepad
    gp = make_gamepad(backend="mock", deadzone=0.15)
    gp.connect()
    gp.set_axis("left_stick_x", 0.1)
    gp.update()
    assert gp.get_state().axes[Axis.LEFT_STICK_X] == 0.0
    gp.disconnect()


# -----------------------------------------------------------------------
# Mock basics
# -----------------------------------------------------------------------

def test_mock_connect_disconnect():
    from bridges.gamepad.mock_backend import MockGamepad

    gp = MockGamepad()
    gp.connect()
    gp.set_axis("left_stick_x", 0.5)
    gp.update()
    gp.disconnect()

    with pytest.raises(RuntimeError):
        gp.set_axis("left_stick_x", 0.0)


def test_mock_axis_clamping():
    from bridges.gamepad.mock_backend import MockGamepad

    gp = MockGamepad()
    gp.connect()

    gp.set_axis("left_stick_x", 2.0)
    gp.update()
    state = gp.get_state()
    assert state.axes[Axis.LEFT_STICK_X] == 1.0

    gp.set_axis("left_trigger", -0.5)
    gp.update()
    state = gp.get_state()
    assert state.axes[Axis.LEFT_TRIGGER] == 0.0

    gp.disconnect()


def test_mock_invalid_axis():
    from bridges.gamepad.mock_backend import MockGamepad

    gp = MockGamepad()
    gp.connect()
    with pytest.raises(ValueError):
        gp.set_axis("nonexistent_axis", 0.0)
    gp.disconnect()


def test_mock_invalid_button():
    from bridges.gamepad.mock_backend import MockGamepad

    gp = MockGamepad()
    gp.connect()
    with pytest.raises(ValueError):
        gp.set_button("nonexistent_button", True)
    gp.disconnect()


def test_mock_history():
    from bridges.gamepad.mock_backend import MockGamepad

    gp = MockGamepad(max_history=5)
    gp.connect()

    for i in range(7):
        gp.set_axis("left_stick_x", i * 0.1)
        gp.update()

    history = gp.get_history()
    assert len(history) == 5
    gp.disconnect()


def test_mock_reset():
    from bridges.gamepad.mock_backend import MockGamepad

    gp = MockGamepad()
    gp.connect()
    gp.set_axis("left_stick_x", 0.8)
    gp.set_button("a", True)
    gp.update()

    gp.reset()
    state = gp.get_state()
    assert state.axes[Axis.LEFT_STICK_X] == 0.0
    assert state.buttons[Button.A] is False
    gp.disconnect()


def test_mock_buttons():
    from bridges.gamepad.mock_backend import MockGamepad

    gp = MockGamepad()
    gp.connect()

    gp.set_button("a", True)
    gp.set_button("b", True)
    gp.update()
    state = gp.get_state()
    assert state.buttons[Button.A] is True
    assert state.buttons[Button.B] is True

    gp.set_button("a", False)
    gp.update()
    state = gp.get_state()
    assert state.buttons[Button.A] is False
    assert state.buttons[Button.B] is True
    gp.disconnect()


def test_mock_all_axes():
    from bridges.gamepad.mock_backend import MockGamepad

    gp = MockGamepad()
    gp.connect()
    for axis in AXIS_NAMES:
        gp.set_axis(axis, 0.5)
    gp.update()
    gp.disconnect()


def test_mock_all_buttons():
    from bridges.gamepad.mock_backend import MockGamepad

    gp = MockGamepad()
    gp.connect()
    for btn in BUTTON_NAMES:
        gp.set_button(btn, True)
    gp.update()
    state = gp.get_state()
    for btn in Button:
        assert state.buttons[btn] is True
    gp.disconnect()


def test_pid_controller_converges():
    from bridges.gamepad.pid import PIDController

    pid = PIDController(kp=4.0, ki=0.5, kd=0.2, output_min=-1.0, output_max=1.0, dt=0.016)
    current = 0.0
    target = 0.8

    for _ in range(200):
        output = pid.update(target, current)
        current += output * 0.016

    assert abs(current - target) < 0.05


def test_pid_controller_clamp():
    from bridges.gamepad.pid import PIDController

    pid = PIDController(kp=100.0, ki=0.0, kd=0.0, output_min=-1.0, output_max=1.0, dt=0.016)
    output = pid.update(target=1.0, current=0.0)
    assert output == 1.0


def test_pid_controller_reset():
    from bridges.gamepad.pid import PIDController

    pid = PIDController(kp=4.0, ki=0.5, kd=0.2, output_min=-1.0, output_max=1.0, dt=0.016)
    pid.update(1.0, 0.0)
    pid.update(1.0, 0.3)
    pid.reset()
    output = pid.update(0.0, 0.0)
    assert output == 0.0


def test_gamepad_pid_controller():
    from bridges.gamepad.pid import GamepadPIDController

    ctrl = GamepadPIDController(dt=0.016)
    ctrl.set_target("left_stick_x", 0.7)
    ctrl.set_target("right_trigger", 0.9)

    current = {"left_stick_x": 0.0, "right_trigger": 0.0}
    for _ in range(200):
        outputs = ctrl.update(current)
        for k in current:
            current[k] = max(-1.0, min(1.0, current[k] + outputs.get(k, 0.0) * 0.016))

    assert abs(current["left_stick_x"] - 0.7) < 0.1
    assert abs(current["right_trigger"] - 0.9) < 0.1


def test_gamepad_pid_set_targets_batch():
    from bridges.gamepad.pid import GamepadPIDController

    ctrl = GamepadPIDController(dt=0.016)
    ctrl.set_targets({
        "left_stick_x": 0.5,
        "left_stick_y": -0.3,
        "right_trigger": 1.0,
    })
    targets = ctrl.get_targets()
    assert targets["left_stick_x"] == 0.5
    assert targets["left_stick_y"] == -0.3
    assert targets["right_trigger"] == 1.0


def test_factory_returns_mock():
    from bridges.gamepad.factory import make_gamepad

    gp = make_gamepad(backend="mock")
    gp.connect()
    gp.set_axis("left_stick_x", 0.5)
    gp.update()
    gp.disconnect()


def test_factory_auto():
    from bridges.gamepad.factory import make_gamepad

    gp = make_gamepad(backend="auto")
    gp.connect()
    gp.set_axis("right_trigger", 0.8)
    gp.update()
    gp.disconnect()


def test_factory_make_pid_gamepad():
    from bridges.gamepad.factory import make_pid_gamepad

    gp, pid = make_pid_gamepad(backend="mock")
    gp.connect()
    pid.set_target("left_stick_x", 0.5)
    current = {"left_stick_x": 0.0}
    outputs = pid.update(current)
    assert "left_stick_x" in outputs
    gp.disconnect()


def test_full_integration_loop():
    """Simulate a racing control loop: PID smoothly steers to target while throttling."""
    from bridges.gamepad.factory import make_pid_gamepad

    gp, pid = make_pid_gamepad(backend="mock", dt=0.016)
    gp.connect()

    pid.set_targets({
        "left_stick_x": 0.3,
        "right_trigger": 0.8,
    })

    current_axes = {name: 0.0 for name in AXIS_NAMES}

    for _ in range(100):
        outputs = pid.update(current_axes)
        for axis, val in outputs.items():
            new_val = current_axes[axis] + val * 0.016
            if axis in ("left_trigger", "right_trigger"):
                new_val = max(0.0, min(1.0, new_val))
            else:
                new_val = max(-1.0, min(1.0, new_val))
            current_axes[axis] = new_val
            gp.set_axis(axis, new_val)
        gp.update()

    state = gp.get_state()
    assert abs(state.axes[Axis.LEFT_STICK_X] - 0.3) < 0.15
    assert abs(state.axes[Axis.RIGHT_TRIGGER] - 0.8) < 0.15

    history = gp.get_history()
    assert len(history) == 100

    gp.disconnect()


def test_pid_steering_then_change_target():
    """PID should track a target change smoothly."""
    from bridges.gamepad.pid import GamepadPIDController

    ctrl = GamepadPIDController(dt=0.016)
    ctrl.set_target("left_stick_x", 0.5)

    current = {"left_stick_x": 0.0}
    for _ in range(100):
        outputs = ctrl.update(current)
        current["left_stick_x"] = max(-1.0, min(1.0, current["left_stick_x"] + outputs["left_stick_x"] * 0.016))

    assert abs(current["left_stick_x"] - 0.5) < 0.1

    ctrl.set_target("left_stick_x", -0.5)
    for _ in range(200):
        outputs = ctrl.update(current)
        current["left_stick_x"] = max(-1.0, min(1.0, current["left_stick_x"] + outputs["left_stick_x"] * 0.016))

    assert abs(current["left_stick_x"] - (-0.5)) < 0.1


def test_enum_string_equivalence():
    """Axis/Button str enums work as plain strings for dict access."""
    assert Axis.LEFT_STICK_X == "left_stick_x"
    assert Button.A == "a"
    assert Axis("left_stick_x") == Axis.LEFT_STICK_X
    assert Button("a") == Button.A

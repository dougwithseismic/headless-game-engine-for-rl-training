"""Tests for ActionSink implementations."""

import numpy as np
import pytest

from bridges.core.action_sink import ActionSink
from bridges.sinks.mock_sink import MockActionSink
from bridges.sinks.gamepad_sink import GamepadActionSink
from bridges.gamepad.protocol import Axis, Button


# ---------------------------------------------------------------------------
# GamepadActionSink
# ---------------------------------------------------------------------------


def test_gamepad_sink_satisfies_protocol():
    sink = GamepadActionSink(
        backend="mock",
        axis_mapping={0: Axis.LEFT_STICK_X},
    )
    assert isinstance(sink, ActionSink)


def test_gamepad_sink_axis_mapping():
    sink = GamepadActionSink(
        backend="mock",
        axis_mapping={0: Axis.LEFT_STICK_X, 1: Axis.LEFT_STICK_Y},
    )
    sink.connect()
    sink.send(np.array([0.5, -0.3]))
    state = sink._gamepad.get_state()
    assert abs(state.axes[Axis.LEFT_STICK_X] - 0.5) < 0.1
    assert abs(state.axes[Axis.LEFT_STICK_Y] - (-0.3)) < 0.1


def test_gamepad_sink_combined_axes_positive():
    sink = GamepadActionSink(
        backend="mock",
        combined_axes={0: (Axis.RIGHT_TRIGGER, Axis.LEFT_TRIGGER)},
    )
    sink.connect()
    sink.send(np.array([0.8]))
    state = sink._gamepad.get_state()
    assert state.axes[Axis.RIGHT_TRIGGER] > 0.5
    assert state.axes[Axis.LEFT_TRIGGER] == 0.0


def test_gamepad_sink_combined_axes_negative():
    sink = GamepadActionSink(
        backend="mock",
        combined_axes={0: (Axis.RIGHT_TRIGGER, Axis.LEFT_TRIGGER)},
    )
    sink.connect()
    sink.send(np.array([-0.6]))
    state = sink._gamepad.get_state()
    assert state.axes[Axis.RIGHT_TRIGGER] == 0.0
    assert state.axes[Axis.LEFT_TRIGGER] > 0.3


def test_gamepad_sink_button_mapping():
    sink = GamepadActionSink(
        backend="mock",
        button_mapping={0: Button.A, 1: Button.B},
    )
    sink.connect()
    sink.send(np.array([0.8, 0.2]))
    state = sink._gamepad.get_state()
    assert state.buttons[Button.A] is True
    assert state.buttons[Button.B] is False


def test_gamepad_sink_racing_config():
    """Simulate the Assetto Corsa mapping: steering + combined accel/brake."""
    sink = GamepadActionSink(
        backend="mock",
        axis_mapping={0: Axis.LEFT_STICK_X},
        combined_axes={1: (Axis.RIGHT_TRIGGER, Axis.LEFT_TRIGGER)},
    )
    sink.connect()

    sink.send(np.array([0.3, 0.9]))
    state = sink._gamepad.get_state()
    assert abs(state.axes[Axis.LEFT_STICK_X] - 0.3) < 0.1
    assert state.axes[Axis.RIGHT_TRIGGER] > 0.5
    assert state.axes[Axis.LEFT_TRIGGER] == 0.0

    sink.send(np.array([-0.5, -0.7]))
    state = sink._gamepad.get_state()
    assert state.axes[Axis.LEFT_STICK_X] < -0.3
    assert state.axes[Axis.RIGHT_TRIGGER] == 0.0
    assert state.axes[Axis.LEFT_TRIGGER] > 0.3


def test_gamepad_sink_reset():
    sink = GamepadActionSink(
        backend="mock",
        axis_mapping={0: Axis.LEFT_STICK_X},
    )
    sink.connect()
    sink.send(np.array([0.9]))
    sink.reset()
    state = sink._gamepad.get_state()
    assert state.axes[Axis.LEFT_STICK_X] == 0.0


def test_gamepad_sink_disconnect():
    sink = GamepadActionSink(backend="mock", axis_mapping={0: Axis.LEFT_STICK_X})
    sink.connect()
    assert sink._gamepad is not None
    sink.disconnect()
    assert sink._gamepad is None


def test_gamepad_sink_info():
    sink = GamepadActionSink(backend="mock", axis_mapping={0: Axis.LEFT_STICK_X})
    info = sink.info()
    assert "gamepad" in info.name
    assert info.supports_continuous is True
    assert info.platform == "any"

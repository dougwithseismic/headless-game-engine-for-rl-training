"""PID controller module for smooth gamepad axis control.

Provides cruise-control-like behaviour for virtual gamepad axes: set a target
position (e.g. "steer to 0.3") and the PID controller smoothly ramps the
actual output, preventing jitter and enabling human-like input curves.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Preset
# ---------------------------------------------------------------------------

@dataclass
class PIDPreset:
    """Compact carrier for PID gain parameters."""

    kp: float
    ki: float
    kd: float


STICK_PRESET = PIDPreset(kp=10.0, ki=2.0, kd=0.3)
TRIGGER_PRESET = PIDPreset(kp=14.0, ki=3.0, kd=0.15)

# Axis name -> default preset mapping
DEFAULT_PRESETS: dict[str, PIDPreset] = {
    "left_stick_x": STICK_PRESET,
    "left_stick_y": STICK_PRESET,
    "right_stick_x": STICK_PRESET,
    "right_stick_y": STICK_PRESET,
    "left_trigger": TRIGGER_PRESET,
    "right_trigger": TRIGGER_PRESET,
}

# Output ranges per axis type
_STICK_AXES = {"left_stick_x", "left_stick_y", "right_stick_x", "right_stick_y"}
_TRIGGER_AXES = {"left_trigger", "right_trigger"}


# ---------------------------------------------------------------------------
# Core PID
# ---------------------------------------------------------------------------

class PIDController:
    """Single-axis PID controller with anti-windup and output clamping."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_min: float,
        output_max: float,
        integral_max: float | None = None,
        dt: float | None = None,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_max = integral_max
        self._fixed_dt = dt

        # Internal state
        self._integral: float = 0.0
        self._prev_error: float | None = None
        self._last_time: float | None = None

    # ------------------------------------------------------------------

    def update(self, target: float, current: float) -> float:
        """Compute PID output given the desired *target* and *current* value."""

        # --- Determine dt ---------------------------------------------------
        if self._fixed_dt is not None:
            dt = self._fixed_dt
        else:
            now = time.monotonic()
            if self._last_time is None:
                dt = 0.0  # first call: no delta available
            else:
                dt = now - self._last_time
            self._last_time = now

        error = target - current

        # --- Proportional ----------------------------------------------------
        p_term = self.kp * error

        # --- Integral --------------------------------------------------------
        self._integral += error * dt
        if self.integral_max is not None:
            self._integral = max(-self.integral_max, min(self.integral_max, self._integral))
        i_term = self.ki * self._integral

        # --- Derivative ------------------------------------------------------
        if self._prev_error is None or dt == 0.0:
            d_term = 0.0
        else:
            d_term = self.kd * (error - self._prev_error) / dt
        self._prev_error = error

        # --- Sum & clamp -----------------------------------------------------
        output = p_term + i_term + d_term
        return max(self.output_min, min(self.output_max, output))

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Zero all internal state so the controller starts fresh."""
        self._integral = 0.0
        self._prev_error = None
        self._last_time = None


# ---------------------------------------------------------------------------
# Multi-axis gamepad wrapper
# ---------------------------------------------------------------------------

class GamepadPIDController:
    """Wraps one :class:`PIDController` per gamepad axis.

    Parameters
    ----------
    presets:
        Optional per-axis preset overrides.  Any axis not listed falls back
        to :data:`DEFAULT_PRESETS`.
    dt:
        Fixed timestep forwarded to every underlying PID controller.
        ``None`` means wall-clock delta.
    """

    def __init__(
        self,
        presets: dict[str, PIDPreset] | None = None,
        dt: float | None = None,
    ) -> None:
        merged = {**DEFAULT_PRESETS, **(presets or {})}

        self._controllers: dict[str, PIDController] = {}
        self._targets: dict[str, float] = {}

        for axis, preset in merged.items():
            # Output range is a rate (units/sec), not a position. Wide range
            # lets the PID ramp fast; the axis value itself is clamped to
            # [-1,1] or [0,1] at the gamepad layer.
            if axis in _STICK_AXES:
                out_min, out_max = -120.0, 120.0
            elif axis in _TRIGGER_AXES:
                out_min, out_max = -120.0, 120.0
            else:
                out_min, out_max = -120.0, 120.0

            self._controllers[axis] = PIDController(
                kp=preset.kp,
                ki=preset.ki,
                kd=preset.kd,
                output_min=out_min,
                output_max=out_max,
                dt=dt,
            )
            self._targets[axis] = 0.0

    # ------------------------------------------------------------------
    # Target setters
    # ------------------------------------------------------------------

    def set_target(self, axis: str, target: float) -> None:
        """Set the desired value for a single axis."""
        if axis not in self._controllers:
            raise KeyError(f"Unknown axis '{axis}'. Known: {sorted(self._controllers)}")
        self._targets[axis] = target

    def set_targets(self, targets: dict[str, float]) -> None:
        """Batch-set desired values for multiple axes."""
        for axis, value in targets.items():
            self.set_target(axis, value)

    def get_targets(self) -> dict[str, float]:
        """Return a copy of the current target values."""
        return dict(self._targets)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, current_state: dict[str, float]) -> dict[str, float]:
        """Compute PID outputs for every axis.

        Parameters
        ----------
        current_state:
            Mapping of axis name to its current reading.

        Returns
        -------
        dict[str, float]
            Mapping of axis name to the new output value to send to the
            virtual gamepad.
        """
        outputs: dict[str, float] = {}
        for axis, controller in self._controllers.items():
            current = current_state.get(axis, 0.0)
            target = self._targets.get(axis, 0.0)
            outputs[axis] = controller.update(target, current)
        return outputs

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset every PID controller and zero all targets."""
        for axis, controller in self._controllers.items():
            controller.reset()
            self._targets[axis] = 0.0

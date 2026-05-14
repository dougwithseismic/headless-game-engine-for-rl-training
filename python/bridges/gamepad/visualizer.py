"""Gamepad visualizer — serves a live Xbox controller GUI in the browser.

Usage:
    python bridges/gamepad/visualizer.py [--port 8765]

Opens a browser window showing the gamepad state in real time. Includes
demo modes that exercise the PID controller with different patterns.
"""
from __future__ import annotations

import asyncio
import json
import math
import sys
import webbrowser
from http import HTTPStatus
from pathlib import Path

# Ensure the python/ directory is on sys.path so `bridges.*` imports work
# regardless of where this script is invoked from.
_PYTHON_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

import websockets
from websockets.http11 import Request, Response

from bridges.gamepad.mock_backend import MockGamepad
from bridges.gamepad.pid import GamepadPIDController, PIDPreset
from bridges.gamepad.protocol import Axis, Button

HTML_PATH = Path(__file__).parent / "visualizer.html"
SVG_PATH = Path(__file__).parent / "controller.svg"

# ---------------------------------------------------------------------------
# Demo modes — each is an async generator yielding (targets, buttons) per tick
# ---------------------------------------------------------------------------

MODES = [
    "sine_steer",
    "figure_eight",
    "race_lap",
    "button_hold",
    "button_sweep",
    "pid_step_response",
    "manual_idle",
]


async def demo_sine_steer(tick: int):
    """Smooth sinusoidal steering with steady throttle."""
    t = tick * 0.03
    return {
        "left_stick_x": math.sin(t),
        "right_trigger": 0.6 + 0.2 * math.sin(t * 0.5),
    }, {}


async def demo_figure_eight(tick: int):
    """Both sticks trace a figure-eight pattern."""
    t = tick * 0.04
    return {
        "left_stick_x": math.sin(t),
        "left_stick_y": math.sin(t * 2) * 0.5,
        "right_stick_x": math.cos(t),
        "right_stick_y": math.cos(t * 2) * 0.5,
        "right_trigger": 0.5 + 0.5 * math.sin(t * 0.3),
    }, {}


async def demo_race_lap(tick: int):
    """Simulates a lap: accelerate, steer through corners, brake zones."""
    t = tick % 400
    targets = {}
    buttons = {}

    if t < 60:
        targets["right_trigger"] = min(1.0, t / 30.0)
        targets["left_stick_x"] = 0.0
    elif t < 100:
        p = (t - 60) / 40.0
        targets["left_stick_x"] = math.sin(p * math.pi) * 0.7
        targets["right_trigger"] = 0.4
        targets["left_trigger"] = 0.3 * math.sin(p * math.pi * 0.5)
    elif t < 160:
        targets["right_trigger"] = min(1.0, (t - 100) / 30.0)
        targets["left_stick_x"] = 0.0
    elif t < 220:
        p = (t - 160) / 60.0
        targets["left_stick_x"] = -math.sin(p * math.pi) * 0.9
        targets["left_trigger"] = 0.5 * math.sin(p * math.pi * 0.3)
        targets["right_trigger"] = 0.3
    elif t < 300:
        targets["right_trigger"] = 0.8
        targets["left_stick_x"] = math.sin((t - 220) * 0.05) * 0.3
    else:
        targets["left_trigger"] = min(0.8, (t - 300) / 40.0)
        targets["right_trigger"] = max(0.0, 1.0 - (t - 300) / 30.0)
        targets["left_stick_x"] = 0.0
        if t > 380:
            buttons["b"] = True

    return targets, buttons


async def demo_button_hold(tick: int):
    """Holds buttons for extended periods — A held 2s, then B held 2s, etc."""
    phase_len = 120  # 2 seconds at 60Hz
    btns = ["a", "b", "x", "y", "lb", "rb"]
    phase = (tick // phase_len) % (len(btns) + 1)  # +1 for a gap between cycles
    active = {}
    if phase < len(btns):
        active[btns[phase]] = True
        # Also hold triggers during certain buttons
        if btns[phase] in ("lb", "rb"):
            trig = "left_trigger" if btns[phase] == "lb" else "right_trigger"
            return {trig: 0.8, "left_stick_x": 0.0}, active
    progress = (tick % phase_len) / phase_len
    return {"right_trigger": 0.3 * progress}, active


async def demo_button_sweep(tick: int):
    """Cycles through all buttons, lighting each one for a moment."""
    btns = ["a", "b", "x", "y", "lb", "rb", "dpad_up", "dpad_right",
            "dpad_down", "dpad_left", "start", "back", "l3", "r3"]
    cycle = (tick // 15) % len(btns)
    active = {}
    for i, b in enumerate(btns):
        active[b] = (i == cycle)
    return {
        "left_stick_x": math.sin(tick * 0.05) * 0.3,
        "right_stick_y": math.cos(tick * 0.05) * 0.3,
    }, active


async def demo_pid_step_response(tick: int):
    """Sharp step changes to show PID tracking behavior."""
    phase = (tick // 80) % 6
    steps = [
        {"left_stick_x": 0.8},
        {"left_stick_x": -0.8},
        {"left_stick_x": 0.0, "right_trigger": 1.0},
        {"left_stick_x": 0.5, "right_trigger": 0.0, "left_trigger": 0.7},
        {"left_stick_x": -0.3, "left_trigger": 0.0, "right_trigger": 0.5},
        {"left_stick_x": 0.0, "right_trigger": 0.0},
    ]
    return steps[phase], {}


async def demo_manual_idle(tick: int):
    """All zeros — idle state for manual control testing."""
    return {}, {}


DEMO_FNS = {
    "sine_steer": demo_sine_steer,
    "figure_eight": demo_figure_eight,
    "race_lap": demo_race_lap,
    "button_hold": demo_button_hold,
    "button_sweep": demo_button_sweep,
    "pid_step_response": demo_pid_step_response,
    "manual_idle": demo_manual_idle,
}

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class GamepadServer:
    def __init__(self, port: int = 8765):
        self.port = port
        self.gamepad = MockGamepad()
        self.pid = GamepadPIDController(dt=1 / 60)
        self.mode = "sine_steer"
        self.tick = 0
        self.clients: set[websockets.WebSocketServerProtocol] = set()
        self._running = False
        # User input override — axes/buttons set directly by the browser
        self._user_axes: dict[str, float] = {}
        self._user_held_buttons: set[str] = set()  # persistent until release
        self._user_active_axes: set[str] = set()

    async def handler(self, ws: websockets.WebSocketServerProtocol):
        self.clients.add(ws)
        try:
            await ws.send(json.dumps({
                "type": "modes",
                "modes": MODES,
                "current": self.mode,
            }))
            async for msg in ws:
                try:
                    data = json.loads(msg)
                    if data.get("type") == "set_mode":
                        self.mode = data["mode"]
                        self.pid.reset()
                        self.gamepad.reset()
                        self._user_axes.clear()
                        self._user_held_buttons.clear()
                        self._user_active_axes.clear()
                        self.tick = 0
                        broadcast = json.dumps({
                            "type": "modes",
                            "modes": MODES,
                            "current": self.mode,
                        })
                        await asyncio.gather(
                            *[c.send(broadcast) for c in self.clients],
                            return_exceptions=True,
                        )
                    elif data.get("type") == "input":
                        for axis, val in data.get("axes", {}).items():
                            self._user_axes[axis] = val
                            if abs(val) > 0.001:
                                self._user_active_axes.add(axis)
                            else:
                                self._user_active_axes.discard(axis)
                        for btn, pressed in data.get("buttons", {}).items():
                            if pressed:
                                self._user_held_buttons.add(btn)
                            else:
                                self._user_held_buttons.discard(btn)
                except (json.JSONDecodeError, KeyError):
                    pass
        finally:
            self.clients.discard(ws)

    async def process_request(self, connection, request: Request):
        if request.path in ("/", "/index.html"):
            body = HTML_PATH.read_bytes()
            return Response(HTTPStatus.OK, "OK", websockets.Headers({
                "Content-Type": "text/html; charset=utf-8",
                "Content-Length": str(len(body)),
            }), body)

        if request.path == "/controller.svg":
            body = SVG_PATH.read_bytes()
            return Response(HTTPStatus.OK, "OK", websockets.Headers({
                "Content-Type": "image/svg+xml",
                "Content-Length": str(len(body)),
                "Cache-Control": "public, max-age=3600",
            }), body)

        if request.path == "/ws":
            return None

        return Response(HTTPStatus.NOT_FOUND, "Not Found", websockets.Headers(), b"404")

    async def tick_loop(self):
        self.gamepad.connect()
        self._running = True

        while self._running:
            demo_fn = DEMO_FNS.get(self.mode, demo_manual_idle)
            targets, buttons = await demo_fn(self.tick)

            if targets:
                self.pid.set_targets(targets)

            current_axes = {
                a.value: self.gamepad._state.axes[a] for a in Axis
            }

            # PID drives axes NOT actively controlled by the user
            pid_outputs = self.pid.update(current_axes)
            dt = 1 / 60
            for axis_name, rate in pid_outputs.items():
                if axis_name in self._user_active_axes:
                    continue
                cur = current_axes.get(axis_name, 0.0)
                new_val = cur + rate * dt
                if axis_name in ("left_trigger", "right_trigger"):
                    new_val = max(0.0, min(1.0, new_val))
                else:
                    new_val = max(-1.0, min(1.0, new_val))
                self.gamepad.set_axis(axis_name, new_val)

            # User input overrides — applied directly, no PID
            for axis_name, val in self._user_axes.items():
                self.gamepad.set_axis(axis_name, val)
            self._user_axes.clear()

            # Buttons: user holds persist, demo fills the rest
            for b in Button:
                btn_name = b.value
                if btn_name in self._user_held_buttons:
                    self.gamepad.set_button(b, True)
                elif btn_name in buttons:
                    self.gamepad.set_button(b, buttons[btn_name])
                else:
                    self.gamepad.set_button(b, False)

            self.gamepad.update()
            self.tick += 1

            gp_state = self.gamepad.get_state()
            msg = json.dumps({
                "type": "state",
                "axes": {a.value: round(v, 5) for a, v in gp_state.axes.items()},
                "buttons": {b.value: v for b, v in gp_state.buttons.items()},
                "pid_targets": {k: round(v, 5) for k, v in self.pid.get_targets().items()},
                "mode": self.mode,
                "tick": self.tick,
            })

            if self.clients:
                await asyncio.gather(
                    *[c.send(msg) for c in self.clients],
                    return_exceptions=True,
                )

            await asyncio.sleep(1 / 60)

    async def run(self):
        server = await websockets.serve(
            self.handler,
            "localhost",
            self.port,
            process_request=self.process_request,
        )
        url = f"http://localhost:{self.port}"
        print(f"Gamepad visualizer running at {url}")
        print(f"Modes: {', '.join(MODES)}")
        print("Press Ctrl+C to stop.\n")
        webbrowser.open(url)

        tick_task = asyncio.create_task(self.tick_loop())
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            tick_task.cancel()
            server.close()
            await server.wait_closed()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Gamepad visualizer")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = GamepadServer(port=args.port)
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

"""
Live training viewer — runs a spectator env at real-time speed,
loads the latest model snapshot periodically, streams telemetry to WebSocket.

The training loop saves snapshots via LiveSnapshotCallback.
The spectator thread picks them up and plays at 64 tps.
The React viewer connects to ws://localhost:<port>/ws/observe.
"""

import asyncio
import collections
import json
import os
import sys
import threading
import time

from aiohttp import web
from stable_baselines3.common.callbacks import BaseCallback

from model_utils import load_model as _load_model

sys.path.insert(0, os.path.dirname(__file__))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TelemetryBridge:
    """Thread-safe ring buffer. Spectator writes, WS server reads."""

    def __init__(self, max_size=2000):
        self.buffer = collections.deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.latest_round_start = None

    def append(self, event_json: str):
        with self.lock:
            self.buffer.append(event_json)
            if '"RoundStart"' in event_json:
                self.latest_round_start = event_json

    def drain(self, max_events=8):
        with self.lock:
            batch = []
            for _ in range(min(max_events, len(self.buffer))):
                batch.append(self.buffer.popleft())
            return batch


class LiveSnapshotCallback(BaseCallback):
    """Saves a model snapshot every N steps for the spectator to pick up."""

    def __init__(self, snapshot_path, interval=50_000, verbose=0):
        super().__init__(verbose)
        self.snapshot_path = snapshot_path
        self.interval = interval
        self.last_save = 0

    def _on_step(self):
        if self.num_timesteps - self.last_save >= self.interval:
            self.model.save(self.snapshot_path)
            self.last_save = self.num_timesteps
        return True


def _run_spectator(bridge, config_path, scenario, frame_skip, max_steps, snapshot_path):
    import numpy as np
    from selfplay_gym import SelfPlayGym

    env = SelfPlayGym(config_path, scenario, frame_skip, max_steps)
    env.telemetry_sink = bridge
    tick_interval = 1.0 / 64
    model = None
    model_mtime = 0
    is_recurrent = False
    lstm_states = None

    while True:
        snap = snapshot_path + ".zip"
        if os.path.exists(snap):
            try:
                mt = os.path.getmtime(snap)
                if mt > model_mtime:
                    model, is_recurrent = _load_model(snapshot_path)
                    model_mtime = mt
                    lstm_states = None
            except Exception:
                pass

        if model is None:
            time.sleep(1)
            continue

        obs, _ = env.reset()
        lstm_states = None
        episode_start = True
        for _ in range(max_steps):
            t0 = time.perf_counter()
            if is_recurrent:
                action, lstm_states = model.predict(
                    obs, state=lstm_states,
                    episode_start=np.array([episode_start]),
                    deterministic=False,
                )
                episode_start = False
            else:
                action, _ = model.predict(obs, deterministic=False)
            obs, r, terminated, truncated, info = env.step(action)
            elapsed = time.perf_counter() - t0
            sleep_time = tick_interval * frame_skip - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)
            if terminated or truncated:
                break


async def ws_observe(request):
    ws = web.WebSocketResponse(heartbeat=10.0)
    await ws.prepare(request)
    bridge = request.app["bridge"]

    if bridge.latest_round_start:
        await ws.send_str(bridge.latest_round_start)

    try:
        while not ws.closed:
            events = bridge.drain(max_events=16)
            for event_json in events:
                await ws.send_str(event_json)
            await asyncio.sleep(1.0 / 32)
    except (asyncio.CancelledError, ConnectionResetError):
        pass
    return ws


async def api_health(request):
    return web.json_response({"status": "ok"})


async def api_config(request):
    return web.json_response(request.app["config"])


async def api_match(request):
    return web.json_response({
        "title": "training-live",
        "tick": 0,
        "tick_rate": 64,
        "status": "training",
    })


async def api_obstacles(request):
    bridge = request.app["bridge"]
    if bridge.latest_round_start:
        data = json.loads(bridge.latest_round_start)
        return web.json_response({
            "obstacles": data.get("obstacles", []),
            "spawn_points": data.get("spawn_points", []),
        })
    return web.json_response({"obstacles": [], "spawn_points": []})


def _run_server(bridge, config, port):
    app = web.Application()
    app["bridge"] = bridge
    app["config"] = config

    app.router.add_get("/api/health", api_health)
    app.router.add_get("/api/config", api_config)
    app.router.add_get("/api/match", api_match)
    app.router.add_get("/api/obstacles", api_obstacles)
    app.router.add_get("/ws/observe", ws_observe)
    app.router.add_static("/", os.path.join(PROJECT_ROOT, "web"), show_index=True)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "0.0.0.0", port)
    loop.run_until_complete(site.start())
    loop.run_forever()


def start_live_view(config_path, scenario, frame_skip, max_steps, snapshot_path, port=3000):
    with open(config_path) as f:
        config = json.load(f)

    bridge = TelemetryBridge()

    server_thread = threading.Thread(
        target=_run_server,
        args=(bridge, config, port),
        daemon=True,
    )
    server_thread.start()

    spectator_thread = threading.Thread(
        target=_run_spectator,
        args=(bridge, config_path, scenario, frame_skip, max_steps, snapshot_path),
        daemon=True,
    )
    spectator_thread.start()

    print(f"  [live-view] Spectator on http://localhost:{port} (updates from {snapshot_path})")
    return bridge

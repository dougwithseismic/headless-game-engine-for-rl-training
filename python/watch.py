"""
Watch a trained GhostLobby agent fight in the browser.

Runs the game env at real-time speed with a trained model controlling agent 0,
serves the web viewer, and broadcasts telemetry over WebSocket.

Usage:
    python python/watch.py --model runs/best_model/best_model.zip
    python python/watch.py --model runs/final_model.zip --vec-normalize runs/vec_normalize.pkl
"""

import argparse
import asyncio
import copy
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from aiohttp import web
import ghostlobby

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_config(path):
    return os.path.join(PROJECT_ROOT, path) if not os.path.isabs(path) else path


def parse_args():
    p = argparse.ArgumentParser(description="Watch a trained GhostLobby agent")
    p.add_argument("--model", required=True, help="Path to model .zip file")
    p.add_argument("--vec-normalize", default=None, help="Path to vec_normalize.pkl for observation normalization")
    p.add_argument("--config", default="configs/1v1_deathmatch.json")
    p.add_argument("--scenario", default="fps")
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--port", type=int, default=3000)
    p.add_argument("--stochastic", action="store_true", help="Use stochastic policy instead of deterministic")
    return p.parse_args()


def flatten_obs(agent_obs):
    parts = []
    for key in sorted(agent_obs.keys()):
        parts.extend(agent_obs[key])
    return np.array(parts, dtype=np.float32)


class ObsNormalizer:
    """Applies VecNormalize observation stats to raw observations."""
    def __init__(self, vec_normalize_path):
        import pickle
        with open(vec_normalize_path, "rb") as f:
            data = pickle.load(f)
        self.obs_rms = data.obs_rms
        self.clip_obs = data.clip_obs
        self.epsilon = data.epsilon

    def normalize(self, obs):
        return np.clip(
            (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
            -self.clip_obs, self.clip_obs
        ).astype(np.float32)


class GameLoop:
    def __init__(self, model, config_path, scenario, config, frame_skip,
                 deterministic=True, obs_normalizer=None):
        self.model = model
        self.config_path = config_path
        self.scenario = scenario
        self.config = config
        self.frame_skip = frame_skip
        self.deterministic = deterministic
        self.obs_normalizer = obs_normalizer
        self.ws_clients = set()
        self.running = False
        self.tick = 0

    def make_env(self):
        env = ghostlobby.GhostLobbyEnv(self.config_path, scenario=self.scenario)
        raw_obs, _ = env.reset()
        return env, raw_obs

    async def broadcast(self, msg):
        dead = set()
        for ws in self.ws_clients:
            try:
                await ws.send_str(msg)
            except Exception:
                dead.add(ws)
        self.ws_clients -= dead

    async def run(self):
        self.running = True
        tick_interval = 1.0 / self.config["tick_rate"]

        env, raw_obs = self.make_env()
        obs = flatten_obs(raw_obs[0])
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        episode_ticks = 0

        while self.running:
            t0 = time.perf_counter()

            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            action_list = action.tolist() if hasattr(action, "tolist") else list(action)

            for _ in range(self.frame_skip):
                raw_obs, rewards, terms, truncs, infos = env.step({0: action_list})

                for event_json in env.drain_telemetry():
                    await self.broadcast(event_json)

                self.tick += 1
                episode_ticks += 1

                elapsed = time.perf_counter() - t0
                sleep_time = tick_interval - (elapsed % tick_interval)
                if sleep_time > 0.001:
                    await asyncio.sleep(sleep_time)

            obs = flatten_obs(raw_obs[0])
            if self.obs_normalizer:
                obs = self.obs_normalizer.normalize(obs)

            if terms.get(0, False) or episode_ticks >= 8192:
                env, raw_obs = self.make_env()
                obs = flatten_obs(raw_obs[0])
                if self.obs_normalizer:
                    obs = self.obs_normalizer.normalize(obs)
                episode_ticks = 0


async def ws_observe(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    game_loop = request.app["game_loop"]
    game_loop.ws_clients.add(ws)
    try:
        async for _ in ws:
            pass
    finally:
        game_loop.ws_clients.discard(ws)
    return ws


async def api_health(request):
    return web.json_response({"status": "ok"})


async def api_config(request):
    return web.json_response(request.app["game_config"])


async def api_match(request):
    gl = request.app["game_loop"]
    return web.json_response({
        "tick": gl.tick,
        "title": request.app["game_config"]["title"],
        "status": "running",
    })


async def start_game_loop(app):
    app["game_task"] = asyncio.create_task(app["game_loop"].run())


async def stop_game_loop(app):
    app["game_loop"].running = False
    app["game_task"].cancel()
    try:
        await app["game_task"]
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    args = parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(args.model)
    print(f"Loaded model: {args.model}")

    obs_normalizer = None
    if args.vec_normalize:
        obs_normalizer = ObsNormalizer(args.vec_normalize)
        print(f"Loaded observation normalizer: {args.vec_normalize}")

    config_path = resolve_config(args.config)
    with open(config_path) as f:
        config = json.load(f)

    game_loop = GameLoop(model, config_path, args.scenario, config, args.frame_skip,
                         deterministic=not args.stochastic, obs_normalizer=obs_normalizer)

    app = web.Application()
    app["game_loop"] = game_loop
    app["game_config"] = config
    app.router.add_get("/api/health", api_health)
    app.router.add_get("/api/config", api_config)
    app.router.add_get("/api/match", api_match)
    app.router.add_get("/ws/observe", ws_observe)
    app.router.add_static("/", os.path.join(PROJECT_ROOT, "web"), show_index=True)

    app.on_startup.append(start_game_loop)
    app.on_cleanup.append(stop_game_loop)

    print(f"Watching at http://localhost:{args.port}")
    web.run_app(app, host="0.0.0.0", port=args.port, print=None)

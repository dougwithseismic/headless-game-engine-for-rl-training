#!/usr/bin/env python3
"""Run a trained model and broadcast telemetry for the web viewer."""

import asyncio
import json
import os
import sys
import signal

import numpy as np
import aiohttp.web

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ghostlobby as gl

SESSIONS_PATH = os.path.expanduser("~/.ghostlobby/sessions.json")


def register_session(port, title, config_path):
    os.makedirs(os.path.dirname(SESSIONS_PATH), exist_ok=True)
    sessions = []
    if os.path.exists(SESSIONS_PATH):
        try:
            sessions = json.loads(open(SESSIONS_PATH).read())
        except Exception:
            sessions = []
    sessions = [s for s in sessions if s.get("pid") != os.getpid()]
    sessions.append({"pid": os.getpid(), "port": port, "title": title,
                      "config_path": config_path, "scenario": "cs_lite",
                      "started_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S")})
    with open(SESSIONS_PATH, "w") as f:
        json.dump(sessions, f, indent=2)


def unregister_session():
    if not os.path.exists(SESSIONS_PATH):
        return
    try:
        sessions = json.loads(open(SESSIONS_PATH).read())
        sessions = [s for s in sessions if s.get("pid") != os.getpid()]
        with open(SESSIONS_PATH, "w") as f:
            json.dump(sessions, f, indent=2)
    except Exception:
        pass


def flatten_obs(agent_obs):
    parts = []
    for key in sorted(agent_obs.keys()):
        parts.extend(agent_obs[key])
    return np.array(parts, dtype=np.float32)


async def run(config_path, model_path, port, tick_rate, dummy_ai=False, self_play=False):
    from stable_baselines3 import PPO

    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    scenario = "cs_lite_dummy" if dummy_ai else "cs_lite"
    env = gl.GhostLobbyEnv(config_path, scenario=scenario)
    obs_dict, _ = env.reset()
    ppt = env.num_agents() // 2
    num_agents = env.num_agents()

    if self_play:
        print(f"Self-play mode: model controls all {num_agents} agents")

    ws_clients: set[aiohttp.web.WebSocketResponse] = set()

    app = aiohttp.web.Application()

    async def ws_observe(request):
        ws = aiohttp.web.WebSocketResponse()
        await ws.prepare(request)
        ws_clients.add(ws)
        try:
            async for _ in ws:
                pass
        finally:
            ws_clients.discard(ws)
        return ws

    async def api_health(req):
        return aiohttp.web.json_response({"status": "ok"})

    title = "Self-Play Viewer" if self_play else "Trained Model"

    async def api_match(req):
        return aiohttp.web.json_response({"tick": env.tick_count(), "title": title, "status": "running"})

    async def api_config(req):
        with open(config_path) as f:
            return aiohttp.web.json_response(json.load(f))

    async def api_obstacles(req):
        with open(config_path) as f:
            cfg = json.load(f)
        return aiohttp.web.json_response({"obstacles": cfg.get("obstacles", []), "spawn_points": []})

    app.router.add_get("/ws/observe", ws_observe)
    app.router.add_get("/api/health", api_health)
    app.router.add_get("/api/match", api_match)
    app.router.add_get("/api/config", api_config)
    app.router.add_get("/api/obstacles", api_obstacles)

    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    register_session(port, "Trained Model Viewer", config_path)
    print(f"Watching on port {port}")

    tick_count = 0
    interval = 1.0 / tick_rate

    try:
        while True:
            actions = {}
            if self_play:
                for agent_id in range(num_agents):
                    if agent_id in obs_dict:
                        obs_i = flatten_obs(obs_dict[agent_id])
                        action_i, _ = model.predict(obs_i, deterministic=True)
                        actions[agent_id] = [float(a) for a in action_i.tolist()]
            else:
                obs_0 = flatten_obs(obs_dict[0])
                action_0, _ = model.predict(obs_0, deterministic=True)
                actions = {0: [float(a) for a in action_0.tolist()]}

            obs_dict, rewards, terms, truncs, infos = env.step(actions)
            tick_count += 1

            telemetry = env.drain_telemetry()
            if ws_clients and telemetry:
                dead = set()
                for event_json in telemetry:
                    for ws in ws_clients:
                        try:
                            await ws.send_str(event_json)
                        except Exception:
                            dead.add(ws)
                ws_clients -= dead

            any_done = any(terms.get(i, False) for i in range(env.num_agents()))
            if any_done or tick_count >= 8192:
                env = gl.GhostLobbyEnv(config_path, scenario=scenario)
                obs_dict, _ = env.reset()
                tick_count = 0

            await asyncio.sleep(interval)
    finally:
        unregister_session()
        await runner.cleanup()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=3000)
    p.add_argument("--tick-rate", type=int, default=16)
    p.add_argument("--dummy-ai", action="store_true", help="Use dummy AI opponent")
    p.add_argument("--self-play", action="store_true", help="Model controls both sides")
    args = p.parse_args()

    signal.signal(signal.SIGINT, lambda *_: (unregister_session(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *_: (unregister_session(), sys.exit(0)))
    asyncio.run(run(args.config, args.model, args.port, args.tick_rate, args.dummy_ai, args.self_play))


if __name__ == "__main__":
    main()

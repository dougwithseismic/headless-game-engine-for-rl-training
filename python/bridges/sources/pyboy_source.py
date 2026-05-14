"""ObservationSource for Game Boy games via PyBoy.

Reads game state from RAM addresses and/or screen pixels.
Advances the emulator by calling tick() on each read().

The RAM address map is game-specific — pass it as a dict of
{name: (address, normalize_max)} pairs. Values are read as
unsigned bytes (0-255) and normalized to [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from bridges.core.obs_source import ObservationSourceInfo
from bridges.emulators.pyboy_host import PyBoyHost


@dataclass
class RAMFeature:
    name: str
    address: int
    normalize_max: float = 255.0


class PyBoyObservationSource:

    def __init__(
        self,
        host: PyBoyHost,
        ram_features: list[RAMFeature] | None = None,
        include_screen: bool = False,
        screen_downscale: int = 4,
        ticks_per_step: int = 1,
        render: bool = True,
    ):
        self._host = host
        self._ram_features = ram_features or []
        self._include_screen = include_screen
        self._screen_downscale = screen_downscale
        self._ticks_per_step = ticks_per_step
        self._render = render
        self._terminal = False

        self._n_ram = len(self._ram_features)
        self._screen_pixels = 0
        if include_screen:
            self._screen_h = 144 // screen_downscale
            self._screen_w = 160 // screen_downscale
            self._screen_pixels = self._screen_h * self._screen_w
        self._obs_dim = self._n_ram + self._screen_pixels
        self._obs_buffer = np.zeros(self._obs_dim, dtype=np.float32)
        self._ram_addresses = [f.address for f in self._ram_features]
        self._ram_norms = np.array([f.normalize_max for f in self._ram_features], dtype=np.float32) if self._ram_features else None

    def info(self) -> ObservationSourceInfo:
        names = [f.name for f in self._ram_features]
        if self._include_screen:
            names += [f"px_{i}" for i in range(self._screen_pixels)]
        return ObservationSourceInfo(
            name="pyboy_ram" + ("_screen" if self._include_screen else ""),
            observation_space=gym.spaces.Box(
                low=0.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32,
            ),
            native_hz=None,
            platform="any",
            feature_names=names if names else None,
        )

    def connect(self) -> None:
        if not self._host.started:
            self._host.start()
        self._terminal = False

    def read(self) -> np.ndarray:
        alive = self._host.tick(
            count=self._ticks_per_step,
            render=self._render or self._include_screen,
        )
        if not alive:
            self._terminal = True

        buf = self._obs_buffer

        if self._ram_addresses:
            mem = self._host.pyboy.memory
            for i, addr in enumerate(self._ram_addresses):
                buf[i] = mem[addr]
            buf[:self._n_ram] /= self._ram_norms

        if self._include_screen:
            screen = self._host.screen_ndarray()
            if screen.ndim == 3:
                gray = np.mean(screen[:, :, :3], axis=2)
            else:
                gray = screen.astype(np.float32)
            sh, sw = self._screen_h, self._screen_w
            ds = self._screen_downscale
            gray = gray[:sh * ds, :sw * ds].reshape(sh, ds, sw, ds).mean(axis=(1, 3))
            buf[self._n_ram:] = gray.flatten() * (1.0 / 255.0)

        return buf

    def is_terminal(self) -> bool:
        return self._terminal

    def disconnect(self) -> None:
        self._terminal = False

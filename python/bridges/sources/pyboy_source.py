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

        self._obs_dim = len(self._ram_features)
        if include_screen:
            # Game Boy screen: 160x144, downscaled, grayscale
            h = 144 // screen_downscale
            w = 160 // screen_downscale
            self._obs_dim += h * w

    def info(self) -> ObservationSourceInfo:
        return ObservationSourceInfo(
            name="pyboy_ram" + ("_screen" if self._include_screen else ""),
            observation_space=gym.spaces.Box(
                low=0.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32,
            ),
            native_hz=None,
            platform="any",
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

        parts: list[np.ndarray] = []

        if self._ram_features:
            ram_values = np.array(
                [self._host.read_memory(f.address) / f.normalize_max for f in self._ram_features],
                dtype=np.float32,
            )
            parts.append(ram_values)

        if self._include_screen:
            screen = self._host.screen_ndarray()
            # Convert to grayscale if RGB
            if screen.ndim == 3:
                gray = np.mean(screen[:, :, :3], axis=2)
            else:
                gray = screen.astype(np.float32)
            # Downscale via block averaging
            h, w = gray.shape
            nh = h // self._screen_downscale
            nw = w // self._screen_downscale
            gray = gray[:nh * self._screen_downscale, :nw * self._screen_downscale]
            gray = gray.reshape(nh, self._screen_downscale, nw, self._screen_downscale).mean(axis=(1, 3))
            parts.append((gray.flatten() / 255.0).astype(np.float32))

        if not parts:
            return np.zeros(1, dtype=np.float32)

        return np.concatenate(parts)

    def is_terminal(self) -> bool:
        return self._terminal

    def disconnect(self) -> None:
        self._terminal = False

"""Shared PyBoy emulator instance for bridge components.

All three bridge components (sink, source, reset) need the same PyBoy
instance. PyBoyHost owns the emulator lifecycle and provides the shared
reference. Components receive the host at construction time.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np


GAMEBOY_BUTTONS = ("up", "down", "left", "right", "a", "b", "start", "select")


class PyBoyHost:
    """Owns and manages a PyBoy emulator instance.

    Provides the shared state that ActionSink, ObservationSource, and
    ResetStrategy all need: the emulator itself, save state buffers,
    and frame advancement.
    """

    def __init__(
        self,
        rom_path: str | Path,
        headless: bool = True,
        speed: int = 0,
        cgb: bool = False,
    ):
        self.rom_path = str(rom_path)
        self._headless = headless
        self._speed = speed
        self._cgb = cgb
        self._pyboy = None
        self._save_states: dict[str, bytes] = {}

    @property
    def pyboy(self):
        if self._pyboy is None:
            raise RuntimeError("PyBoyHost not started — call start() first")
        return self._pyboy

    @property
    def started(self) -> bool:
        return self._pyboy is not None

    def start(self) -> None:
        if self._pyboy is not None:
            return

        import pyboy as pb

        kwargs = {}
        if self._headless:
            kwargs["window"] = "null"
        if self._cgb:
            kwargs["cgb"] = True

        self._pyboy = pb.PyBoy(self.rom_path, **kwargs)
        self._pyboy.set_emulation_speed(self._speed)

    def stop(self) -> None:
        if self._pyboy is not None:
            self._pyboy.stop(save=False)
            self._pyboy = None

    def tick(self, count: int = 1, render: bool = True) -> bool:
        return self.pyboy.tick(count=count, render=render)

    def button(self, name: str, delay: int = 1) -> None:
        self.pyboy.button(name, delay)

    def button_press(self, name: str) -> None:
        self.pyboy.button_press(name)

    def button_release(self, name: str) -> None:
        self.pyboy.button_release(name)

    def read_memory(self, address: int) -> int:
        return self.pyboy.memory[address]

    def read_memory_range(self, start: int, length: int) -> list[int]:
        return [self.pyboy.memory[start + i] for i in range(length)]

    def screen_ndarray(self) -> np.ndarray:
        return np.asarray(self.pyboy.screen.ndarray)

    def save_state_to_buffer(self, name: str = "default") -> None:
        buf = io.BytesIO()
        self.pyboy.save_state(buf)
        self._save_states[name] = buf.getvalue()

    def load_state_from_buffer(self, name: str = "default") -> None:
        if name not in self._save_states:
            raise KeyError(f"No save state named '{name}'. Available: {list(self._save_states.keys())}")
        buf = io.BytesIO(self._save_states[name])
        self.pyboy.load_state(buf)

    def save_state_to_file(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            self.pyboy.save_state(f)

    def load_state_from_file(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self.pyboy.load_state(f)

    @property
    def frame_count(self) -> int:
        return self.pyboy.frame_count

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

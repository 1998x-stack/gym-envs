"""Teaching implementation of FrozenLake using the Gymnasium 5-tuple API."""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class FrozenLakeEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array", "human"], "render_fps": 4}
    MAPS = {
        "4x4": ("SFFF", "FHFH", "FFFH", "HFFG"),
        "8x8": ("SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF",
                "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG"),
    }
    ACTION_DIRECTION = ((0, 1), (1, 0), (0, -1), (-1, 0))

    def __init__(self, desc=None, map_name="4x4", is_slippery=True, render_mode=None):
        super().__init__()
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError("unsupported render mode")
        desc = self.MAPS.get(map_name) if desc is None else desc
        if not desc or not all(len(row) == len(desc[0]) for row in desc):
            raise ValueError("map must be a non-empty rectangle")
        if set("".join(desc)) - set("SFHG") or sum(row.count("S") for row in desc) != 1 or not any("G" in row for row in desc):
            raise ValueError("map must use S/F/H/G and contain exactly one S and at least one G")
        self.desc = tuple(desc)
        self.nrow, self.ncol = len(desc), len(desc[0])
        self.is_slippery = bool(is_slippery)
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.nrow * self.ncol)
        self.state = None
        self._terminated = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        start = next((r, c) for r, row in enumerate(self.desc) for c, v in enumerate(row) if v == "S")
        self.state = int(start[0] * self.ncol + start[1])
        self._terminated = False
        return self.state, {"prob": 1.0}

    def step(self, action):
        if self.state is None:
            raise RuntimeError("call reset() before step()")
        if self._terminated:
            raise RuntimeError("episode ended; call reset()")
        if not self.action_space.contains(action):
            raise ValueError("invalid action")
        action = int(action)
        if self.is_slippery:
            action = int(self.np_random.choice((action, (action - 1) % 4, (action + 1) % 4)))
        r, c = divmod(self.state, self.ncol)
        dr, dc = self.ACTION_DIRECTION[action]
        r, c = max(0, min(r + dr, self.nrow - 1)), max(0, min(c + dc, self.ncol - 1))
        self.state = int(r * self.ncol + c)
        tile = self.desc[r][c]
        self._terminated = tile in "HG"
        reward = 1.0 if tile == "G" else 0.0
        return self.state, reward, self._terminated, False, {"prob": 1.0}

    def render(self):
        if self.state is None or self.render_mode is None:
            return None
        r, c = divmod(self.state, self.ncol)
        if self.render_mode in ("human", "ansi"):
            rows = [list(row) for row in self.desc]
            rows[r][c] = "A"
            output = "\n".join("".join(row) for row in rows)
            if self.render_mode == "human":
                print(output)
                return None
            return output
        colors = {"S": (100, 200, 130), "F": (220, 240, 250),
                  "H": (220, 80, 80), "G": (60, 160, 80)}
        image = np.zeros((self.nrow, self.ncol, 3), dtype=np.uint8)
        for y, row in enumerate(self.desc):
            for x, symbol in enumerate(row):
                image[y, x] = colors[symbol]
        image[r, c] = (25, 25, 35)
        return image

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GVGAI Gym environment backed by JPype (direct JVM call-in).

Replaces the subprocess + TCP-socket architecture in gvgai_env.py with a
JPype-based approach: the JVM is embedded in the Python process, and the
Java GVGAIBridge class is called directly — no ports, no serialisation
overhead, no external process management.

Public interface is identical to the original GVGAI_Env so that all
existing LLM agent code works without modification.
"""

import os
import io
import json
from os import path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------------
# JVM initialisation (module-level, done once)
# ---------------------------------------------------------------------------

_JVM_STARTED = False


def _ensure_jvm(build_dir: str, img_path: str) -> None:
    """Start the JVM and configure the GVGAI classpath.  Idempotent."""
    global _JVM_STARTED
    if _JVM_STARTED:
        return

    import jpype
    import jpype.imports

    if jpype.isJVMStarted():
        _JVM_STARTED = True
        return

    # AWT headless so the JVM never tries to open a display window
    jpype.startJVM(
        jpype.getDefaultJVMPath(),
        f"-Djava.class.path={build_dir}",
        "-Djava.awt.headless=true",
        "-Xmx512m",
        convertStrings=False,
    )
    _JVM_STARTED = True


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class GVGAI_Env_JPype(gym.Env):
    """
    OpenAI Gym environment for GVGAI games, using JPype for Java bridging.

    Parameters
    ----------
    game    : str   game name (e.g. "aliens")
    level   : int   level index 0-4
    version : int   game version (usually 0)
    base_dir: str   directory that contains ``gvgai/`` and ``games/``
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, game: str, level: int, version: int, base_dir: str):
        super().__init__()

        self.game    = game
        self.lvl     = level
        self.version = version

        # ------------------------------------------------------------------ #
        # Resolve paths
        # ------------------------------------------------------------------ #
        gvgai_dir  = path.join(base_dir, "gvgai")
        build_dir  = path.join(gvgai_dir, "GVGAI_Build")
        games_dir  = path.join(base_dir, "games", f"{game}_v{version}")
        sprites_dir = path.join(gvgai_dir, "sprites")

        if not path.isdir(build_dir):
            raise FileNotFoundError(
                f"GVGAI build directory not found: {build_dir}\n"
                "Run build.py first, or reinstall with pip."
            )

        # ------------------------------------------------------------------ #
        # Start JVM (once per process)
        # ------------------------------------------------------------------ #
        _ensure_jvm(build_dir, sprites_dir)

        import jpype
        from jpype import JClass

        # ------------------------------------------------------------------ #
        # Resolve game / level file paths for Java
        # ------------------------------------------------------------------ #
        game_file   = path.realpath(path.join(games_dir, f"{game}.txt"))
        level_files = [
            path.realpath(path.join(games_dir, f"{game}_lvl{i}.txt"))
            for i in range(5)
        ]
        # Level-5 slot is reserved for custom levels (kept empty here)
        level_files.append("")

        # ------------------------------------------------------------------ #
        # Set CompetitionParameters.IMG_PATH so sprites are found
        # ------------------------------------------------------------------ #
        CompetitionParameters = JClass("core.competition.CompetitionParameters")
        CompetitionParameters.IMG_PATH = sprites_dir + "/"

        # ------------------------------------------------------------------ #
        # Instantiate GVGAIBridge
        # ------------------------------------------------------------------ #
        GVGAIBridge = JClass("core.game.GVGAIBridge")
        jlevel_files = jpype.JArray(jpype.JString)(level_files)
        self._bridge = GVGAIBridge(game_file, jlevel_files)

        # ------------------------------------------------------------------ #
        # Bootstrap: reset once to discover actions / image shape
        # ------------------------------------------------------------------ #
        self._bridge.reset(level, 0)
        self._actions = self._get_actions()
        img = self._get_image()
        self._img_shape = img.shape  # (H, W, 4) RGBA from PNG

        self.action_space      = spaces.Discrete(len(self._actions))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self._img_shape, dtype=np.uint8
        )

        self._current_img = img
        self.viewer = None

    # ----------------------------------------------------------------------- #
    # Gym API
    # ----------------------------------------------------------------------- #

    def step(self, action: int):
        """
        Apply *action* and return either a 5-tuple (gymnasium) or 4-tuple (legacy gym).

        action  : int index into action_space.
                  0 = ACTION_NIL, 1+ = available actions.
        """
        if action == 0:
            action_name = "ACTION_NIL"
        else:
            action_name = self._actions[action]

        self._bridge.step(action_name)

        reward     = float(self._bridge.getScoreDelta())
        terminated = bool(self._bridge.isGameOver())
        winner     = str(self._bridge.getWinner())
        obs_str    = str(self._bridge.getObservationString())
        obs_grid   = self._parse_observation_grid(obs_str)
        img        = self._get_image()

        self._current_img = img
        info = {
            "winner":  winner,
            "actions": self._actions,
            "ascii":   obs_str,
            "grid":    obs_grid,
        }
        return img, reward, terminated, False, info   # truncated=False (handled by max_episode_steps wrapper)

    def reset(self, *, seed=None, options=None):
        """Reset to the stored level and return the initial observation.

        Returns (obs, info) for gymnasium, or just obs for legacy gym.
        """
        import random as _random
        rng_seed = seed if seed is not None else _random.randint(0, 2**31 - 1)
        self._bridge.reset(self.lvl, rng_seed)
        self._actions = self._get_actions()
        img = self._get_image()
        self._current_img = img
        return img, {}   # gymnasium: (obs, info)

    def render(self):
        """Return the current frame as an RGB numpy array."""
        return self._current_img[:, :, :3]  # drop alpha channel

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def get_action_meanings(self):
        """Return list of all action name strings (index 0 = ACTION_NIL)."""
        return self._actions

    def _setLevel(self, level):
        if isinstance(level, int):
            self.lvl = min(max(level, 0), 4)
        else:
            self.lvl = 0

    # ----------------------------------------------------------------------- #
    # Internal helpers
    # ----------------------------------------------------------------------- #

    def _get_actions(self) -> list:
        """Build action list: [ACTION_NIL] + available actions from Java."""
        import jpype
        java_actions = self._bridge.getAvailableActions()
        available = [str(a) for a in java_actions]
        return ["ACTION_NIL"] + available

    def _get_image(self) -> np.ndarray:
        """Render current frame; return as (H, W, 4) uint8 numpy array."""
        import jpype
        raw = bytes(self._bridge.renderToBytes())
        if not raw:
            if hasattr(self, "_img_shape"):
                return np.zeros(self._img_shape, dtype=np.uint8)
            return np.zeros((64, 64, 4), dtype=np.uint8)

        from PIL import Image
        img = np.array(Image.open(io.BytesIO(raw)).convert("RGBA"))
        return img

    @staticmethod
    def _parse_observation_grid(obs_str: str) -> np.ndarray:
        """
        Convert the ASCII observation string back into a 2-D list of lists.
        Each cell is a list of itypeKey strings (mirrors the original ``grid``
        field from the socket version).
        """
        if not obs_str:
            return np.array([])
        rows = obs_str.split("\n")
        grid = []
        for row in rows:
            grid.append(row.split(","))
        return np.array(grid, dtype=object)

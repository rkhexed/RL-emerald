"""Checks for emerald_rl/env.py. Needs Emerald.gba and states/01_mudkip.state (see SETUP.md).

Run: python tests/test_env.py
"""

import os
import tempfile
import time

import numpy as np
from pygba import PyGBA
from pygba.game_wrappers.pokemon_emerald import get_game_state

from emerald_rl.env import ACTION_NAMES, EmeraldEnv


def test_ram_matches_pygba():
    env = EmeraldEnv()
    env.reset()
    s = env.read()
    ref = get_game_state(PyGBA(env.core))  # pygba's full decoder, checked against the screen earlier
    assert s["map"] == (ref["location"]["mapGroup"], ref["location"]["mapNum"]) == (1, 4)  # Birch's lab
    assert (s["x"], s["y"]) == (ref["pos"]["x"], ref["pos"]["y"])
    assert list(s["levels"]) == [p["level"] for p in ref["party"]] == [5]
    assert s["hp"][0] == s["max_hp"][0] > 0
    assert s["badges"] == ref["badges"] and s["towns"] == 1  # only Littleroot visited so far
    env.close()


def test_deterministic_and_monotone():
    rng = np.random.default_rng(0)
    actions = rng.integers(0, len(ACTION_NAMES), 400)
    runs = []
    for _ in range(2):
        env = EmeraldEnv(max_steps=400)
        obs, _ = env.reset()
        trace, events = [], []
        for a in actions:
            obs, r, term, trunc, info = env.step(int(a))
            assert env.observation_space.contains(obs) and np.isfinite(r)
            s = env.read()
            trace.append((s["map"], s["x"], s["y"], r))
            events.append(env.scores["event"])
        assert trunc and "stats" in info
        assert all(b >= a for a, b in zip(events, events[1:])), "event score must never go down"
        runs.append(trace)
        env.close()
    assert runs[0] == runs[1], "same actions from the same state must give the same game"


def test_log_file():
    with tempfile.TemporaryDirectory() as d:
        env = EmeraldEnv(log_dir=d, max_steps=50)
        env.reset()
        for _ in range(50):
            env.step(0)
        env.close()
        log = np.fromfile(os.path.join(d, "env000.bin"), EmeraldEnv.LOG_DTYPE)
        assert len(log) == 50 and log.dtype.itemsize == 7 and (log["action"] == 0).all()


def test_speed():
    env = EmeraldEnv()
    env.reset()
    t = time.perf_counter()
    for i in range(300):
        env.step(i % 7)
    print(f"  {300 / (time.perf_counter() - t):.0f} steps/s in one process")
    env.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

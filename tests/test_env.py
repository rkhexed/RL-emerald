"""Checks for emerald_rl/env.py. Needs Emerald.gba and states/01_mudkip.state (see SETUP.md).

Run: python tests/test_env.py
"""

import os
import tempfile
import time
from pathlib import Path

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


def test_milestones_and_swarm():
    """Replays run02 env 0 episode 29, a real game that beat May; skipped if that run isn't on disk."""
    from emerald_rl.env import MILESTONES
    from emerald_rl.replay import load_episode
    if not os.path.exists("runs/run02/logs/env000.bin"):
        print("  skipped: needs runs/run02")
        return
    rows, state = load_episode("runs/run02", 0, 29)
    with tempfile.TemporaryDirectory() as d:
        env = EmeraldEnv(init_state=state, max_steps=10**6, swarm_dir=d, swarm_explore=0)
        env.reset()
        order = []
        for i, row in enumerate(rows):
            _, r, *_ = env.step(int(row["action"]))
            for m in sorted(env.milestones - {o for o, _ in order}):
                order.append((m, i))
        names = [MILESTONES[m][0] for m, _ in order]
        print("  milestones reached (step):", [(MILESTONES[m][0], i) for m, i in order])
        assert names.index("OLDALE_TOWN") < names.index("ROUTE_103") < names.index("RIVAL_BATTLE_WON")
        assert sorted(p.name for p in Path(d).glob("*.json")) == sorted(f"m{m:02d}_{MILESTONES[m][0]}.json" for m, _ in order)

        fresh = EmeraldEnv(init_state=state, swarm_dir=d, swarm_explore=0)
        obs, _ = fresh.reset()
        assert fresh.milestones == {m for m, _ in order}, "should start from the furthest swarm state"
        assert fresh.scores["milestone"] == 0, "milestones carried in from the swarm state are not paid again"
        assert obs["stats"][22 + MILESTONES.index(next(m for m in MILESTONES if m[0] == "RIVAL_BATTLE_WON"))] == 1
        assert fresh.read()["options"] == fresh.start_options, "swarm states are saved with our game settings"
        assert len(fresh.visits) == fresh.start["tiles"] > 100, "explored tiles come with the swarm state"
        assert fresh.scores["explore"] == 0, "inherited tiles are not paid again"


def test_counters():
    """run03 env 12 episode 29 entered Rustboro Gym and blacked out there; skipped if not on disk."""
    from emerald_rl.replay import load_episode
    if not os.path.exists("runs/run03/logs/env012.bin"):
        print("  skipped: needs runs/run03")
        return
    rows, state = load_episode("runs/run03", 12, 29)
    env = EmeraldEnv(init_state=state, max_steps=10**6)
    env.reset()
    for row in rows:
        env.step(int(row["action"]))
    st = env.stats()
    print(f"  battles {st['battles']}, blackouts {st['blackouts']}, steps moved/battle/locked/idle "
          f"{st['steps/moved']:.2f}/{st['steps/battle']:.2f}/{st['steps/locked']:.2f}/{st['steps/idle']:.2f}")
    assert st["blackouts"] >= 1 and st["battles"] > 0 and env.scores["blackout"] < 0
    assert abs(sum(st[f"steps/{k}"] for k in ("moved", "battle", "locked", "idle")) - 1) < 1e-6


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

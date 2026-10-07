"""Replay episodes from a run and report what the agents actually did, from game memory.

  python -m emerald_rl.analyze runs/run04 --episodes 15,35,56 --envs 0-7

Per episode: how each battle ended (won / lost / ran / caught, from gBattleOutcome), the lead
Pokemon's level and party size at the end, where blackouts happened, and the share of steps
moving / in battle / locked (dialogue or menu) / idle. Replays are exact (deterministic emulator).
"""

import argparse
import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from emerald_rl.env import EmeraldEnv
from emerald_rl.replay import load_episode, parse_envs

BATTLE_OUTCOME = 0x0202433A  # gBattleOutcome
OUTCOMES = {1: "won", 2: "lost", 3: "drew", 4: "ran", 5: "teleported", 6: "wild fled", 7: "caught"}
NAMES = {(m["bank"], m["num"]): m["name"]
         for m in json.loads((Path(__file__).parent / "data/map_data.json").read_text())["maps"]}


def analyze(job):
    run, env_id, episode = job
    rows, state = load_episode(run, env_id, episode)
    env = EmeraldEnv(init_state=state, max_steps=10**9)
    env.reset()
    outcomes, blackout_maps = Counter(), Counter()
    was_battle, before = False, 0
    for row in rows:
        before = env.counts["blackouts"]
        env.step(int(row["action"]))
        s = env.read()
        if was_battle and not s["in_battle"]:
            outcomes[OUTCOMES.get(int(env.ewram[BATTLE_OUTCOME - 0x02000000]), "other")] += 1
        was_battle = s["in_battle"]
        if env.counts["blackouts"] > before:
            blackout_maps[NAMES.get(s["map"], s["map"])] += 1
    st, s = env.stats(), env.read()
    env.close()
    return {"env": env_id, "episode": episode, "start": Path(state).stem, "outcomes": outcomes,
            "lead_level": int(s["levels"][0]) if len(s["levels"]) else 0, "party": len(s["party"]),
            "blackouts": blackout_maps, "steps": {k: st[f"steps/{k}"] for k in ("moved", "battle", "locked", "idle")}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--episodes", default="0", help="e.g. 15,35,56")
    p.add_argument("--envs", default="0-7")
    a = p.parse_args()
    jobs = [(a.run, e, int(ep)) for ep in a.episodes.split(",") for e in parse_envs(a.envs)]
    with Pool(min(len(jobs), 24)) as pool:
        results = pool.map(analyze, jobs)
    for ep in sorted({r["episode"] for r in results}):
        rs = [r for r in results if r["episode"] == ep]
        out = sum((r["outcomes"] for r in rs), Counter())
        n = sum(out.values()) or 1
        print(f"\nepisode {ep} ({len(rs)} games; starts: {dict(Counter(r['start'] for r in rs))})")
        print(f"  battles per game {n / len(rs):.1f}: " + ", ".join(f"{k} {100 * v / n:.0f}%" for k, v in out.most_common()))
        print(f"  lead Pokemon level at end: {sorted(r['lead_level'] for r in rs)}   party sizes: {sorted(r['party'] for r in rs)}")
        bl = sum((r["blackouts"] for r in rs), Counter())
        print(f"  blackouts per game {sum(bl.values()) / len(rs):.1f}, where: {dict(bl.most_common(5))}")
        print("  steps: " + ", ".join(f"{k} {100 * sum(r['steps'][k] for r in rs) / len(rs):.0f}%" for k in ("moved", "battle", "locked", "idle")))


if __name__ == "__main__":
    main()

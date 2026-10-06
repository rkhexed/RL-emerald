"""Re-render games from a run's button logs, in full colour, and tile them into a grid video.

Training only logs 7 bytes per step (map, x, y, button). The emulator is deterministic, so
replaying the buttons from the episode's start state reproduces the game exactly; every step
is checked against the logged (map, x, y) and mismatches are reported.

  python -m emerald_rl.replay runs/run01 --envs 0 --episode 0                 # one game
  python -m emerald_rl.replay runs/run01 --envs 0-15 --episode 0 --every 24   # 4x4 grid, one frame per step
  python -m emerald_rl.replay runs/run01 --envs 0-15 --start 18000 --steps 2000 --every 4

Output: runs/<run>/videos/ (one mp4 per env, plus grid_*.mp4). Upscaled with nearest-neighbour.
"""

import argparse
import math
import subprocess
from multiprocessing import Pool
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from emerald_rl.env import EmeraldEnv


def load_episode(run, env_id, episode):
    """Actions and logged positions of one episode, plus its start state."""
    logs = Path(run) / "logs"
    rows = np.fromfile(logs / f"env{env_id:03d}.bin", EmeraldEnv.LOG_DTYPE)
    starts = [line.split(",") for line in (logs / f"env{env_id:03d}.episodes.csv").read_text().split()]
    begin, state = int(starts[episode][0]), starts[episode][1]
    end = int(starts[episode + 1][0]) if episode + 1 < len(starts) else len(rows)
    return rows[begin:end], state


def render(job):
    run, env_id, episode, start, steps, every, scale, out = job
    rows, state = load_episode(run, env_id, episode)
    env = EmeraldEnv(init_state=state, max_steps=len(rows) + 1)
    env.reset()
    end = len(rows) if steps is None else min(len(rows), start + steps)
    mismatches, frame_no = 0, 0
    with imageio.get_writer(out, fps=60, codec="libx264", quality=8, macro_block_size=1) as w:

        def grab():
            nonlocal frame_no
            frame_no += 1
            if frame_no % every == 0:
                w.append_data(env.render().repeat(scale, 0).repeat(scale, 1))

        for i, row in enumerate(rows[:end]):
            env.press(int(row["action"]), grab if i >= start else None)
            s = env.read()
            mismatches += (s["map"], s["x"], s["y"]) != ((row["bank"], row["num"]), row["x"], row["y"])
    env.close()
    return out, end - start, mismatches


def grid(paths, out, cols):
    """Tile equally sized videos with ffmpeg's xstack filter (as Red's tile_vids_to_grid.py does)."""
    rows = math.ceil(len(paths) / cols)
    layout = "|".join(
        f"{'+'.join(f'w{c}' for c in range(x)) or 0}_{'+'.join(f'h{r * cols}' for r in range(y)) or 0}"
        for y in range(rows) for x in range(cols) if y * cols + x < len(paths)
    )
    inputs = [arg for p in paths for arg in ("-i", str(p))]
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex",
                    f"xstack=inputs={len(paths)}:layout={layout}:fill=black", "-c:v", "libx264",
                    "-crf", "20", "-pix_fmt", "yuv420p", str(out)], check=True)


def parse_envs(spec):
    a, _, b = spec.partition("-")
    return list(range(int(a), int(b) + 1)) if b else [int(x) for x in spec.split(",")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--envs", default="0", help="e.g. 0, 0-15 or 0,3,7")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--start", type=int, default=0, help="first step to record (earlier steps are replayed silently)")
    p.add_argument("--steps", type=int, help="how many steps to record (default: to the end)")
    p.add_argument("--every", type=int, default=24, help="keep every Nth frame; 24 = one frame per step")
    p.add_argument("--scale", type=int, help="pixel upscale (default 3 for one game, 1 in a grid)")
    p.add_argument("--cols", type=int, help="grid columns (default: square)")
    a = p.parse_args()

    envs = parse_envs(a.envs)
    scale = a.scale or (3 if len(envs) == 1 else 1)
    out_dir = Path(a.run) / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"ep{a.episode}_s{a.start}" + (f"_n{a.steps}" if a.steps else "")
    jobs = [(a.run, e, a.episode, a.start, a.steps, a.every, scale, str(out_dir / f"env{e:03d}_{tag}.mp4"))
            for e in envs]
    with Pool(min(len(jobs), 16)) as pool:
        for out, n, bad in pool.imap(render, jobs):
            print(f"{out}: {n} steps, {bad} position mismatches")
    if len(jobs) > 1:
        out = out_dir / f"grid_{tag}.mp4"
        grid([j[-1] for j in jobs], out, a.cols or math.ceil(math.sqrt(len(jobs))))
        print(out)


if __name__ == "__main__":
    main()

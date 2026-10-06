"""Train PPO on Emerald (the "layer 4" of LEARNING.md), in the style of Red's baseline_fast_v2.py.

  python -m emerald_rl.train --name first --envs 16 --steps 10_000_000
  python -m emerald_rl.train --name first --resume runs/first/ckpt_1000000_steps.zip
  tensorboard --logdir runs        # graphs (forward port 6006 over SSH)

Writes runs/<name>/: checkpoints, tensorboard/, logs/ (per-step map, x, y, action per env).
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from emerald_rl.env import EmeraldEnv


class StatsCallback(BaseCallback):
    """When episodes end, log mean and max of every env stat (reward terms, tiles, badges...)."""

    def _on_step(self):
        ended = [i["stats"] for i in self.locals["infos"] if "stats" in i]
        for key in ended[0] if ended else []:
            vals = np.array([e[key] for e in ended], float)
            self.logger.record(f"env_mean/{key}", vals.mean())
            self.logger.record(f"env_max/{key}", vals.max())
        return True


def make_env(i, run_dir, max_steps, state):
    return lambda: EmeraldEnv(init_state=state, max_steps=max_steps, log_dir=run_dir / "logs", env_id=i)


def snapshot(run_dir, args):
    """Everything a replay needs to reproduce this run exactly: its own copy of the start state
    (states/ gets rebuilt) and the code version and settings."""
    run_dir.mkdir(parents=True, exist_ok=True)
    state = run_dir / "start.state"
    if not state.exists():
        shutil.copy(args.state, state)
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True).stdout.strip()
    info = {"args": vars(args), "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "emerald_rl"))}
    with open(run_dir / "run.json", "a") as f:  # appended on every --resume
        f.write(json.dumps(info) + "\n")
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--steps", type=int, default=10_000_000, help="total env steps")
    p.add_argument("--episode", type=int, default=20_480, help="steps before an env resets to the start state")
    p.add_argument("--rollout", type=int, default=2048, help="steps per env between PPO updates")
    p.add_argument("--state", default="states/01_mudkip.state", help="start state (copied into the run)")
    p.add_argument("--resume")
    a = p.parse_args()

    run_dir = Path("runs") / a.name
    state = snapshot(run_dir, a)
    env = SubprocVecEnv([make_env(i, run_dir, a.episode, state) for i in range(a.envs)])
    if a.resume:
        model = PPO.load(a.resume, env=env, tensorboard_log=str(run_dir / "tensorboard"))
    else:
        model = PPO(
            "MultiInputPolicy", env, n_steps=a.rollout, batch_size=512, n_epochs=3,
            gamma=0.997, gae_lambda=0.95, ent_coef=0.01, learning_rate=3e-4,
            tensorboard_log=str(run_dir / "tensorboard"), verbose=1,
        )
    model.learn(
        total_timesteps=a.steps,
        callback=[StatsCallback(), CheckpointCallback(max(a.rollout * 10, 1), str(run_dir), "ckpt")],
        reset_num_timesteps=not a.resume,
    )
    model.save(run_dir / "final")
    env.close()


if __name__ == "__main__":
    main()

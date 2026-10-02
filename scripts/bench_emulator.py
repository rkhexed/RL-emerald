"""Measure how fast headless mGBA runs on this machine.

Reports, per process count:
  raw fps        emulated frames per second, nothing else
  decisions/s    agent steps per second at --frames-per-action (default 24,
                 as in Pleines et al.), including one screen grab per step
and, single-process only, the cost of a screen grab (numpy vs PIL) and of a
savestate save/load.

Usage:
  python scripts/bench_emulator.py --rom rom.gba
  python scripts/bench_emulator.py --rom rom.gba --procs 1,8,16 --seconds 20
  python scripts/bench_emulator.py --rom rom.gba --pygba   # adds pygba Emerald wrapper overhead
  python scripts/bench_emulator.py --rom rom.gba --state states/00_truck.state  # measure in-game, not the intro
"""

import argparse
import multiprocessing as mp
import time

import numpy as np


def make_core(rom, state=None):
    import mgba.core
    import mgba.image
    import mgba.log

    mgba.log.silence()
    core = mgba.core.load_path(rom)
    assert core is not None, f"could not load {rom}"
    img = mgba.image.Image(*core.desired_video_dimensions())
    core.set_video_buffer(img)
    core.reset()
    if state:
        from mgba._pylib import ffi

        core.load_raw_state(ffi.from_buffer(bytearray(open(state, "rb").read())))
    return core, img


def screen_numpy(img):
    from mgba._pylib import ffi

    # color_t is 32-bit XBGR: view the buffer directly, no PIL round trip
    raw = np.frombuffer(ffi.buffer(img.buffer), dtype=np.uint8)
    return raw.reshape(img.height, img.stride, 4)[:, : img.width, :3]


def worker(rom, state, seconds, fpa, out):
    core, img = make_core(rom, state)
    rng = np.random.default_rng()
    keys = [0, 1, 3, 4, 5, 6, 7]  # GBA_KEY_*: A B Start Right Left Up Down

    t0 = time.perf_counter()
    frames = 0
    while time.perf_counter() - t0 < seconds / 2:
        core.run_frame()
        frames += 1
    raw_fps = frames / (time.perf_counter() - t0)

    t0 = time.perf_counter()
    decisions = 0
    while time.perf_counter() - t0 < seconds / 2:
        core.set_keys(int(rng.choice(keys)))
        for i in range(fpa):
            if i == fpa // 3:
                core.set_keys()  # release after a third of the action, like 8 held / 16 released
            core.run_frame()
        screen_numpy(img).mean()
        decisions += 1
    out.put((raw_fps, decisions / (time.perf_counter() - t0)))


def single_process_extras(rom, state, pygba_wrapper):
    core, img = make_core(rom, state)
    for _ in range(600):
        core.run_frame()

    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        screen_numpy(img).copy()
    np_us = (time.perf_counter() - t0) / n * 1e6
    t0 = time.perf_counter()
    for _ in range(n):
        np.array(img.to_pil().convert("RGB"))
    pil_us = (time.perf_counter() - t0) / n * 1e6

    state = core.save_raw_state()
    n = 500
    t0 = time.perf_counter()
    for _ in range(n):
        core.save_raw_state()
    save_us = (time.perf_counter() - t0) / n * 1e6
    t0 = time.perf_counter()
    for _ in range(n):
        core.load_raw_state(state)
    load_us = (time.perf_counter() - t0) / n * 1e6
    print(f"screen grab: numpy {np_us:.0f} us, PIL {pil_us:.0f} us")
    print(f"savestate:   save {save_us:.0f} us, load {load_us:.0f} us, size {len(bytes(state)) / 1024:.0f} KiB")

    if pygba_wrapper:
        from pygba import PokemonEmerald, PyGBA

        gba = PyGBA(core)
        wrapper = PokemonEmerald()
        n = 50
        t0 = time.perf_counter()
        for _ in range(n):
            gba.core.run_frame()
            wrapper.reward(gba, None)
        print(f"pygba PokemonEmerald.reward(): {(time.perf_counter() - t0) / n * 1e3:.1f} ms per call")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rom", required=True)
    p.add_argument("--state", help="savestate to start from, e.g. one in the overworld")
    p.add_argument("--procs", default="1,4,8,16")
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--frames-per-action", type=int, default=24)
    p.add_argument("--pygba", action="store_true", help="also time pygba's Emerald reward wrapper")
    a = p.parse_args()

    single_process_extras(a.rom, a.state, a.pygba)
    print(f"\n{'procs':>5} {'raw fps/proc':>13} {'decisions/s/proc':>17} {'decisions/s total':>18}")
    for n in map(int, a.procs.split(",")):
        q = mp.Queue()
        ps = [mp.Process(target=worker, args=(a.rom, a.state, a.seconds, a.frames_per_action, q)) for _ in range(n)]
        for proc in ps:
            proc.start()
        res = [q.get() for _ in ps]
        for proc in ps:
            proc.join()
        fps, dps = np.mean(res, axis=0)
        print(f"{n:>5} {fps:>13,.0f} {dps:>17,.0f} {dps * n:>18,.0f}")


if __name__ == "__main__":
    main()

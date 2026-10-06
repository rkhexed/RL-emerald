"""Drive Emerald by hand from a savestate: press keys, save a screenshot, print RAM.

Used to script the intro and make the starting savestates, and to check the
RAM readers against what is on screen.

Key spec, comma separated:
  A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, L, R   press once (hold 8, release 16 frames)
  A*20                                               press 20 times
  W120                                               wait 120 frames

Examples:
  python scripts/drive.py --keys "W600,START*3,A*40" --save states/title.state --shot shot.png
  python scripts/drive.py --load states/title.state --keys "RIGHT*4" --shot shot.png
  python scripts/drive.py --keys @scripts/intro.keys --save states/00_truck.state   # rebuild the start state
  python scripts/drive.py --keys @scripts/intro.keys --video intro.mp4 --speed 8
"""

import argparse
from pathlib import Path

import numpy as np

import mgba.core
import mgba.image
import mgba.log
from mgba._pylib import ffi
from pygba import PyGBA
from pygba.game_wrappers.pokemon_emerald import get_game_state

from emerald_rl.env import pin_clock

KEYS = {"A": 0, "B": 1, "SELECT": 2, "START": 3, "RIGHT": 4, "LEFT": 5, "UP": 6, "DOWN": 7, "R": 8, "L": 9}


def run(core, spec, on_frame=lambda: None):
    def frames(n):
        for _ in range(n):
            core.run_frame()
            on_frame()

    for tok in filter(None, (t.strip() for t in spec.split(","))):
        if tok[0] == "W" and tok[1:].isdigit():
            frames(int(tok[1:]))
            continue
        name, _, n = tok.partition("*")
        for _ in range(int(n or 1)):
            core.set_keys(KEYS[name.upper()])
            frames(8)
            core.set_keys()
            frames(16)


class VideoWriter:
    """3x nearest-neighbour upscale, plus a 2 s end card with the RAM readout in a bar below the game."""

    def __init__(self, path):
        import imageio.v2 as imageio
        from PIL import ImageFont

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.w = imageio.get_writer(path, fps=60, codec="libx264", quality=8, macro_block_size=8)
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        self.font = ImageFont.truetype(font, 18)

    def _frame(self, img, bottom=""):
        from PIL import Image, ImageDraw

        game = img.to_pil().convert("RGB").resize((720, 480), Image.NEAREST)
        canvas = Image.new("RGB", (720, 480 + 40), (16, 16, 24))
        canvas.paste(game, (0, 0))
        if bottom:
            ImageDraw.Draw(canvas).text((12, 490), bottom, font=self.font, fill=(120, 220, 140))
        return np.asarray(canvas)

    def add(self, img):
        self.w.append_data(self._frame(img))

    def close(self, img, readout):
        for _ in range(120):
            self.w.append_data(self._frame(img, "RAM: " + readout))
        self.w.close()


def describe(gba):
    s = get_game_state(gba)
    if "location" not in s:
        return "save blocks not initialised yet"
    loc, pos = s["location"], s["pos"]
    flags = sum(b.bit_count() for b in s["script_flags"])
    party = [(p["box"]["nickname"], p["level"]) for p in s["party"] if p.get("box")]
    return (f"map {loc['mapGroup']}.{loc['mapNum']} pos ({pos['x']},{pos['y']}) | "
            f"badges {s['num_badges']} | script flags set {flags} | party {party}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rom", default="Emerald.gba")
    p.add_argument("--load")
    p.add_argument("--keys", default="")
    p.add_argument("--save")
    p.add_argument("--shot")
    p.add_argument("--video", help="record an mp4 of the whole key sequence")
    p.add_argument("--speed", type=int, default=1, help="keep every Nth frame (N x speed at 60 fps)")
    a = p.parse_args()
    if a.keys.startswith("@"):
        a.keys = ",".join(l.split("#")[0].strip() for l in Path(a.keys[1:]).read_text().splitlines())

    mgba.log.silence()
    gba = PyGBA.load(a.rom)
    core = gba.core
    pin_clock(core)
    img = mgba.image.Image(*core.desired_video_dimensions())
    core.set_video_buffer(img)
    core.reset()
    if a.load:
        data = bytearray(Path(a.load).read_bytes())
        core.load_raw_state(ffi.from_buffer(data))
        core.run_frame()

    writer, frame_no = None, [0]
    if a.video:
        writer = VideoWriter(a.video)

        def on_frame():
            frame_no[0] += 1
            if frame_no[0] % a.speed == 0:
                writer.add(img)
    else:
        on_frame = lambda: None

    run(core, a.keys, on_frame)

    if writer:
        writer.close(img, describe(gba))

    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        Path(a.save).write_bytes(bytes(ffi.buffer(core.save_raw_state())))
    if a.shot:
        img.to_pil().convert("RGB").resize((480, 320)).save(a.shot)
    print(describe(gba))


if __name__ == "__main__":
    main()

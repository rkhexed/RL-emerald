"""Hoenn map visuals: where every agent walked, on the real game map.

  python -m emerald_rl.mapviz heatmap runs/run01            # runs/run01/map/heatmap.png
  python -m emerald_rl.mapviz walkers runs/run01 --episode 5 --steps 3000
  python -m emerald_rl.mapviz world                          # cache/hoenn.png, the whole stitched overworld

The background is drawn from the pokeemerald decomp (no ROM needed): each map is a grid of
16x16 metatiles, each metatile is 2 layers of 4 tiles (8x8, 16-colour, with flip bits and a
palette number). Agent positions come from the 7-byte step logs; map_data.json places every
map on one global Hoenn grid (indoor maps at the door that leads into them).
"""

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from emerald_rl.env import EmeraldEnv

DECOMP = Path.home() / "refs/pokeemerald"
MAPS = json.loads((Path(__file__).parent / "data/map_data.json").read_text())
CACHE = Path("cache")
T = 16  # pixels per map tile (metatile)


# ---- drawing maps from the decomp ---------------------------------------------------------

@lru_cache(None)
def _tileset_files():
    """gTileset_X -> (tiles.png, [16 .pal], metatiles.bin), resolved through the decomp's C headers."""
    src = DECOMP / "src/data/tilesets"
    headers = (src / "headers.h").read_text()
    graphics = (src / "graphics.h").read_text() + (DECOMP / "src/graphics.c").read_text()  # primary tilesets live in graphics.c
    metatiles = (src / "metatiles.h").read_text()
    out = {}
    for name, body in re.findall(r"const struct Tileset (\w+) =\s*\{(.*?)\};", headers, re.S):
        tiles_sym = re.search(r"\.tiles = (\w+)", body).group(1)
        pals_sym = re.search(r"\.palettes = (\w+)", body).group(1)
        meta_sym = re.search(r"\.metatiles = (\w+)", body).group(1)
        tiles = re.search(rf"{tiles_sym}\[\] = INCGFX_U32\(\"([^\"]+)\"", graphics).group(1)
        pal_block = re.search(rf"{pals_sym}\[\]\[16\] =\s*\{{(.*?)\}};", graphics, re.S).group(1)
        pals = re.findall(r"\"([^\"]+\.pal)\"", pal_block)
        meta = re.search(rf"{meta_sym}\[\] = INCBIN_U16\(\"([^\"]+)\"", metatiles).group(1)
        out[name] = (DECOMP / tiles, [DECOMP / p for p in pals], DECOMP / meta)
    return out


def _palette(path):
    lines = path.read_text().split("\n")[3:19]  # JASC-PAL header, then 16 "r g b" lines
    return np.array([[int(v) for v in line.split()] for line in lines], np.uint8)


@lru_cache(None)
def _tileset(name):
    tiles_png, pals, meta = _tileset_files()[name]
    idx = np.array(Image.open(tiles_png))  # 4bpp indexed image, 16 tiles wide
    h, w = idx.shape
    tiles = idx.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8) & 15
    return tiles, np.stack([_palette(p) for p in pals]), np.fromfile(meta, "<u2").reshape(-1, 8)


@lru_cache(None)
def _layout(layout_id):
    layouts = {l["id"]: l for l in json.loads((DECOMP / "data/layouts/layouts.json").read_text())["layouts"] if "id" in l}
    return layouts[layout_id]


@lru_cache(None)
def map_image(map_name):
    """RGB image of one map, 16 px per tile, as the game draws it (minus sprites and animation)."""
    m = json.loads((DECOMP / "data/maps" / map_name / "map.json").read_text())
    lay = _layout(m["layout"])
    p_tiles, p_pals, p_meta = _tileset(lay["primary_tileset"])
    s_tiles, s_pals, s_meta = _tileset(lay["secondary_tileset"])
    tiles = np.concatenate([p_tiles[:512], s_tiles])  # VRAM: primary tiles 0-511, secondary from 512
    pals = np.concatenate([p_pals[:6], s_pals[6:13]])  # palettes 0-5 primary, 6-12 secondary
    metas = np.concatenate([p_meta[:512], s_meta])  # metatiles 0-511 primary, then secondary
    blocks = np.fromfile(DECOMP / lay["blockdata_filepath"], "<u2").reshape(lay["height"], lay["width"]) & 0x3FF

    img = np.zeros((lay["height"] * T, lay["width"] * T, 3), np.uint8)
    for (r, c), block in np.ndenumerate(blocks):
        for layer in (0, 1):
            for q in range(4):
                v = int(metas[block][layer * 4 + q])
                tile = tiles[min(v & 0x3FF, len(tiles) - 1)]
                tile = tile[:, ::-1] if v >> 10 & 1 else tile
                tile = tile[::-1] if v >> 11 & 1 else tile
                pal = pals[min(v >> 12, len(pals) - 1)]
                y, x = r * T + (q // 2) * 8, c * T + (q % 2) * 8
                dst = img[y:y + 8, x:x + 8]
                opaque = tile != 0 if layer else np.ones_like(tile, bool)  # colour 0 is see-through on top
                dst[opaque] = pal[tile[opaque]]
    return img


def world(region=None):
    """The stitched overworld (outdoor maps only), optionally cropped to (x0, y0, x1, y1) in tiles."""
    W, H = MAPS["world_tiles"]
    x0, y0, x1, y1 = region or (0, 0, W, H)
    img = np.zeros(((y1 - y0) * T, (x1 - x0) * T, 3), np.uint8)
    for m in MAPS["maps"]:
        if "coordinates" not in m:
            continue
        mx, my = m["coordinates"]
        if mx >= x1 or my >= y1 or mx + m["width"] <= x0 or my + m["height"] <= y0:
            continue
        tile = map_image(m["name"])
        # paste the overlapping part
        sx0, sy0 = max(x0 - mx, 0), max(y0 - my, 0)
        sx1, sy1 = min(x1 - mx, m["width"]), min(y1 - my, m["height"])
        img[(my + sy0 - y0) * T:(my + sy1 - y0) * T, (mx + sx0 - x0) * T:(mx + sx1 - x0) * T] = \
            tile[sy0 * T:sy1 * T, sx0 * T:sx1 * T]
    return img


# ---- agent positions ----------------------------------------------------------------------

PLACE = {(m["bank"], m["num"]): m for m in MAPS["maps"]}


def to_global(rows):
    """Step log rows -> global Hoenn tile (gx, gy); indoor maps collapse to their door. -1 if unplaced."""
    gx = np.full(len(rows), -1, np.int32)
    gy = np.full(len(rows), -1, np.int32)
    keys = rows["bank"].astype(np.int32) * 256 + rows["num"]
    blip = np.zeros(len(keys), bool)  # one step on a map between two steps elsewhere: pre-guard garbage
    blip[1:-1] = (keys[1:-1] != keys[:-2]) & (keys[1:-1] != keys[2:])
    for key in np.unique(keys):
        m = PLACE.get((key // 256, key % 256))
        sel = (keys == key) & ~blip & (rows["x"] >= 0) & (rows["y"] >= 0)  # garbage from logs made before env.read()'s guard
        if m:
            sel &= (rows["x"] < m["width"]) & (rows["y"] < m["height"])
        if m and "coordinates" in m:
            gx[sel] = m["coordinates"][0] + rows["x"][sel]
            gy[sel] = m["coordinates"][1] + rows["y"][sel]
        elif m and "anchor" in m:
            gx[sel], gy[sel] = m["anchor"]
    return gx, gy


def load_run(run):
    return {int(p.stem[3:]): np.fromfile(p, EmeraldEnv.LOG_DTYPE) for p in sorted((Path(run) / "logs").glob("env*.bin"))}


def crop_box(gx, gy, margin=5):
    ok = gx >= 0
    W, H = MAPS["world_tiles"]
    return (max(gx[ok].min() - margin, 0), max(gy[ok].min() - margin, 0),
            min(gx[ok].max() + margin + 1, W), min(gy[ok].max() + margin + 1, H))


# ---- outputs ------------------------------------------------------------------------------

def heatmap(run, last=None):
    """Every visited tile, coloured by how often agents stood there (log scale), over the map."""
    logs = load_run(run)
    gx, gy = map(np.concatenate, zip(*(to_global(r[-last:] if last else r) for r in logs.values())))
    box = crop_box(gx, gy)
    bg = world(box).astype(np.float32) * 0.55
    ok = gx >= 0
    counts = np.zeros((box[3] - box[1], box[2] - box[0]))
    np.add.at(counts, (gy[ok] - box[1], gx[ok] - box[0]), 1)
    heat = np.log1p(counts) / np.log1p(counts.max())
    colour = np.stack([255 * np.ones_like(heat), 255 * (1 - heat), 40 * np.ones_like(heat)], -1)  # yellow -> red
    alpha = np.where(counts > 0, 0.35 + 0.55 * heat, 0)[..., None]
    up = lambda a: a.repeat(T, 0).repeat(T, 1)
    img = (bg * (1 - up(alpha)) + up(colour) * up(alpha)).astype(np.uint8)
    out = Path(run) / "map" / (f"heatmap_last{last}.png" if last else "heatmap.png")
    out.parent.mkdir(exist_ok=True)
    Image.fromarray(img).save(out)
    print(f"{out}: {int(ok.sum())} steps from {len(logs)} envs, {int((counts > 0).sum())} tiles visited")


@lru_cache(None)
def _player_sprites():
    """Brendan's 9 overworld frames (16x32 RGBA): stand down/up/left, walk down x2, up x2, left x2."""
    idx = np.array(Image.open(DECOMP / "graphics/object_events/pics/people/brendan/walking.png"))
    pal = _palette(DECOMP / "graphics/object_events/palettes/brendan.pal")
    rgba = np.concatenate([pal[idx], np.where(idx == 0, 0, 255)[..., None].astype(np.uint8)], -1)
    return [rgba[:, i * 16:(i + 1) * 16] for i in range(9)]


def _sprite(facing, moving, phase):
    """facing: 0 down, 1 up, 2 left, 3 right (left mirrored, as the game does)."""
    frames = _player_sprites()
    d = 2 if facing == 3 else facing
    img = frames[3 + 2 * d + phase % 2] if moving else frames[d]
    return img[:, ::-1] if facing == 3 else img


def walkers(run, episode=0, start=0, steps=3000, inter=2, scale=2):
    """Red-style overlay: every env's player sprite walking on the Hoenn map at once, all starting
    from the same episode start, moving smoothly between tiles (inter frames per step)."""
    logs = load_run(run)
    pos = {}
    for e, rows in logs.items():
        starts = [int(l.split(",")[0]) for l in (Path(run) / "logs" / f"env{e:03d}.episodes.csv").read_text().split()]
        a = starts[episode] + start
        pos[e] = to_global(rows[a:a + steps])
    n = min(len(p[0]) for p in pos.values())
    gx = np.concatenate([p[0][:n] for p in pos.values()])
    gy = np.concatenate([p[1][:n] for p in pos.values()])
    box = crop_box(gx, gy)
    bg = world(box)
    out = Path(run) / "map" / f"walkers_ep{episode}_s{start}_n{n}.mp4"
    out.parent.mkdir(exist_ok=True)
    facing = {e: 0 for e in pos}
    with imageio.get_writer(out, fps=60, codec="libx264", quality=8, macro_block_size=1) as w:
        for i in range(1, n):
            for f in range(inter):
                frame = bg.copy()
                t = (f + 1) / inter
                for e, (xs, ys) in pos.items():
                    x0, y0, x1, y1 = xs[i - 1], ys[i - 1], xs[i], ys[i]
                    if x1 < 0:
                        continue
                    dx, dy = x1 - x0, y1 - y0
                    moving = x0 >= 0 and abs(dx) + abs(dy) == 1  # one tile: walk; bigger jump = warp, snap
                    if moving:
                        facing[e] = 1 if dy < 0 else 0 if dy > 0 else 2 if dx < 0 else 3
                        x, y = x0 + dx * t, y0 + dy * t
                    else:
                        x, y = x1, y1
                    spr = _sprite(facing[e], moving, i)
                    px, py = int((x - box[0]) * T), int((y - box[1]) * T) - T  # 16x32 sprite stands on its tile
                    if not (0 <= px <= bg.shape[1] - 16 and 0 <= py <= bg.shape[0] - 32):
                        continue
                    region = frame[py:py + 32, px:px + 16]
                    mask = spr[..., 3:] > 0
                    region[:] = np.where(mask, spr[..., :3], region)
                w.append_data(frame.repeat(scale, 0).repeat(scale, 1))
    print(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["heatmap", "walkers", "world"])
    p.add_argument("run", nargs="?")
    p.add_argument("--last", type=int, help="heatmap: only the last N steps of each env")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--episode", type=int, default=0, help="walkers: episode whose start every env begins from")
    p.add_argument("--inter", type=int, default=2, help="walkers: frames per step (smooth movement between tiles)")
    a = p.parse_args()
    if a.what == "world":
        CACHE.mkdir(exist_ok=True)
        Image.fromarray(world()).save(CACHE / "hoenn.png")
        print(CACHE / "hoenn.png")
    elif a.what == "heatmap":
        heatmap(a.run, a.last)
    else:
        walkers(a.run, a.episode, a.start, a.steps, a.inter)


if __name__ == "__main__":
    main()

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
from collections import deque
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


def _draw(blocks, lay):
    """RGB image of a grid of metatile ids, 16 px per tile, drawn with the layout's two tilesets."""
    p_tiles, p_pals, p_meta = _tileset(lay["primary_tileset"])
    s_tiles, s_pals, s_meta = _tileset(lay["secondary_tileset"])
    tiles = np.concatenate([p_tiles[:512], s_tiles])  # VRAM: primary tiles 0-511, secondary from 512
    pals = np.concatenate([p_pals[:6], s_pals[6:13]])  # palettes 0-5 primary, 6-12 secondary
    metas = np.concatenate([p_meta[:512], s_meta])  # metatiles 0-511 primary, then secondary
    img = np.zeros((blocks.shape[0] * T, blocks.shape[1] * T, 3), np.uint8)
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


def _map_layout(map_name):
    return _layout(json.loads((DECOMP / "data/maps" / map_name / "map.json").read_text())["layout"])


@lru_cache(None)
def map_image(map_name):
    """RGB image of one map, 16 px per tile, as the game draws it (minus sprites and animation)."""
    lay = _map_layout(map_name)
    blocks = np.fromfile(DECOMP / lay["blockdata_filepath"], "<u2").reshape(lay["height"], lay["width"]) & 0x3FF
    return _draw(blocks, lay)


@lru_cache(None)
def border_image(map_name):
    """The 2x2-metatile pattern the game draws past this map's edges (trees, water...), 32x32 px."""
    lay = _map_layout(map_name)
    return _draw(np.fromfile(DECOMP / lay["border_filepath"], "<u2").reshape(2, 2) & 0x3FF, lay)


@lru_cache(None)
def _town_map():
    """Emerald's Town Map: per-pixel sea mask (512x512) and each Town Map section's pixel rect by id."""
    D = DECOMP / "graphics/pokenav/region_map"
    im = Image.open(D / "map.png")
    sheet = np.array(im)
    tiles = sheet.reshape(sheet.shape[0] // 8, 8, sheet.shape[1] // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    grid = np.fromfile(D / "map.bin", np.uint8).reshape(64, 64)
    px = np.array(im.getpalette(), np.uint8).reshape(-1, 3)[tiles[grid].transpose(0, 2, 1, 3).reshape(512, 512)]
    sea = (px[..., 2] > px[..., 1]) & (px[..., 2] > px[..., 0])  # blue: open sea and water routes
    secs = json.loads((DECOMP / "src/data/region_map/region_map_sections.json").read_text())["map_sections"]
    # Town Map cell (x, y) sits at pixel ((x + 1) * 8, (y + 2) * 8)
    return sea, {s["id"]: ((s["x"] + 1) * 8, (s["y"] + 2) * 8, s["width"] * 8, s["height"] * 8) for s in secs if "x" in s}


def town_map_sea(m, gx, gy):
    """True where Emerald's Town Map shows sea at world tiles (gx, gy), lined up locally: map `m`'s
    rectangle is stretched onto its own Town Map section and the tiles around it follow. None if `m`
    has no section on the Town Map."""
    sea, rects = _town_map()
    rect = rects.get(json.loads((DECOMP / f"data/maps/{m['name']}/map.json").read_text())["region_map_section"])
    if rect is None:
        return None
    (mx, my), (u0, v0, uw, vh) = m["coordinates"], rect
    u = (u0 + (gx - mx) / m["width"] * uw).astype(int)
    v = (v0 + (gy - my) / m["height"] * vh).astype(int)
    inside = (u >= 0) & (u < 512) & (v >= 0) & (v < 512)
    return ~inside | sea[np.clip(v, 0, 511), np.clip(u, 0, 511)]


def world(region=None, visited=None, borders=False):
    """The stitched overworld, optionally cropped to (x0, y0, x1, y1) in tiles (may extend past the
    world). Maps placed beside their door (Petalburg Woods, caves) are drawn only if in `visited`
    (a set of (bank, num)), when given. borders=True fills gaps instead of leaving them black: sea
    or forest as Emerald's Town Map shows there, with trees on the nearest map's grid."""
    W, H = MAPS["world_tiles"]
    x0, y0, x1, y1 = region or (0, 0, W, H)
    img = np.zeros(((y1 - y0) * T, (x1 - x0) * T, 3), np.uint8)
    owner = np.full((y1 - y0, x1 - x0), -1)  # per tile: index into `drawn` of the map drawn there
    drawn = []
    for m in MAPS["maps"]:
        if "coordinates" not in m or (m.get("beside_door") and visited is not None and (m["bank"], m["num"]) not in visited):
            continue
        mx, my = m["coordinates"]
        drawn.append(m)
        if mx >= x1 or my >= y1 or mx + m["width"] <= x0 or my + m["height"] <= y0:
            continue
        tile = map_image(m["name"])
        # paste the overlapping part
        sx0, sy0 = max(x0 - mx, 0), max(y0 - my, 0)
        sx1, sy1 = min(x1 - mx, m["width"]), min(y1 - my, m["height"])
        img[(my + sy0 - y0) * T:(my + sy1 - y0) * T, (mx + sx0 - x0) * T:(mx + sx1 - x0) * T] = \
            tile[sy0 * T:sy1 * T, sx0 * T:sx1 * T]
        owner[my + sy0 - y0:my + sy1 - y0, mx + sx0 - x0:mx + sx1 - x0] = len(drawn) - 1
    if borders:  # gaps: sea or forest as the Town Map shows there, trees lined up with the nearest map
        h, w = owner.shape
        src = np.full((h, w, 2), -1)  # nearest drawn tile (row, col) for every tile, by breadth-first search
        q = deque()
        for r, c in zip(*np.nonzero(owner >= 0)):
            src[r, c] = (r, c)
            q.append((r, c))
        while q:
            r, c = q.popleft()
            for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= rr < h and 0 <= cc < w and src[rr, cc, 0] < 0:
                    src[rr, cc] = src[r, c]
                    q.append((rr, cc))
        gap = (owner < 0) & (src[..., 0] >= 0)
        near = np.where(gap, owner[src[..., 0], src[..., 1]], -1)  # nearest map for every gap tile
        wet = np.zeros((h, w), bool)
        for i in np.unique(near[near >= 0]):
            rows, cols = np.nonzero(near == i)
            sea_here = town_map_sea(drawn[i], x0 + cols + 0.5, y0 + rows + 0.5)  # land or sea as the Town Map draws it
            if sea_here is not None:
                wet[rows, cols] = sea_here
        seen = np.zeros((h, w), bool)  # drop sea specks (< 64 tiles): Town Map pixels that land just inland
        for r0, c0 in zip(*np.nonzero(wet)):
            if seen[r0, c0]:
                continue
            blob, q = [(r0, c0)], deque([(r0, c0)])
            seen[r0, c0] = True
            while q:
                r, c = q.popleft()
                for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if 0 <= rr < h and 0 <= cc < w and wet[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        blob.append((rr, cc))
                        q.append((rr, cc))
            if len(blob) < 64:
                wet[tuple(np.array(blob).T)] = False
        sea, forest = border_image("Route105"), border_image("Route101")
        for r, c in zip(*np.nonzero(gap)):
            m = drawn[near[r, c]]
            ux, uy = (x0 + c - m["coordinates"][0]) % 2, (y0 + r - m["coordinates"][1]) % 2  # that map's tree grid
            pattern = sea if wet[r, c] else forest
            img[r * T:(r + 1) * T, c * T:(c + 1) * T] = pattern[uy * T:(uy + 1) * T, ux * T:(ux + 1) * T]
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


def load_run(run, envs=None):
    logs = {int(p.stem[3:]): np.fromfile(p, EmeraldEnv.LOG_DTYPE) for p in sorted((Path(run) / "logs").glob("env*.bin"))}
    return {e: r for e, r in logs.items() if envs is None or e in envs}


def episode_rows(run, env, rows, episode):
    starts = [int(l.split(",")[0]) for l in (Path(run) / "logs" / f"env{env:03d}.episodes.csv").read_text().split()]
    return rows[starts[episode]:starts[episode + 1] if episode + 1 < len(starts) else len(rows)]


def _tag(envs, episode):
    return (f"_env{'-'.join(map(str, envs))}" if envs else "") + (f"_ep{episode}" if episode is not None else "")


def _visited(rows_list):
    return {(int(b), int(n)) for r in rows_list for b, n in set(zip(r["bank"].tolist(), r["num"].tolist()))}


def crop_box(gx, gy, margin=5):
    ok = gx >= 0
    W, H = MAPS["world_tiles"]
    return (max(gx[ok].min() - margin, 0), max(gy[ok].min() - margin, 0),
            min(gx[ok].max() + margin + 1, W), min(gy[ok].max() + margin + 1, H))


# ---- outputs ------------------------------------------------------------------------------

def heatmap(run, last=None, envs=None, episode=None):
    """Every visited tile, coloured by how often agents stood there (log scale), over the map."""
    logs = load_run(run, envs)
    if episode is not None:
        logs = {e: episode_rows(run, e, r, episode) for e, r in logs.items()}
    gx, gy = map(np.concatenate, zip(*(to_global(r[-last:] if last else r) for r in logs.values())))
    box = crop_box(gx, gy)
    bg = world(box, _visited(logs.values())).astype(np.float32) * 0.55
    ok = gx >= 0
    counts = np.zeros((box[3] - box[1], box[2] - box[0]))
    np.add.at(counts, (gy[ok] - box[1], gx[ok] - box[0]), 1)
    heat = np.log1p(counts) / np.log1p(counts.max())
    colour = np.stack([255 * np.ones_like(heat), 255 * (1 - heat), 40 * np.ones_like(heat)], -1)  # yellow -> red
    alpha = np.where(counts > 0, 0.35 + 0.55 * heat, 0)[..., None]
    up = lambda a: a.repeat(T, 0).repeat(T, 1)
    img = (bg * (1 - up(alpha)) + up(colour) * up(alpha)).astype(np.uint8)
    out = Path(run) / "map" / (f"heatmap{_tag(envs, episode)}" + (f"_last{last}" if last else "") + ".png")
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


def walkers(run, episode=0, start=0, steps=3000, inter=2, scale=2, envs=None):
    """Red-style overlay: every env's player sprite walking on the Hoenn map at once, all starting
    from the same episode start, moving smoothly between tiles (inter frames per step)."""
    pos = {e: to_global(episode_rows(run, e, rows, episode)[start:start + steps])
           for e, rows in load_run(run, envs).items()}
    n = min(len(p[0]) for p in pos.values())
    gx = np.concatenate([p[0][:n] for p in pos.values()])
    gy = np.concatenate([p[1][:n] for p in pos.values()])
    box = crop_box(gx, gy)
    bg = world(box, _visited(load_run(run, envs).values()))
    out = Path(run) / "map" / f"walkers{_tag(envs, episode)}_s{start}_n{n}.mp4"
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
    p.add_argument("--episode", type=int, help="only this episode (walkers default: 0)")
    p.add_argument("--envs", type=lambda v: [int(x) for x in v.split(",")], help="only these envs, e.g. 0 or 0,3")
    p.add_argument("--inter", type=int, default=2, help="walkers: frames per step (smooth movement between tiles)")
    p.add_argument("--scale", type=int, default=2, help="walkers: pixel upscale (use 1 for long routes)")
    a = p.parse_args()
    if a.what == "world":
        CACHE.mkdir(exist_ok=True)
        Image.fromarray(world()).save(CACHE / "hoenn.png")
        print(CACHE / "hoenn.png")
    elif a.what == "heatmap":
        heatmap(a.run, a.last, a.envs, a.episode)
    else:
        walkers(a.run, a.episode or 0, a.start, a.steps, a.inter, a.scale, envs=a.envs)


if __name__ == "__main__":
    main()

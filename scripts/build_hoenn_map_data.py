"""Build Hoenn global map coordinates from the pokeemerald decomp (no ROM needed).

Emerald stores the player position as (map bank, map num, x, y) local to the
current map. To draw every agent on one shared Hoenn map (the PokemonRedExperiments
visual) we need each map's offset in a global grid, which Red's repo hand-placed
in map_data.json. Here it is computed:

  * outdoor maps: breadth-first walk over map.json `connections` from Littleroot,
    each neighbour placed by its edge direction and offset
  * warp-reached routes and caves (Petalburg Woods): full size, in the nearest free space beside their door
  * indoor maps: anchored at the outdoor door tile that warps into them
    (`anchor`), so an agent inside a building shows up at that building's door

Usage:
  git clone --depth 1 https://github.com/pret/pokeemerald ~/refs/pokeemerald
  python scripts/build_hoenn_map_data.py ~/refs/pokeemerald emerald_rl/data/map_data.json
"""

import json
import sys

import numpy as np
from collections import deque
from pathlib import Path


def free_spot(door, m, pos, maps, gap=1, pad=200):
    """Origin closest to `door` where an m-sized map overlaps no placed map (with a `gap` tile margin)."""
    x0 = min(p[0] for p in pos.values()) - pad
    y0 = min(p[1] for p in pos.values()) - pad
    x1 = max(p[0] + maps[k]["width"] for k, p in pos.items()) + pad
    y1 = max(p[1] + maps[k]["height"] for k, p in pos.items()) + pad
    occ = np.zeros((y1 - y0, x1 - x0), np.int32)
    for k, (px, py) in pos.items():
        occ[py - y0 - gap:py - y0 + maps[k]["height"] + gap, px - x0 - gap:px - x0 + maps[k]["width"] + gap] = 1
    S = np.pad(occ.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    W, H = m["width"], m["height"]
    oy, ox = np.mgrid[0:occ.shape[0] - H + 1, 0:occ.shape[1] - W + 1]
    free = (S[oy + H, ox + W] - S[oy, ox + W] - S[oy + H, ox] + S[oy, ox]) == 0
    dx, dy = door[0] - x0, door[1] - y0
    dist = np.maximum(np.maximum(ox - dx, dx - (ox + W - 1)), 0) + np.maximum(np.maximum(oy - dy, dy - (oy + H - 1)), 0)
    i = np.argmin(np.where(free, dist, 10**9))
    return int(ox.flat[i] + x0), int(oy.flat[i] + y0)


def main(decomp, out):
    decomp = Path(decomp)
    groups = json.loads((decomp / "data/maps/map_groups.json").read_text())
    layouts = {
        l["id"]: l
        for l in json.loads((decomp / "data/layouts/layouts.json").read_text())["layouts"]
        if "id" in l
    }

    maps = {}
    for bank, group in enumerate(groups["group_order"]):
        for num, name in enumerate(groups[group]):
            m = json.loads((decomp / "data/maps" / name / "map.json").read_text())
            lay = layouts[m["layout"]]
            maps[m["id"]] = {
                "id": m["id"],
                "name": name,
                "bank": bank,
                "num": num,
                "width": lay["width"],
                "height": lay["height"],
                "type": m["map_type"],
                "connections": m.get("connections") or [],
                "warps": m.get("warp_events") or [],
            }

    # outdoor stitching via connections
    pos = {"MAP_LITTLEROOT_TOWN": (0, 0)}
    conflicts = []
    q = deque(pos)
    while q:
        cur = maps[q.popleft()]
        x, y = pos[cur["id"]]
        for c in cur["connections"]:
            nb = maps.get(c["map"])
            if nb is None or c["direction"] not in ("up", "down", "left", "right"):
                continue  # dive/emerge connect underwater layers, not neighbours
            o = c["offset"]
            p = {
                "up": (x + o, y - nb["height"]),
                "down": (x + o, y + cur["height"]),
                "left": (x - nb["width"], y + o),
                "right": (x + cur["width"], y + o),
            }[c["direction"]]
            if nb["id"] in pos:
                if pos[nb["id"]] != p:
                    conflicts.append((cur["id"], nb["id"], pos[nb["id"]], p))
            else:
                pos[nb["id"]] = p
                q.append(nb["id"])

    # warp-reached routes and caves (Petalburg Woods, tunnels) are bigger inside than the overworld gap
    # they fill, so they can't line up with both doors. Draw them full size in the nearest free space
    # beside the door that leads in, as Red's map does with Viridian Forest.
    beside_door = set()
    changed = True
    while changed:
        changed = False
        for m in list(maps.values()):
            if m["id"] not in pos:
                continue
            for w in m["warps"]:
                nb = maps.get(w["dest_map"])
                if nb and nb["type"] in ("MAP_TYPE_ROUTE", "MAP_TYPE_UNDERGROUND") and nb["id"] not in pos:
                    pos[nb["id"]] = free_spot((pos[m["id"]][0] + w["x"], pos[m["id"]][1] + w["y"]), nb, pos, maps)
                    beside_door.add(nb["id"])
                    changed = True

    # indoor maps: anchor at the door of an already placed map that warps into them
    anchor = {}
    changed = True
    while changed:
        changed = False
        for m in maps.values():
            if m["id"] in pos:
                base = pos[m["id"]]
            elif m["id"] in anchor:
                base = anchor[m["id"]]
            else:
                continue
            for w in m["warps"]:
                dest = w["dest_map"]
                if dest in maps and dest not in pos and dest not in anchor:
                    door = (base[0] + w["x"], base[1] + w["y"]) if m["id"] in pos else base
                    anchor[dest] = door
                    changed = True

    minx = min(p[0] for p in pos.values())
    miny = min(p[1] for p in pos.values())
    out_maps = []
    for m in maps.values():
        e = {k: m[k] for k in ("id", "name", "bank", "num", "width", "height", "type")}
        if m["id"] in pos:
            e["coordinates"] = [pos[m["id"]][0] - minx, pos[m["id"]][1] - miny]
            if m["id"] in beside_door:
                e["beside_door"] = True  # not part of the overworld picture; visuals draw it only if visited
        elif m["id"] in anchor:
            e["anchor"] = [anchor[m["id"]][0] - minx, anchor[m["id"]][1] - miny]
        out_maps.append(e)

    w = max(p[0] + maps[k]["width"] for k, p in pos.items()) - minx
    h = max(p[1] + maps[k]["height"] for k, p in pos.items()) - miny
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"world_tiles": [w, h], "maps": out_maps}, indent=1))

    print(f"{len(maps)} maps | {len(pos)} placed full size (stitched or beside their door) | {len(anchor)} anchored indoors | "
          f"{len(maps) - len(pos) - len(anchor)} unplaced | world {w}x{h} tiles")
    for c in conflicts[:10]:
        print("conflict:", c)


if __name__ == "__main__":
    main(*sys.argv[1:3])

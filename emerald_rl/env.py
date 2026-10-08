"""Pokemon Emerald as a Gymnasium environment (the "layer 3" of LEARNING.md).

One step = press one of 7 buttons for 8 frames, release for 16, then look.
The agent sees a 120x80 grayscale screen (last 3 frames) plus a 4th channel
marking tiles it has already walked on, and a small vector of RAM facts.
Rewards are "best so far" scores, so nothing can be farmed by repetition.

Milestones (the PokeAgent Challenge route, read from game flags and maps) are both rewarded once
and shown to the agent as a vector, so it knows which part of the story it is in (the Hamburg
PokeRunners approach). With swarm_dir set, the first env to reach a milestone saves its state
there, and later episodes start from the furthest saved state (swarming).
"""

import json
import os
from collections import deque
from pathlib import Path

import gymnasium as gym
import mgba.core
import mgba.image
import mgba.log
import numpy as np
from mgba._pylib import ffi, lib

# GBA key ids (mGBA): A B SELECT START RIGHT LEFT UP DOWN R L = 0..9
ACTIONS = {"UP": 6, "DOWN": 7, "LEFT": 5, "RIGHT": 4, "A": 0, "B": 1, "START": 3}
ACTION_NAMES = list(ACTIONS)

# Addresses from pret/pokeemerald (pokeemerald.sym, include/global.h, include/pokemon.h)
SAVEBLOCK1_PTR = 0x03005D8C  # SaveBlock1 moves around in EWRAM ("DMA protection"); this pointer doesn't
SAVEBLOCK2_PTR = 0x03005D90
GMAIN_VBLANK_CB = 0x030022C0 + 0x0C  # gMain.vblankCallback: NULL while the game moves its save blocks
GMAIN_IN_BATTLE = 0x030022C0 + 0x439  # gMain.inBattle is bit 1 of this byte
LOCK_FIELD_CONTROLS = 0x03000F2C  # sLockFieldControls: player can't move (dialogue, START menu, cutscene)
PARTY_COUNT = 0x020244E9
PARTY = 0x020244EC  # 6 x struct Pokemon (100 bytes); level/HP sit after the encrypted 80-byte box
SB1_POS, SB1_LOCATION, SB1_FLAGS = 0x00, 0x04, 0x1270
SB2_OPTIONS = 0x14
FLAG_BYTES = slice(0x50 // 8, 0x920 // 8)  # script + trainer + system flags; skips temp flags 0x00-0x4F
BADGE_FLAGS = range(0x867, 0x86F)
TOWN_FLAGS = range(0x86F, 0x87F)  # FLAG_VISITED_LITTLEROOT_TOWN .. EVER_GRANDE_CITY

MILESTONES = [  # official route (sethkarten/pokeagent-speedrun) from our start state, plus Pokemon Centers
    ("OLDALE_TOWN", "map", (0, 10)),
    ("OLDALE_POKEMON_CENTER", "map", (2, 2)),
    ("ROUTE_103", "map", (0, 18)),
    ("RIVAL_BATTLE_WON", "flag", (0x500 + 529, 0x500 + 532, 0x500 + 535)),  # May's trainer flag, one per starter
    ("RECEIVED_POKEDEX", "flag", (0x861,)),
    ("ROUTE_102", "map", (0, 17)),
    ("PETALBURG_CITY", "map", (0, 0)),
    ("PETALBURG_POKEMON_CENTER", "map", (8, 4)),
    ("DAD_FIRST_MEETING", "map", (8, 1)),  # Petalburg Gym
    ("ROUTE_104", "map", (0, 19)),
    ("PETALBURG_WOODS", "map", (24, 11)),
    ("TEAM_AQUA_GRUNT_DEFEATED", "flag", (0x500 + 10,)),
    ("RUSTBORO_CITY", "map", (0, 3)),
    ("RUSTBORO_POKEMON_CENTER", "map", (11, 5)),
    ("RUSTBORO_GYM_ENTERED", "map", (11, 3)),
    ("STONE_BADGE", "flag", (0x867,)),
]

# struct Pokemon: personality and OT id are plain, the 48 bytes at 32 are 4 substructures XORed
# with (personality ^ otId) and stored in one of 24 orders picked by personality % 24
GROWTH_POS = [0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 2, 3, 1, 1, 2, 3, 2, 3, 1, 1, 2, 3, 2, 3]  # where the growth block (species) sits
ATTACKS_POS = [1, 1, 2, 3, 2, 3, 0, 0, 0, 0, 0, 0, 2, 3, 1, 1, 3, 2, 2, 3, 1, 1, 3, 2]  # where the attacks block (4 moves) sits
SB1_LAST_HEAL = 0x1C  # SaveBlock1.lastHealLocation (WarpData): where a blackout sends you; changes when you heal
SB2_DEX_OWNED, SB2_DEX_SEEN = 0x28, 0x5C  # Pokedex flags, 52 bytes each
RECENT_ACTIONS = 4
POS_FREQS = 6  # global position encoded as sin/cos at 6 frequencies per axis (Hamburg-style)
N_STATS = 22 + len(MILESTONES) + RECENT_ACTIONS * len(ACTIONS) + 2 + 4 * POS_FREQS

SCREEN_TILES = (10, 15)  # GBA screen = 15 x 10 tiles of 16 px
PLAYER_TILE = (5, 7)  # (row, col) of the player's tile on screen; checked in tests/test_env.py

_MAP_DATA = json.loads((Path(__file__).parent / "data/map_data.json").read_text())
MAP_SIZES = {(m["bank"], m["num"]): (m["width"], m["height"]) for m in _MAP_DATA["maps"]}
# where each map sits on the global Hoenn grid: (x, y, True) for full maps, (door x, door y, False) for interiors
MAP_ORIGIN = {(m["bank"], m["num"]): (*m["coordinates"], True) if "coordinates" in m else (*m["anchor"], False)
              for m in _MAP_DATA["maps"] if "coordinates" in m or "anchor" in m}
WORLD = _MAP_DATA["world_tiles"]

DEFAULT_WEIGHTS = {
    "event": 1.0,  # per story/trainer/system flag set (most ever, minus flags set at start)
    "explore": 0.02,  # per unique (map bank, map num, x, y) tile visited this episode
    "level": 0.5,  # per party level gained (full to +15, then 1/4); 2.0 (run05) caused grinding, 0 fails (puffer)
    "badge": 5.0,
    "town": 2.0,  # per "visited town" flag, on top of its event flag
    "milestone": 5.0,  # per milestone reached for the first time this episode
    "evolve": 5.0,  # per party member that evolved (cancelling with B gives up this reward)
    "catch": 2.0,  # per new Pokemon in the party (caught)
    "blackout": 0.0,  # penalty per blackout; off: in run04 it taught the agent to run from every battle
    "seen": 1.0,  # per new species seen (meeting it in battle), as pokemonred_puffer
    "moves": 2.0,  # per distinct move ever known by the party, as pokemonred_puffer
    "heal": 1.0,  # per new place healed at (the blackout respawn point changes), as pokemonred_puffer
    "options": 1.0,  # refundable penalty while game options differ from the start state
    "stuck": 0.0,  # per step on a tile visited 600+ times (Red); off: Hamburg found negative rewards made agents timid
}


def pin_clock(core, epoch_ms=1_767_268_800_000):  # 2026-01-01 12:00 UTC
    """Emerald reads the cartridge clock. By default mGBA feeds it the host's date, so replays
    differ by day. FAKE_EPOCH starts it at a fixed date and advances it with emulated frames."""
    core._core.rtc.override = lib.RTC_FAKE_EPOCH
    core._core.rtc.value = epoch_ms


_LOG_FILTER = None


def silence_logs():
    """Drop all mGBA log messages in C. mgba.log.silence() alone still calls back into Python for
    every message, and once PyTorch is imported that callback deadlocks on the GIL (seen as
    load_raw_state hanging forever). A filter that allows no levels stops messages before Python."""
    global _LOG_FILTER
    mgba.log.silence()
    if _LOG_FILTER is None:
        _LOG_FILTER = ffi.new("struct mLogFilter *")
        lib.mLogFilterInit(_LOG_FILTER)
        _LOG_FILTER.defaultLevels = 0
    mgba.log.Logger._DEFAULT_LOGGER._native.filter = _LOG_FILTER


def no_python_callbacks(core):
    """The Python bindings hook every core with callbacks into Python several times per frame
    (frame start/end, key reads, sleep). We use none of them; with PyTorch imported they deadlock
    on the GIL, and they cost time every frame. The core keeps its own copy, so clear that list."""
    core._core.clearCoreCallbacks(core._core)


def _memory(core, region):
    """Zero-copy numpy view of a GBA memory region (2 = EWRAM, 3 = IWRAM). Always current."""
    size = ffi.new("size_t *")
    ptr = core._core.getMemoryBlock(core._core, region, size)
    return np.frombuffer(ffi.buffer(ffi.cast("uint8_t *", ptr), size[0]), np.uint8)


class EmeraldEnv(gym.Env):
    def __init__(self, rom="Emerald.gba", init_state="states/01_mudkip.state", max_steps=20_480,
                 weights=None, log_dir=None, env_id=0, swarm_dir=None, swarm_explore=0.25):
        silence_logs()
        self.core = mgba.core.load_path(str(rom))
        no_python_callbacks(self.core)
        pin_clock(self.core)
        self.img = mgba.image.Image(240, 160)
        self.core.set_video_buffer(self.img)
        self.core.reset()
        self.ewram, self.iwram = _memory(self.core, 2), _memory(self.core, 3)
        self.frame = np.frombuffer(ffi.buffer(self.img.buffer), np.uint8).reshape(160, 240, 4)

        self.init_state = Path(init_state)
        self.max_steps = max_steps
        self.w = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.env_id = env_id
        self.swarm_dir = Path(swarm_dir) if swarm_dir else None
        self.swarm_explore = swarm_explore  # share of episodes starting from a random earlier swarm state
        if self.swarm_dir:
            self.swarm_dir.mkdir(parents=True, exist_ok=True)

        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Dict({
            "screen": gym.spaces.Box(0, 255, (80, 120, 4), np.uint8),  # 3 stacked frames + visited mask
            "stats": gym.spaces.Box(0, 1, (N_STATS,), np.float32),
        })

        self.log = None
        if log_dir is not None:  # per-step (map, x, y, action) for map visuals and full-quality replays
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self.log = open(Path(log_dir) / f"env{env_id:03d}.bin", "ab")
            self.episodes = open(Path(log_dir) / f"env{env_id:03d}.episodes.csv", "a")
        self.rows = []
        self.total_steps = 0
        # our game settings, from the original start state (swarm states may have them changed)
        self.core.load_raw_state(ffi.from_buffer(bytearray(self.init_state.read_bytes())))
        self.core.run_frame()
        self.start_options = self._read_raw()["options"]

    # ---- RAM ---------------------------------------------------------------------------------

    def _u32(self, mem, off):
        return int.from_bytes(mem[off:off + 4].tobytes(), "little")

    def _sb(self, ptr_addr):
        return self._u32(self.iwram, ptr_addr - 0x03000000) - 0x02000000

    def read(self):
        """Every RAM fact the env uses, in one place.

        On map loads Emerald copies its save blocks to the heap and back ("DMA protection"), and
        a frame can end mid-copy, so RAM briefly holds garbage (maps that don't exist, x = 12802).
        Then the last good reading is returned instead."""
        s = self._read_raw()
        size = MAP_SIZES.get(s["map"])
        ok = (self._u32(self.iwram, GMAIN_VBLANK_CB - 0x03000000) != 0 and size is not None
              and 0 <= s["x"] < size[0] and 0 <= s["y"] < size[1])
        if ok or not hasattr(self, "last_read"):
            self.last_read = s
        return self.last_read

    def _read_raw(self):
        sb1, sb2 = self._sb(SAVEBLOCK1_PTR), self._sb(SAVEBLOCK2_PTR)
        ew = self.ewram
        x, y = np.frombuffer(ew[sb1 + SB1_POS:sb1 + SB1_POS + 4].tobytes(), np.int16)
        bank, num = ew[sb1 + SB1_LOCATION], ew[sb1 + SB1_LOCATION + 1]
        flags = ew[sb1 + SB1_FLAGS:sb1 + SB1_FLAGS + 300]
        n = min(int(ew[PARTY_COUNT - 0x02000000]), 6)
        party = ew[PARTY - 0x02000000:PARTY - 0x02000000 + 600].reshape(6, 100)[:n]
        levels = party[:, 84].astype(int)
        words = party.copy().view("<u4")  # (n, 25) little-endian words
        pid, otid = words[:, 0], words[:, 1]
        species = [int((words[i, 8 + 3 * GROWTH_POS[pid[i] % 24]] ^ pid[i] ^ otid[i]) & 0xFFFF) for i in range(n)]
        moves = set()
        for i in range(n):
            a = words[i, 8 + 3 * ATTACKS_POS[pid[i] % 24]: 8 + 3 * ATTACKS_POS[pid[i] % 24] + 2] ^ (pid[i] ^ otid[i])
            moves |= {int(a[0] & 0xFFFF), int(a[0] >> 16), int(a[1] & 0xFFFF), int(a[1] >> 16)} - {0}
        hp = party[:, 86:88].copy().view(np.uint16)[:, 0].astype(int)
        max_hp = party[:, 88:90].copy().view(np.uint16)[:, 0].astype(int)

        def flag(i):
            return bool(flags[i // 8] >> (i % 8) & 1)

        return {
            "map": (int(bank), int(num)), "x": int(x), "y": int(y),
            "flags_set": int(np.unpackbits(flags[FLAG_BYTES]).sum()),
            "badges": [flag(i) for i in BADGE_FLAGS],
            "towns": sum(flag(i) for i in TOWN_FLAGS),
            "levels": levels, "hp": hp, "max_hp": max_hp,
            "party": dict(zip(pid.tolist(), species)),  # personality -> species
            "moves": moves,
            "heal": (int(ew[sb1 + SB1_LAST_HEAL]), int(ew[sb1 + SB1_LAST_HEAL + 1])),
            "seen": int(np.unpackbits(ew[sb2 + SB2_DEX_SEEN:sb2 + SB2_DEX_SEEN + 52]).sum()),
            "in_battle": bool(self.iwram[GMAIN_IN_BATTLE - 0x03000000] >> 1 & 1),
            "locked": bool(self.iwram[LOCK_FIELD_CONTROLS - 0x03000000]),
            "options": self._u32(ew, sb2 + SB2_OPTIONS) & 0xFFFF,
            "milestones_now": {i for i, (_, kind, v) in enumerate(MILESTONES)
                               if (kind == "map" and (int(bank), int(num)) == v) or (kind == "flag" and any(map(flag, v)))},
        }

    # ---- gym API -----------------------------------------------------------------------------

    def _pick_start(self):
        """(state path, milestones already reached). Swarming: usually the furthest saved state,
        sometimes a random earlier one so earlier parts of the route aren't forgotten."""
        options = [(self.init_state, set())]
        if self.swarm_dir:
            for meta in sorted(self.swarm_dir.glob("*.json")):
                options.append((meta.with_suffix(".state"), set(json.loads(meta.read_text()))))
        # each option: (state, milestones); explored tiles live next to the state as .tiles.npy
        if len(options) > 1 and self.np_random.random() < self.swarm_explore:
            return options[self.np_random.integers(len(options))]
        return max(options, key=lambda o: len(o[1]))

    def _save_swarm(self, new):
        """First env to reach a milestone saves its state for everyone (written atomically)."""
        # save with our game settings, so games starting here don't inherit text speed changes
        opt = self._sb(SAVEBLOCK2_PTR) + SB2_OPTIONS
        mine = self.ewram[opt:opt + 2].copy()
        self.ewram[opt:opt + 2] = np.frombuffer(self.start_options.to_bytes(2, "little"), np.uint8)
        state = bytes(ffi.buffer(self.core.save_raw_state()))
        self.ewram[opt:opt + 2] = mine
        tiles = np.array([b << 40 | n << 32 | (x & 0xFFFF) << 16 | (y & 0xFFFF) for b, n, x, y in self.visits], np.int64)
        for i in new:
            base = self.swarm_dir / f"m{i:02d}_{MILESTONES[i][0]}"
            if base.with_suffix(".json").exists():
                continue
            for suffix, data in ((".state", state), (".tiles.npy", tiles.tobytes()),
                                 (".json", json.dumps(sorted(self.milestones)).encode())):
                tmp = base.with_suffix(f"{suffix}.tmp{self.env_id}")
                tmp.write_bytes(data)
                os.replace(tmp, base.with_suffix(suffix))  # .json last: a state is only visible once complete

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        start_path, self.milestones = self._pick_start()
        self.core.load_raw_state(ffi.from_buffer(bytearray(Path(start_path).read_bytes())))
        self.core.run_frame()
        self.step_count = 0
        # (bank, num, x, y) -> times stood there; a swarm state brings the tiles explored on the way there,
        # so only new ground pays (walking back down the route is worth nothing)
        tiles_file = Path(start_path).with_suffix(".tiles.npy")
        known = np.fromfile(tiles_file, np.int64) if tiles_file.exists() else []
        self.visits = {(int(k >> 40), int(k >> 32 & 0xFF), int(np.int16(k >> 16 & 0xFFFF)), int(np.int16(k & 0xFFFF))): 1
                       for k in known}
        self.recent = deque([0] * RECENT_ACTIONS, maxlen=RECENT_ACTIONS)
        self.still = 0  # steps since the player's position last changed
        self.counts = {"moved": 0, "battle": 0, "locked": 0, "idle": 0, "battles": 0, "blackouts": 0}
        self.frames = deque([self._small_screen()] * 3, maxlen=3)
        s = self.read()
        self.milestones |= s["milestones_now"]
        self.start = {"flags_set": s["flags_set"], "level_sum": int(s["levels"].sum()),
                      "options": self.start_options, "badges": sum(s["badges"]), "towns": s["towns"],
                      "milestones": len(self.milestones), "tiles": len(self.visits)}
        self.first_species = dict(s["party"])  # personality -> species when first seen this episode
        self.moves_known, self.heals = set(s["moves"]), {s["heal"]}
        self.start.update(moves=len(self.moves_known), heals=1, seen=s["seen"])
        self.start_party = set(s["party"])
        self.fainted = False
        self.last_in_battle = s["in_battle"]
        self.best = {"flags": 0, "level": 0}
        self.stuck_total = 0.0
        self._visit(s)
        self.scores = self._scores(s)
        self.prev_total = sum(self.scores.values())
        if self.log:
            self.episodes.write(f"{self.total_steps},{start_path}\n")  # step offset where it began, and from which state
        return self._obs(s), {}

    def press(self, action, on_frame=None):
        """Hold the button 8 frames, release 16. Shared with replay.py so replays match exactly."""
        for frame in range(24):
            if frame == 0:
                self.core.set_keys(ACTIONS[ACTION_NAMES[action]])
            elif frame == 8:
                self.core.set_keys()
            self.core.run_frame()
            if on_frame:
                on_frame()

    def step(self, action):
        self.press(action)
        self.frames.append(self._small_screen())

        prev_key = self._key(self.last_read)
        s = self.read()
        moved = self._key(s) != prev_key
        self.still = 0 if moved else self.still + 1
        self.recent.append(action)
        if s["in_battle"] and not self.last_in_battle:
            self.counts["battles"] += 1
        self.last_in_battle = s["in_battle"]
        self.counts["battle" if s["in_battle"] else "moved" if moved else "locked" if s["locked"] else "idle"] += 1
        fainted = len(s["hp"]) > 0 and s["hp"].sum() == 0
        if fainted and not self.fainted:
            self.counts["blackouts"] += 1
        self.fainted = fainted
        for p, sp in s["party"].items():
            self.first_species.setdefault(p, sp)
        self.moves_known |= s["moves"]
        self.heals.add(s["heal"])
        if self.visits.get(self._key(s), 0) > 600:
            self.stuck_total += self.w["stuck"]
        self._visit(s)
        new = s["milestones_now"] - self.milestones
        if new:
            self.milestones |= new
            if self.swarm_dir:
                self._save_swarm(new)
        self.scores = self._scores(s)
        total = sum(self.scores.values())
        reward, self.prev_total = total - self.prev_total, total

        self.step_count += 1
        self.total_steps += 1
        self._log_step(s, action)
        truncated = self.step_count >= self.max_steps
        info = {"stats": self.stats(s)} if truncated else {}
        return self._obs(s), reward, False, truncated, info

    # ---- reward ------------------------------------------------------------------------------

    @staticmethod
    def _key(s):
        return (*s["map"], s["x"], s["y"])

    def _visit(self, s):
        k = self._key(s)
        self.visits[k] = self.visits.get(k, 0) + 1

    def _scores(self, s):
        """Score of the current state per term; the reward is the change in their sum."""
        self.best["flags"] = max(self.best["flags"], s["flags_set"] - self.start["flags_set"])
        gained = int(s["levels"].sum()) - self.start["level_sum"]
        self.best["level"] = max(self.best["level"], gained if gained <= 15 else 15 + (gained - 15) / 4)
        w = self.w
        return {
            "event": w["event"] * self.best["flags"],
            "explore": w["explore"] * (len(self.visits) - self.start["tiles"]),
            "level": w["level"] * self.best["level"],
            "badge": w["badge"] * (sum(s["badges"]) - self.start["badges"]),
            "town": w["town"] * (s["towns"] - self.start["towns"]),
            "milestone": w["milestone"] * (len(self.milestones) - self.start["milestones"]),
            "evolve": w["evolve"] * sum(s["party"].get(p, sp) != sp for p, sp in self.first_species.items()),
            "catch": w["catch"] * len(set(self.first_species) - self.start_party),
            "blackout": -w["blackout"] * self.counts["blackouts"],
            "seen": w["seen"] * max(s["seen"] - self.start["seen"], 0),
            "moves": w["moves"] * (len(self.moves_known) - self.start["moves"]),
            "heal": w["heal"] * (len(self.heals) - self.start["heals"]),
            "options": -w["options"] * (s["options"] != self.start["options"]),
            "stuck": -self.stuck_total,
        }

    # ---- observation -------------------------------------------------------------------------

    def _small_screen(self):
        rgb = self.frame[:, :, :3].astype(np.float32)  # bytes are R, G, B, X
        gray = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
        return gray.reshape(80, 2, 120, 2).mean((1, 3)).astype(np.uint8)

    def _visited_mask(self, s):
        """Tiles on screen the agent has stood on this episode, aligned with the screen."""
        rows, cols = SCREEN_TILES
        pr, pc = PLAYER_TILE
        mask = np.zeros(SCREEN_TILES, np.uint8)
        bank, num = s["map"]
        for r in range(rows):
            for c in range(cols):
                if (bank, num, s["x"] + c - pc, s["y"] + r - pr) in self.visits:
                    mask[r, c] = 255
        return mask.repeat(8, 0).repeat(8, 1)  # 16 px tiles at half resolution = 8 px

    def _obs(self, s):
        n = len(s["levels"])
        stats = np.zeros(N_STATS, np.float32)
        stats[:n] = s["levels"] / 100
        stats[6:6 + n] = s["hp"] / np.maximum(s["max_hp"], 1)
        stats[12:20] = s["badges"]
        stats[20] = s["options"] == self.start["options"]
        stats[21] = n / 6
        stats[22 + np.array(sorted(self.milestones), int)] = 1  # where am I in the story (Hamburg's milestone vector)
        o = 22 + len(MILESTONES)
        for i, a in enumerate(self.recent):  # last buttons pressed, so it can notice repeating itself
            stats[o + i * len(ACTIONS) + a] = 1
        o += RECENT_ACTIONS * len(ACTIONS)
        stats[o] = min(self.still / 200, 1)  # how long it has been standing still
        stats[o + 1] = s["in_battle"]
        o += 2
        ox, oy, full = MAP_ORIGIN.get(s["map"], (0, 0, False))  # where am I in Hoenn (Hamburg's global position)
        gx, gy = (ox + s["x"], oy + s["y"]) if full else (ox, oy)
        f = 2.0 ** np.arange(POS_FREQS) * np.pi
        for i, g in enumerate((gx / WORLD[0], gy / WORLD[1])):
            stats[o + 4 * POS_FREQS // 2 * i: o + 4 * POS_FREQS // 2 * (i + 1)] = (np.r_[np.sin(f * g), np.cos(f * g)] + 1) / 2
        return {"screen": np.stack([*self.frames, self._visited_mask(s)], axis=-1), "stats": stats}

    # ---- logging -----------------------------------------------------------------------------

    LOG_DTYPE = np.dtype([("bank", "u1"), ("num", "u1"), ("x", "<i2"), ("y", "<i2"), ("action", "u1")])

    def _log_step(self, s, action):
        if not self.log:
            return
        self.rows.append((*s["map"], s["x"], s["y"], action))
        if len(self.rows) >= 1000:
            self.flush()

    def flush(self):
        if self.log and self.rows:
            np.array(self.rows, self.LOG_DTYPE).tofile(self.log)
            self.log.flush()
            self.episodes.flush()
            self.rows = []

    def stats(self, s=None):
        s = s or self.read()
        return {**{f"reward/{k}": v for k, v in self.scores.items()},
                "tiles": len(self.visits), "flags": self.best["flags"], "badges": sum(s["badges"]),
                "towns": s["towns"], "level_sum": int(s["levels"].sum()),
                "milestones": len(self.milestones), "furthest_milestone": max(self.milestones, default=-1),
                **{f"steps/{k}": v / max(self.step_count, 1) for k, v in self.counts.items() if k in ("moved", "battle", "locked", "idle")},
                "battles": self.counts["battles"], "blackouts": self.counts["blackouts"],
                "party_size": len(s["party"]), "evolutions": sum(s["party"].get(p, sp) != sp for p, sp in self.first_species.items()),
                "moves_known": len(self.moves_known), "species_seen": s["seen"], "heal_places": len(self.heals),
                "map_bank": s["map"][0], "map_num": s["map"][1]}

    def render(self):
        return self.frame[:, :, :3].copy()

    def close(self):
        self.flush()
        if self.log:
            self.log.close()
            self.episodes.close()

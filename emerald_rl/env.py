"""Pokemon Emerald as a Gymnasium environment (the "layer 3" of LEARNING.md).

One step = press one of 7 buttons for 8 frames, release for 16, then look.
The agent sees a 120x80 grayscale screen (last 3 frames) plus a 4th channel
marking tiles it has already walked on, and a small vector of RAM facts.
Rewards are "best so far" scores, so nothing can be farmed by repetition.
"""

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
PARTY_COUNT = 0x020244E9
PARTY = 0x020244EC  # 6 x struct Pokemon (100 bytes); level/HP sit after the encrypted 80-byte box
SB1_POS, SB1_LOCATION, SB1_FLAGS = 0x00, 0x04, 0x1270
SB2_OPTIONS = 0x14
FLAG_BYTES = slice(0x50 // 8, 0x920 // 8)  # script + trainer + system flags; skips temp flags 0x00-0x4F
BADGE_FLAGS = range(0x867, 0x86F)
TOWN_FLAGS = range(0x86F, 0x87F)  # FLAG_VISITED_LITTLEROOT_TOWN .. EVER_GRANDE_CITY

SCREEN_TILES = (10, 15)  # GBA screen = 15 x 10 tiles of 16 px
PLAYER_TILE = (5, 7)  # (row, col) of the player's tile on screen; checked in tests/test_env.py

DEFAULT_WEIGHTS = {
    "event": 1.0,  # per story/trainer/system flag set (most ever, minus flags set at start)
    "explore": 0.02,  # per unique (map bank, map num, x, y) tile visited this episode
    "level": 0.5,  # per party level gained (full value to +15, then 1/4)
    "badge": 5.0,
    "town": 2.0,  # per "visited town" flag, on top of its event flag
    "options": 0.1,  # refundable penalty while game options differ from the start state
    "stuck": 0.025,  # per step standing on a tile visited more than 600 times (as in Red)
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
                 weights=None, log_dir=None, env_id=0):
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
        self.state_bytes = bytearray(self.init_state.read_bytes())
        self.max_steps = max_steps
        self.w = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.env_id = env_id

        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Dict({
            "screen": gym.spaces.Box(0, 255, (80, 120, 4), np.uint8),  # 3 stacked frames + visited mask
            "stats": gym.spaces.Box(0, 1, (22,), np.float32),
        })

        self.log = None
        if log_dir is not None:  # per-step (map, x, y, action) for map visuals and full-quality replays
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self.log = open(Path(log_dir) / f"env{env_id:03d}.bin", "ab")
            self.episodes = open(Path(log_dir) / f"env{env_id:03d}.episodes.csv", "a")
        self.rows = []
        self.total_steps = 0

    # ---- RAM ---------------------------------------------------------------------------------

    def _u32(self, mem, off):
        return int.from_bytes(mem[off:off + 4].tobytes(), "little")

    def _sb(self, ptr_addr):
        return self._u32(self.iwram, ptr_addr - 0x03000000) - 0x02000000

    def read(self):
        """Every RAM fact the env uses, in one place."""
        sb1, sb2 = self._sb(SAVEBLOCK1_PTR), self._sb(SAVEBLOCK2_PTR)
        ew = self.ewram
        x, y = np.frombuffer(ew[sb1 + SB1_POS:sb1 + SB1_POS + 4].tobytes(), np.int16)
        bank, num = ew[sb1 + SB1_LOCATION], ew[sb1 + SB1_LOCATION + 1]
        flags = ew[sb1 + SB1_FLAGS:sb1 + SB1_FLAGS + 300]
        n = min(int(ew[PARTY_COUNT - 0x02000000]), 6)
        party = ew[PARTY - 0x02000000:PARTY - 0x02000000 + 600].reshape(6, 100)[:n]
        levels = party[:, 84].astype(int)
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
            "options": self._u32(ew, sb2 + SB2_OPTIONS) & 0xFFFF,
        }

    # ---- gym API -----------------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.core.load_raw_state(ffi.from_buffer(self.state_bytes))
        self.core.run_frame()
        self.step_count = 0
        self.visits = {}  # (bank, num, x, y) -> times stood there
        self.frames = deque([self._small_screen()] * 3, maxlen=3)
        s = self.read()
        self.start = {"flags_set": s["flags_set"], "level_sum": int(s["levels"].sum()),
                      "options": s["options"], "badges": sum(s["badges"]), "towns": s["towns"]}
        self.best = {"flags": 0, "level": 0}
        self.stuck_total = 0.0
        self._visit(s)
        self.scores = self._scores(s)
        self.prev_total = sum(self.scores.values())
        if self.log:
            self.episodes.write(f"{self.total_steps},{self.init_state}\n")  # step offset where it began
        return self._obs(s), {}

    def step(self, action):
        self.core.set_keys(ACTIONS[ACTION_NAMES[action]])
        for _ in range(8):
            self.core.run_frame()
        self.core.set_keys()
        for _ in range(16):
            self.core.run_frame()
        self.frames.append(self._small_screen())

        s = self.read()
        if self.visits.get(self._key(s), 0) > 600:
            self.stuck_total += self.w["stuck"]
        self._visit(s)
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
            "explore": w["explore"] * len(self.visits),
            "level": w["level"] * self.best["level"],
            "badge": w["badge"] * (sum(s["badges"]) - self.start["badges"]),
            "town": w["town"] * (s["towns"] - self.start["towns"]),
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
        stats = np.zeros(22, np.float32)
        stats[:n] = s["levels"] / 100
        stats[6:6 + n] = s["hp"] / np.maximum(s["max_hp"], 1)
        stats[12:20] = s["badges"]
        stats[20] = s["options"] == self.start["options"]
        stats[21] = n / 6
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
                "map_bank": s["map"][0], "map_num": s["map"][1]}

    def render(self):
        return self.frame[:, :, :3].copy()

    def close(self):
        self.flush()
        if self.log:
            self.log.close()
            self.episodes.close()

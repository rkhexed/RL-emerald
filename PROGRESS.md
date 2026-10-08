# Training progress

One section per training run: what the policy and reward were, what we
observed, the results, and what changed for the next run. Times are US
Eastern. The reasoning behind each design choice is in `LEARNING.md`; the
raw data for every run is in `runs/<name>/` (logs, checkpoints, TensorBoard,
swarm states, videos).

## At a glance

| run | dates (EDT) | steps | games | key idea | furthest point |
|---|---|---|---|---|---|
| run01 | Oct 5 11:30 PM – Oct 6 12:41 AM | 2M | 16 | baseline, Red-style rewards | Route 103 (best games) |
| run02 | Oct 6 2:14 AM – 8:05 AM | 10M | 16 | weaker stuck penalty, stronger options penalty | beat May (1 game) |
| run03 | Oct 6 12:17 PM – 6:10 PM | 10M | 16 | milestones + swarming (Hamburg approach) | **inside Rustboro Gym** (15/16 milestones) |
| run04 | Oct 7 12:49 AM – 1:24 PM (stopped) | 29M | 24 | fixes from run03's analysis, evolve/catch rewards, blackout penalty | beat the Aqua grunt (12/16); **ran from battles** |
| run05 | Oct 7 2:26 PM – 6:50 PM | 10M | 24 | fight instead of flee: no blackout penalty, level ×4, seen/moves/heal rewards, global position input | Route 103; **grinding loop**, May never beaten |

| run06 | Oct 7 8:33 PM – (continues as the big run, +30M) | 8M + 30M | 24 | run05 with level reward back to 0.5 | in progress |

Every run starts from `states/01_mudkip.state` (Birch's lab, Mudkip Lv 5),
with 7 buttons (↑ ↓ ← → A B START), one decision every 24 frames, and
20,480-step episodes.

---

## run01: baseline

**Policy and training**
- Observation: 3 stacked 120×80 grayscale frames + a channel marking tiles
  visited this episode; 22 RAM numbers (levels, HP, badges, options OK, party size).
- PPO (stable-baselines3): 2,048 steps per game per update, minibatch 512,
  3 epochs, γ 0.997, λ 0.95, entropy 0.01, learning rate 3e-4.

**Reward**

| term | weight |
|---|---|
| event flags (best so far) | +1 per flag |
| new tiles this episode | +0.02 per tile |
| party levels gained | +0.5 per level (¼ after +15) |
| badges | +5 |
| towns visited | +2 |
| options changed (refundable) | −0.1 |
| stuck (tile visited 600+ times) | −0.025 per step |

**Observations**
- A random agent spends 99% of its time in Birch's lab.
- The stuck penalty dominated: −80.5 per episode against +2.6 for exploration.
- Every game ended every episode with the game options changed; −0.1 was too weak to matter.
- Bugs found and fixed during the run's tooling work: a PyTorch/mGBA deadlock
  (C→Python callbacks), and garbage RAM reads while Emerald moves its save blocks.

**Results** (2M steps, ~70 min)
- Tiles per episode: 130 → **557** average, 820 best.
- Stuck penalty: −80.5 → −26.
- Best games reached **Oldale Town and Route 103**; one spent 3,617 steps stuck in the Oldale Mart.
- Nobody went west to Route 102 (blocked until the Pokédex).

**Changed for run02:** options penalty 0.1 → 1.0; stuck penalty 0.025 → 0.01; 10M steps.

---

## run02: rebalanced penalties

**Policy and training:** as run01.
**Reward:** as run01, with options 1.0 and stuck 0.01.

**Observations**
- Exploration became the largest reward term (+26 vs −8 stuck); the policy became much less random (entropy −1.9 → −1.1).
- About 60% of games still ended episodes with options changed.
- Blackouts rose as agents went further: 0.16 → **1.67 per episode**.
- The gate: in Emerald, Route 102 is blocked until you beat May on Route 103
  and walk back to Birch's lab for the Pokédex. Walking back earned nothing
  (every tile already counted), and the agent couldn't tell "before May" from
  "after May".

**Results** (10M steps, 5.9 h)
- Tiles per episode: **1,324** average, 1,473 best.
- **All 16 games** reached Oldale and Route 103 in every one of the last 3 episodes.
- Mudkip Lv 8–13 at episode end.
- **One game beat May** (env 0, episode 29, step 10,391). **None got the Pokédex.**

**Research before run03:** other Emerald RL projects. Hamburg PokéRunners
(2nd place, NeurIPS 2025 PokéAgent Challenge, pure RL) used a **milestone
vector** as input for memory and goal-conditioning, and positive-only rewards
("all negative rewards… resulted in model degradation"). Red's original video
skipped its equivalent gate with a start state that already had the Pokédex.

**Changed for run03:** the Hamburg approach. Milestones + milestone vector +
swarming; stuck penalty off.

---

## run03: milestones and swarming

**Policy and training:** as before, plus a **16-bit milestone vector** in the
observation (38 RAM numbers in total).

**Reward**

| term | weight |
|---|---|
| event flags, tiles, levels, badges, towns | as run01 |
| **milestones** (official PokéAgent route + Pokémon Centers) | **+5 each, once** |
| options changed (refundable) | −1.0 |
| stuck | **0 (off)** |

**Swarming:** the first game to reach a milestone saves its state; later
episodes start from the furthest saved state (25% from a random earlier one).

**Observations**
- Swarming broke every gate. The Pokédex errand came **6 minutes** after the
  first win against May, after blocking run02 for 10M steps.
- The value network learned to predict returns much better (explained variance 0.7–0.94, up from ~0.5).
- Post-run analysis (replays classified with game-memory markers):
  - **76% of steps:** free to move but standing still (41% of those presses RIGHT into walls). Battles were only 2.3%.
  - Swarm states **inherited broken settings** (MID, then SLOW text from Petalburg Woods on).
  - Exploration **paid for walking backwards**: games starting in Rustboro walked back to Littleroot, because tiles counted per episode.
  - **Evolution cancelled:** still Mudkip at Lv 16–17 (B pressed during the evolution animation).
  - **Never caught a Pokémon**, despite carrying Poké Balls from Petalburg on.
  - Smaller traps: shop-menu loops; the double-battle twins on Route 104's bridge (needs 2 Pokémon).

**Results** (10M steps, 5.9 h)

| milestone | first reached | ≈ steps |
|---|---|---|
| Oldale, Route 103 | 1:16 PM | 1.7M |
| beat May | 2:42 PM | 4.1M |
| Pokédex | 2:48 PM | 4.3M |
| Route 102, Petalburg, met Dad | 2:51–3:13 PM | 4.3–5.0M |
| Route 104, Petalburg Woods, beat Aqua grunt | 3:36–3:45 PM | 5.6–5.9M |
| Rustboro City | 5:46 PM | 9.3M |
| Rustboro Pokémon Center, **entered the gym** | 5:57–5:58 PM | 9.6M |

- Best episode: **2,035 tiles**; whole run covered 4,485 tiles from Littleroot to Rustboro.
- Mudkip Lv 16–17. The gym attempt (env 12) got halfway, then **blacked out**.
- No Stone Badge.

**Changed for run04:** every issue from the analysis (below); a fresh start, not continued from run03.

---

## run04: fixes from run03's analysis

**Policy and training**
- Observation: as run03, plus **the last 4 buttons pressed**, **steps since
  last moved**, and an **in-battle flag** (68 RAM numbers).
- **24 games** on the resized 24-vCPU VM (~770 steps/s, up from ~470).

**Reward**

| term | weight | why |
|---|---|---|
| everything in run03 | same | |
| **evolution** | **+5 each** | Mudkip → Marshtomp was being cancelled |
| **catch** | **+2 per new party member** | never caught anything |
| **blackout** | **−1 each** | frequent fainting; watch battles per episode for timidity |

**Swarming fixes**
- Swarm states are saved with **our game settings** (the saving game's own episode is untouched).
- Swarm states carry the **tiles explored on the way there**, so only new ground pays.

**New logging:** every episode records the share of steps moving / in battle /
in dialogue or menu / idle, plus battles, blackouts, party size and
evolutions, all in TensorBoard.

**Targets to check:** idle share well below 76%; at least one evolution and
one catch; no collapse in battles per episode; the Stone Badge.

**Observations**
- Stopped early at 28.95M steps (1:24 PM): it had been stuck after the Aqua
  grunt for 6 hours. Speed was ~640 steps/s with 24 games (PPO's updates take
  longer with more games).
- What worked: idle share 76% → **~40–47%**; battle time 2.3% → **~20%**;
  episodes ending with broken settings ~94% → **~10%**; **first catches**
  (party up to 3, from ~18M steps).
- **The blackout penalty taught the agent to run from battles.** Replay
  analysis (`emerald_rl/analyze.py`, battle outcomes read from
  `gBattleOutcome`):

  | phase | battles/game | outcomes | lead level | blackouts |
  |---|---|---|---|---|
  | episode 15 (~7M) | 18.5 | **ran 100%** | **5 in all 8 games** | 0 |
  | episode 35 (~17M) | 28.9 | won 58%, ran 35%, lost 6%, caught 1% | 7–15 | 1.6 |
  | episode 56 (~28M) | 43.8 | **ran 63%**, won 33%, lost 4% | 9–14 | 1.6 |

  Running is the better deal even without the penalty: a battle costs dozens
  of steps and pays only a slow level reward, while walking pays exploration.
  With Mudkip at level 5, beating May took 5.4 hours (run03: ~2.4 h).
- **Too weak for the road north.** Mudkip ended around level 13 (run03 had
  16–17 here). Most blackouts happen in Petalburg Woods, respawning in
  Petalburg, back in the south. No Mudkip reached level 16, so **no evolutions**.
- **Going the wrong way.** After the grunt, 90% of Route 104 time was in the
  south half (unexplored beach tiles still pay); the closest any game came to
  Rustboro was y = 23 on Route 104, 7 tiles past the woods' north exit.

**Results** (28.95M steps, 12.6 h)

| milestone | first reached |
|---|---|
| Oldale, Route 103, Oldale Pokémon Center | 1:18–1:23 AM |
| beat May | 6:43 AM (5.4 h stuck at Route 103) |
| Pokédex → Route 102 → Petalburg → Dad → Route 104 → Woods | 6:46–7:11 AM |
| beat the Aqua grunt | 7:24 AM |
| Rustboro | not reached |

**Proposed for run05:** keep every run04 fix except the blackout penalty;
**remove the blackout penalty**, and make winning worth more than running
(a bigger level reward, so experience pays for the steps a battle costs).

---

## run05: fight instead of flee

A 10M-step diagnostic run, from Mudkip in the lab, 24 games. Everything from
run04 stays except the blackout penalty.

**Research before run05** (pokemonred_puffer's write-up and config, Hamburg):
- puffer on the level reward: "Any attempt to remove the reward led to a failed experiment."
- puffer paid for **new species seen** (+2.2), **new moves learned** (+4.1)
  and a **Pokémon Center heal** (+0.75); rewarding A presses made agents spam A.
- Hamburg fed the agent its **global position** (sinusoidal x, y + location)
  to avoid spatial ambiguity, and found negative rewards made agents timid
  (confirmed by run04).

**Reward changes**

| term | run04 | run05 |
|---|---|---|
| blackout | −1 | **0** |
| level gained | +0.5 | **+2** |
| new species seen | — | **+1** |
| new move known (distinct, ever) | — | **+2** |
| new place healed at (respawn point changes) | — | **+1** |

**Observation change:** the agent's **global Hoenn position** (sin/cos at 6
frequencies per axis, from `map_data.json`; 92 numbers in total).

**Targets to check:** battles won rather than fled (`emerald_rl/analyze.py`);
Mudkip reaches level 16 and **evolves**; moves learned; Rustboro.

**Observations**
- **Overcorrected into a grinding loop.** Battle share 20% → **60%** of
  steps, ~150 battles per episode (86% won), Mudkip Lv 15–17, but **5–7
  blackouts per episode**, nearly all on Route 101. The agents fight wild
  Pokémon until Mudkip faints; with no blackout penalty, fainting is a free
  heal, and they go back to grinding.
- The level reward became the largest term (~+20 per episode vs ~+10
  exploration), so going to fight May was never worth it.
- Good signs: **first evolution** (~5.4M steps), ~5 moves known, idle share
  down to ~23%.

**Results** (10M steps, 4.4 h): only Oldale, its Pokémon Center and Route
103; **May never beaten**.

**Lesson across run03–run05:** run03 (level 0.5, no penalty) fought when
needed and reached Rustboro; run04 (level 0.5, blackout −1) fled; run05
(level 2.0, no penalty) grinds. run04's fleeing came from the penalty, not
from the level reward being too small.

**Proposed for run06:** everything from run05, with the level reward back at
**0.5** (run03's proven value).

---

## run06: run05 with the level reward back at 0.5 (the big run)

Everything from run05, with the level reward at **0.5** (run03's value). An
8M-step diagnostic that then **continues from its own checkpoint and swarm
states for 30M more steps** (chained in tmux, so it doesn't depend on an
open session).

**4M-step check (10:18 PM):** battle behaviour fixed. Won 76%, ran 17%, lost
7%, ~27 battles per game (run04: ran 63–100%; run05: 150 battles, grinding).
Mudkip mostly Lv 6–7 (best 12–14); blackouts ~2 per game while levelling on
Route 101. May not beaten yet (run03 beat her at ~4.1M). Still open: idle
share ~53% and settings broken in ~85% of episodes.

**Reward hack found (Oct 8, 4 PM, run06c at ~41M steps).** The moves reward
was being farmed. Mean distinct moves known rose from ~3 early in run06 to
**25.8**, the moves term reached **~41 per episode** (exploration ~7), and
the worst game spent **96%** of its steps locked in menus. A replay (env 3,
latest episode) showed all 33 extra "moves" appearing on the party menu: the
agents switch Pokémon around, the env sometimes reads a half-copied slot, and
the junk decodes to move IDs like 48573 (real ones stop at 354). Fixed in
`env.py` (commit `3676252`): only Pokémon whose checksum matches are read;
the same replay now counts 6 moves, not 39. The live run keeps the old code.
This also revises the earlier diagnosis: the ~71% of time locked in menus
was mostly this exploit, not weak leads.

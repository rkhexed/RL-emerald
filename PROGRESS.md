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
| run04 | Oct 7 12:49 AM – (ETA 11:40 AM) | 30M | 24 | fixes from run03's analysis, evolve/catch rewards | in progress |

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

## run04: fixes from run03's analysis (in progress)

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

**Observations:** _to be filled in._
**Results:** _to be filled in._

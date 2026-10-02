# RL-Emerald

Reinforcement learning agents that play **Pokémon Emerald**, with a comparison
against LLM-guided play.

Side project, separate from the drone navigation research. Written 2026-10-01.
Every number and claim below is checked against the cited paper, repository or
GitHub API — guesses are marked as such.

---

## 1. The goal

Train an agent to progress through Pokémon Emerald, then compare three families
on equal terms:

1. **Pure RL** — PPO over emulator frames, in the style of PokemonRedExperiments.
2. **LLM-guided** — a frontier model driving the game through a tool harness.
3. **Hybrid** — an LLM proposing subgoals, RL executing them.

The honest framing: this comparison has already been run (see §2), so the
contribution has to be something else. The open angles are in §6.

---

## 2. What already exists — read this before building anything

| Work | What it did | Status |
|---|---|---|
| **PokemonRedExperiments** (Whiddy) | PPO on Pokémon **Red** via PyBoy; the viral 2023 video | 7.9k stars, MIT, active |
| **Pokémon Red via RL** (Pleines et al., arXiv 2502.19920) | Formal write-up; PPO baseline reaching Cerulean City; reward-hacking ablations | published |
| **PokeRL** (arXiv 2604.10812, Apr 2026) | Red early game, anti-loop wrappers, dense rewards | preprint |
| **PokéAgent Challenge** (NeurIPS 2025) | **An entire Emerald speedrunning track.** 100+ teams, 22 valid submissions, 6 reached 100% | public infra + leaderboard |
| **Continual Harness** (arXiv 2605.09998, May 2026) | Self-improving LLM harness on Red *and* Emerald | code public |
| pygba, PkmnRLArena, Emeraude_IA_RL | Emerald RL environments (mGBA wrappers) | small, 0–22 stars |

### The competition results that matter

| Team | Approach | Result |
|---|---|---|
| Heatz (1st) | LLM writes tool-using policy code, distilled into a neural policy (DAgger) | **40:13** |
| Hamburg PokéRunners (2nd) | **Pure RL** — recurrent PPO, milestone-conditioned rewards | ~80 min |
| Anthonys (3rd) | Pure LLM harness | 1:29:17 |
| Frontier VLMs, no harness | — | **~0% completion** |
| Human expert speedrunner | — | 18 min |

So: hybrid beats either alone, and raw vision-language models are useless without
scaffolding. Re-running that comparison would be reproduction, not contribution.

**Cost of LLM play:** Continual Harness reports **$130 per complete Emerald run**
with Gemini 3 Pro ($215 for their minimal-harness baseline). Weaker models fail
outright — Flash-Lite stalls below 20%.

---

## 3. Compute, with real numbers

From Pleines et al., measured on an AMD Ryzen 7 2700X:

* Pokémon **Red** runs at **9,403 frames/sec per core**, but one decision takes
  24 frames (8 held, 16 released), so **~392 decisions/sec per core**.
* Their runs: 32 workers, **400M steps, ~36 hours per run**.
* The recurrent (GRU) variant did not fit in VRAM, so it ran on CPU: **24 days
  per run**.
* Evaluating a trained agent costs about as much wall time as training it.

**Emerald is GBA, which is heavier.** Headless mGBA is reported at 1,000–3,400 fps
(unverified by us), so roughly **40–140 decisions/sec per core** — 3–10× slower
than Red.

Extrapolated (estimates, not measurements):

| vCPUs | One 400M-step run | Cost at ~$2/hr |
|---|---|---|
| 16 | ~4–14 days | $200–650 |
| 32 | ~2–7 days | $100–350 |
| 48 | ~1.5–5 days | $70–250 |

**No GPU.** The network is a small CNN over downsampled frames; the bottleneck is
emulation, which is CPU-only. A GPU would sit idle and bill.

**Scope reduction worth considering:** the competition scores progress to the
first gym (Roxanne), which plausibly needs ~100M steps rather than 400M — about a
quarter of the above.

---

## 4. VM specification

| Field | Value |
|---|---|
| Machine family | General purpose |
| Series | **C4** (or **N2** if C4 capacity is short) |
| Machine type | **`c4-highcpu-16`** to start (16 vCPU, 32 GB) |
| GPU | **none** |
| Boot disk | 100 GB balanced persistent disk |
| Image | Ubuntu 22.04 LTS |
| Region | us-central1 |
| Provisioning | Standard (not Spot) for runs you need to finish |
| Rough cost | ~$0.6–0.7/hr running; pennies when stopped |

### The quota catch — read this first

The approved quota is **64 CPUs globally** and **48 per C4 family in
us-central1**. The drone-navigation VM already uses **48**, leaving **16**.

So, three options:

1. **Start at 16 vCPU now** (recommended): enough to install everything, measure
   the real emulator speed, and train the scoped-down first-gym target.
2. **Raise the global CPU quota** to ~112 if both projects need to run at once.
3. **Wait and reuse the drone VM** once its runs finish — same hardware, no new
   quota.

### Why many cores rather than a big GPU

Each emulator instance is a single-threaded process using a few hundred MB. More
cores means more parallel games. 16 vCPUs supports roughly 16–32 instances; RAM is
not the constraint (32 GB is ample).

---

## 5. ROM legality

Every Emerald environment requires a `rom.gba` that **you supply**. You need to
own the cartridge, and the ROM must never be committed to a repository or copied
to a shared bucket. Keep it on the VM disk only, and leave it out of any
screenshots or published artefacts.

---

## 6. Where the novelty could be

These survive the prior work:

1. **A compute-matched cost comparison.** Everyone reports wall-clock or dollars
   for LLM agents and environment steps for RL, but nobody puts them on one axis:
   *dollars and FLOPs to each milestone*, for RL vs LLM vs hybrid. No new
   algorithm needed, and it is exactly what a small lab can do well.
2. **Distil the harness into something free to run.** The winner went LLM →
   scripted → RL. Nobody has pushed it to "a small recurrent policy that
   completes Emerald with zero API calls at inference". That is a deployment-cost
   result with a hard number attached.
3. **LLM-written reward functions vs hand-written milestone rewards**
   (Eureka-style). Pleines et al. found agents exploiting shaped rewards; testing
   whether LLM-authored rewards hack *more or less* extends that directly.
4. **Cross-game transfer**: train on Red (cheap), transfer to Emerald
   (expensive). No RL transfer between the two has been published.
5. **Fix a documented limitation**: recurrent policies choked at 2048-step
   horizons. A state-space model or hierarchical options agent addresses a
   problem the authors named in print.

Best pair for a small project: **2 + 1** — distil a hybrid agent into an API-free
policy, and report the compute-matched Pareto curve. Neither requires beating the
leaderboard.

---

## 7. First steps, in order

1. Create the VM (§4) and install **pygba** (mGBA with Python bindings) plus a
   Gymnasium wrapper.
2. **Measure the real steps-per-second.** This replaces the single biggest guess
   in §3 and determines every later estimate.
3. Reproduce a short RL run on the scoped target (first gym) to validate the
   pipeline.
4. Run one LLM harness pass for the comparison baseline, budgeting ~$130 of API
   credit.
5. Only then pick a novelty angle from §6 and commit to it.

---

## 8. References

* Pokémon Red via RL — https://arxiv.org/abs/2502.19920
* PokemonRedExperiments — https://github.com/PWhiddy/PokemonRedExperiments
* PokeRL — https://arxiv.org/abs/2604.10812
* PokéAgent Challenge — https://arxiv.org/abs/2603.15563 · https://pokeagentchallenge.com
* Winning speedrun solution — https://github.com/heatz123/pokeagent-solution
* Continual Harness — https://arxiv.org/abs/2605.09998 · https://github.com/sethkarten/continual-harness
* pygba (mGBA + Gymnasium) — https://github.com/dvruette/pygba
* Emerald RL experiments — https://github.com/dvruette/pokemon-emerald-experiments
* PkmnRLArena (PettingZoo, GBA) — https://github.com/wissammm/PkmnRLArena

---

## 9. Not verified yet

* The 1,000–3,400 fps headless mGBA figure — ours to measure in step 2.
* Whether the competition's Emerald environment can be reused directly, or
  whether a custom wrapper is needed.
* Whether a scoped first-gym target really needs ~100M steps rather than 400M.

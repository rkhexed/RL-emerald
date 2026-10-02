# Deep research: recreating PokemonRedExperiments for Pokémon Emerald

Run 2026-10-02 with the `deep-research` workflow: 5 search angles, 19 sources fetched, 87 claims extracted, 25 checked by 3 independent verifiers each (a claim is dropped if 2 of 3 refute it). 23 claims held up and 2 were refuted. After merging duplicates, 12 findings remain.

Confidence: **high** means the verifiers agreed unanimously on primary sources. **medium** means a single unreviewed source, or a design proposal of mine built from verified parts.

See also [Code-verified addendum](#code-verified-addendum-2026-10-02) at the bottom. Those points were checked directly against cloned source code and on our VM after the research ran.

## The question

> Research how to recreate PWhiddy's PokemonRedExperiments (PPO on Pokémon Red via PyBoy, viral 2023 video, github.com/PWhiddy/PokemonRedExperiments) for Pokémon EMERALD (GBA), pure RL focus (no paid LLM APIs; everything free/open-source; CPU-only 16 vCPU Ubuntu VM, scalable to 48). Cover:
> 1) PokemonRedExperiments deep dive: repo structure (baselines/ v2 env, red_gym_env, memory addresses, reward function components—exploration via KNN/coordinate novelty, level, badges, healing, events, death penalty), training loop (stable-baselines3 PPO, SubprocVecEnv, n_steps, batch size, savestate starts, episode length), and how the iconic visuals were made: the overlaid map of many agents' trajectories, the 'collated replays' grid of many simultaneous games, the live shared-map website (pwhiddy map visualizer / pokerl map), wandb/tensorboard stats.
> 2) PPO fundamentals and math as used there: clipped surrogate objective, GAE advantage, value loss, entropy bonus, key hyperparameters (gamma, lambda, clip, n_epochs), why recurrent/frame-stacking, and what to learn from replays (reward hacking, getting stuck, loops).
> 3) Emerald emulator environments: pygba (dvruette/pygba, mGBA Python bindings + Gymnasium), dvruette/pokemon-emerald-experiments, PokéAgent Challenge (NeurIPS 2025) Emerald infrastructure and the Hamburg PokéRunners pure-RL recurrent PPO solution, PkmnRLArena, Emeraude_IA_RL, PyBoy vs mGBA throughput (headless mGBA fps, decisions/sec per core), Emerald RAM/memory map (pokeemerald decomp, DMA-protected/save-block pointer shuffling in Emerald, player x/y/map bank+number, badges flags, party data encryption), savestates, frame skip.
> 4) Reward design and anti-stuck techniques from follow-up work: Pleines et al. arXiv 2502.19920 (Pokémon Red via RL, reward-hacking ablations), PokeRL arXiv 2604.10812 (anti-loop wrappers, dense rewards), drubinstein/pokemonred_puffer (PufferLib-based successor that beat Red; swarm/savestate sharing, event rewards, cut/surf scripting), PufferLib as a faster alternative to SB3, novelty/exploration bonuses, milestone-conditioned rewards, episode resets from checkpoints.
> 5) Free visualization and experiment tracking tooling: recording mp4/gif per env, collated grid replays, map trajectory overlay for Hoenn (where to get a free Hoenn full map image/coordinates from pokeemerald map data), tensorboard vs wandb free tier, live streaming the training map.
> Output a recommended repo structure and training method for an Emerald version that improves on Red's, plus a curated learning path (papers, videos, docs) to deeply understand PPO and the pipeline for making an explanatory YouTube video.

## Summary

PWhiddy's PokemonRedExperiments is a PPO agent (policy plus value network, stable-baselines3) that plays Pokemon Red in PyBoy. Its recommended V2 swaps the original frame-KNN novelty reward for a coordinate-based exploration reward, trains faster, reaches Cerulean, and streams every env's (x, y, map) to a shared live map, pokerl-map-viz, through a gym wrapper that sends JSON over a websocket. For Emerald, the closest free starting point is dvruette/pygba, a Python wrapper around mGBA with Gymnasium support and a built-in Emerald wrapper whose reward weights you can set. Its sister repo, dvruette/pokemon-emerald-experiments, is a direct fork of the Red repo. It reached the second gym, then got stuck farming event reward on Mr. Briney's boat, which is a clear case of reward hacking. Later work shows how to do better: the PufferLib-based pokemonred_puffer, which beat Red with a policy of about 5M parameters, uses an LSTM instead of frame stacking, dense shaping with 25+ terms, a modular config-driven repo, and "swarming". Swarming means sharing save states whenever any agent hits a milestone, an idea inspired by Go-Explore. In the PokeAgent Challenge, a pure-RL recurrent PPO with milestone-conditioned rewards (Hamburg PokeRunners) finished the Emerald route to Roxanne in second place. The recommended plan: a pygba/mGBA env with a pokemonred_puffer-style modular layout, recurrent PPO, milestone-conditioned and anti-loop rewards, swarm savestates, and a Hoenn version of the shared map and replay-grid visuals, all built on free tools (tensorboard or wandb free tier, plus self-hosted map-viz).

## Findings

### 1. (high, vote 3-0)

Red baseline to recreate: V2 of PokemonRedExperiments is the recommended version. It replaces the frame-KNN exploration reward with a coordinate-based exploration reward, trains faster with less memory, reaches Cerulean, and is a PPO agent made of a policy network and a value network. Use V2, not the original baselines version, as the reference design.

*Evidence:* Unanimous 3-0 on the primary README: 'V2 is now recommended... Trains faster and with less memory... Reaches Cerulean... Replaces the frame KNN with a coordinate based exploration reward'. The policy/value description comes from the README of the fork.

*Sources:* <https://github.com/PWhiddy/PokemonRedExperiments> · <https://github.com/dvruette/pokemon-emerald-experiments>

### 2. (high, vote 3-0 (merged claims 1,2,3,4,5))

How the shared-map visual works: each training env is wrapped in StreamWrapper (gym.Wrapper, baselines/stream_agent_wrapper.py). On every step it reads player X (0xD362), Y (0xD361) and map number (0xD35E) from Red RAM and appends [x, y, map]. Every 300 steps it sends the batch as JSON over a websocket to wss://transdimensional.xyz/broadcast, and pokerl-map-viz renders it. Metadata is optional: user, env_id, color (hex), extra text, sprite_id (0-50). V2 streams by default. For Emerald you would replace the RAM reads with Emerald's coordinate and map bank/number reads, and self-host map-viz with a Hoenn map.

*Evidence:* Primary code checked directly: class StreamWrapper(gym.Wrapper), ws_address, the RAM constants, upload_interval=300, and a silent failure when the socket drops. The live demo at pwhiddy.github.io/pokerl-map-viz loads (showed 0 envs streaming when checked). Its title and coordinate system are specific to Red.

*Sources:* <https://github.com/PWhiddy/PokemonRedExperiments/blob/master/baselines/stream_agent_wrapper.py> · <https://github.com/PWhiddy/PokemonRedExperiments> · <https://github.com/PWhiddy/pokerl-map-viz>

### 3. (high, vote 3-0 / 2-1 (merged claims 6,7,8))

Emerald emulator base: dvruette/pygba wraps mGBA with Gymnasium support. Emerald is its only game wrapper: PyGBAEnv(PyGBA.load(rom, autoload_save=True), PokemonEmerald()). Reward weights are constructor arguments: badge 10, champion 100, visit_city 5, money 0, seen 0.2, caught 1, trainer_beat 1, event 1, exp_scale 0.1. In practice you build mGBA from source with -DBUILD_PYTHON=ON, because the prebuilt mGBA wheels are marked broken.

*Evidence:* The pygba README and src/pygba/game_wrappers/pokemon_emerald.py (constructor near line 164) were checked. The pokemon-emerald-experiments README says pygba 'can be a bit of a pain... requires compiling mGBA from source'. The project is small (about 33 stars), so maintenance is uncertain.

*Sources:* <https://github.com/dvruette/pygba> · <https://github.com/dvruette/pokemon-emerald-experiments>

### 4. (high, vote 3-0 (merged claims 9,10))

Prior Emerald attempt and its failure mode: dvruette/pokemon-emerald-experiments is a fork of the Red repo and keeps the baselines/ layout (run_baseline_parallel_fast.py). Its best checkpoint (2023-12-03) reached the second gym, then got stuck farming event reward by riding Mr. Briney's boat between Dewford and Petalburg. Event rewards must count only once and be tied to milestones, or the agent will loop on them.

*Evidence:* The README quote: 'able to reach the second Gym but is currently stuck exploiting the reward function by travelling with Mr. Briney between Dewford Town and Petalburg City (farming event reward)'. The result is from 2023 and has not been updated.

*Sources:* <https://github.com/dvruette/pokemon-emerald-experiments>

### 5. (high, vote 3-0 (merged claims 17,18))

The best repo template is pokemonred_puffer. It is a modular package with policies/, rewards/ (gym.Env reward trackers), wrappers/, cleanrl_puffer.py (training loop), environment.py (core env), eval.py (visualizations), train.py and a high-resolution region map PNG. config.yaml picks rewards, policies and wrappers by module_name.class_name, and wrappers apply top to bottom. You can swap and tune reward variants without editing code; only new classes need Python. An Emerald repo should copy this layout, with a hoenn_map.png.

*Evidence:* The README lists each module word for word along with the config keying rules. The repo also holds cut-config.yaml, sweep-config.yaml and pyboy_states/. The README says it is meant to 'eventually' be a library.

*Sources:* <https://github.com/drubinstein/pokemonred_puffer>

### 6. (high, vote 3-0 (merged claims 13,14))

Training method that beat Red: PPO with an LSTM (size 128, 1 cell, after a 512-unit linear layer over CNN screen features and embeddings). The authors chose this over frame stacking because stacked input grows linearly and slows training. The policy is about 5M parameters (20MB), starts from random button presses, and has no pretraining. Beating the game as of February 2025 with 'minimal simplifications' still involved scripted HM use such as cut and surf.

*Evidence:* Primary write-up by Rubinstein, Donovan, Addis, Choe, Suarez and Whidden: 'We went with the easiest to integrate solution, an LSTM... ≈5M parameters or 20MB'.

*Sources:* <https://drubinstein.github.io/pokerl/> · <https://drubinstein.github.io/pokerl/docs/chapter-2/policy/>

### 7. (high, vote 3-0 (claim 15))

Anti-stuck technique 'swarming', inspired by Go-Explore: whenever any agent completes a required objective, its save state is recorded and every agent loads it. This keeps agents close together in time and space, which keeps training data stable. The tracker resets when the game is won so the policy still generalizes. This maps directly onto Emerald milestones: Littleroot, starter, Petalburg, Rustboro, Roxanne, and so on.

*Evidence:* Source text: 'Every time an agent completed a required objective, every agent loads the save state from the agent that completed the new objective... We dubbed this method swarming.' It is presented as a fix for data stability, alongside scripted HMs and event rewards.

*Sources:* <https://drubinstein.github.io/pokerl/docs/chapter-3/swarm/>

### 8. (high, vote 3-0 (merged claims 16,19))

Reward design needs dense shaping, but every term invites hacking. pokemonred_puffer uses more than 25 weighted terms: exploration, events, badges, levels, seen and caught, HM count, cut/surf tiles, menus, signs, warps, heals, required events and items. Its authors say sparse rewards alone would leave the agent stuck in Pallet Town. Pleines et al.'s ablations show the risks. A heal reward led a Bulbasaur agent to loop on Zubats in Mt. Moon, with Bulbasaur's Leech Seed against Zubat's Leech Life. Removing the level reward slightly raised milestones (4.60±0.7 vs 4.19±0.2). Removing the navigation reward gave 0 milestones.

*Evidence:* Both are primary sources. The ablation gain from removing the level reward is high-variance (±0.7).

*Sources:* <https://drubinstein.github.io/pokerl/docs/chapter-2/rewards/> · <https://arxiv.org/abs/2502.19920>

### 9. (high, vote 3-0 (claim 20))

Concrete env design from Pleines et al.: each action holds a button for 8 frames and releases it for 16, so the agent decides once every 24 frames. The observation is a 72x80 grayscale screen stacked over 3 frames, plus a 48x48 binary crop of visited tiles centered on the player and a game-state vector (HP, levels, event flags). Actions are 7 buttons, without Select. A GBA env (240x160) would scale the screen accordingly and might add L/R.

*Evidence:* Quoted from the paper's environment spec. This is their setup and is not guaranteed to match PWhiddy's code.

*Sources:* <https://arxiv.org/abs/2502.19920>

### 10. (medium, vote 3-0 (merged claims 21,22))

Anti-loop wrappers and a visited-mask channel help, according to a preprint. PokeRL's three layers are a tile-visit penalty after 3 visits, detection of repeated patterns in a sliding window of the last 20 actions, and detection of position loops. Together they reduced loop episodes from 41.2% to 4.7%. Adding a 72x80 per-map visited-mask channel raised unique positions per episode from 34.2 to 48.1 and cut the revisit ratio from 4.8 to 3.1.

*Evidence:* Single unreviewed 2026 preprint. The visited-mask test ran only 300k timesteps with no seeds or variance reported. The loop metric is partly circular, because it uses the same criteria the penalties target.

*Sources:* <https://arxiv.org/html/2604.10812v1>

### 11. (high, vote 3-0 (merged claims 11,12))

Emerald benchmark context: the PokeAgent Challenge speedrun track runs Emerald on a GBA emulator with 15 standardized milestones, from Littleroot to defeating Roxanne. It ranks by completion percentage, then time, with ties broken by action count. Hamburg PokeRunners placed second with pure recurrent PPO and milestone-conditioned rewards, taking about twice as long as the winner. The winner, Heatz, used scripted policy distillation and finished in 40:13. This is the most relevant pure-RL Emerald precedent and gives a ready-made milestone list for rewards and swarm checkpoints.

*Evidence:* From the organizers' paper. 6 of 22 valid teams reached 100%. The PokeRunners' exact time and code were not retrieved.

*Sources:* <https://arxiv.org/html/2603.15563v1>

### 12. (medium, vote synthesis)

Recommended Emerald repo and training method, synthesized from the findings above. emerald_rl/ package: env/ (pygba/mGBA core, RAM readers for coords, map bank/number, badges and flags, party), rewards/ (pluggable classes: coordinate novelty per (map_bank, map_num, x, y), one-time milestone and event rewards, level with a cap, no raw heal reward), wrappers/ (anti-loop, visited-mask obs, StreamWrapper for Hoenn, per-env mp4 recorder), policies/ (CNN plus LSTM), train.py with config.yaml (puffer-style), states/ (savestates per milestone, using swarming), viz/ (trajectory overlay on a Hoenn map, collated grid video), runs/ (checkpoints, tensorboard, videos, coordinate logs kept for the YouTube video).

*Evidence:* This is a design recommendation built from verified components. It has not been tested end to end on Emerald.

*Sources:* <https://github.com/drubinstein/pokemonred_puffer> · <https://github.com/dvruette/pygba> · <https://drubinstein.github.io/pokerl/> · <https://arxiv.org/html/2603.15563v1>

## Refuted claims (do not rely on these)

- ~~Training is launched via baseline_fast.py (baselines) or baseline_fast_v2.py (v2), with progress tracked in tensorboard and optional wandb logging toggled by a use_wandb_logging flag.~~ (vote 0-3, <https://github.com/PWhiddy/PokemonRedExperiments>)
- ~~The baseline PPO agent's reward is the sum of four parts: +2 per completed event, +0.005 per newly visited overworld coordinate, a healing reward of 2.5 times the fraction of HP restored, and a level reward with coefficient 0.5 that is capped at a level threshold of 22.~~ (vote 0-3, <https://arxiv.org/abs/2502.19920>)

## Caveats

Several parts of the research question have no verified claims behind them, so these gaps need direct reading of code and docs: PPO math (clipped surrogate, GAE, value and entropy terms, SB3 hyperparameters such as n_steps, batch size, gamma and GAE lambda), PyBoy vs mGBA throughput and steps per second per core, Emerald RAM details (save-block pointer shuffling via gSaveBlock pointers, encrypted party data, map bank/number addresses from the pokeemerald decomp), where to get a Hoenn map image and coordinates, PufferLib vs SB3 speed, and how the collated replay grid video is made. Two claims were refuted and should not be relied on: (a) the exact launch script names and the use_wandb_logging flag, and (b) the exact reward coefficients attributed to Pleines et al. Check the actual baseline_fast_v2.py and red_gym_env_v2.py directly. pygba is small, building mGBA's Python bindings from source can be fragile, and its prebuilt wheels are marked broken. PWhiddy's broadcast server and the map-viz demo are Red-specific, and their uptime is not guaranteed, so plan to self-host. PokeRL (2604.10812) is an unreviewed small-scale preprint. Mr. Briney's boat actually departs from Route 104, not Petalburg City as the README says. The dvruette Emerald results date from 2023. The PokeAgent Challenge results date from 2025-2026, and the PokeRunners code was not located.

## Open questions (as of the research run)

- How many env steps per second does headless mGBA via pygba reach per core compared with PyBoy, and how many parallel envs can 16 to 48 vCPUs sustain under PPO with an LSTM?
- Which Emerald RAM addresses are stable for player x/y, map bank/number, badges and event flags, given the DMA-protected and shuffled save-block pointers? Are pygba's emerald_utils readers correct and complete?
- Is the Hamburg PokeRunners' recurrent PPO code public, and what milestone-conditioned reward schedule and hyperparameters did they use?
- What is the best free source for a stitched Hoenn world map with per-map pixel offsets (for example, generated from pokeemerald's map layouts and connections) to support trajectory overlays and a Hoenn version of pokerl-map-viz?

## Sources

| Source | Quality | Angle |
|---|---|---|
| <https://github.com/PWhiddy/PokemonRedExperiments> | primary | PokemonRedExperiments internals and visuals |
| <https://github.com/PWhiddy/PokemonRedExperiments/blob/master/baselines/stream_agent_wrapper.py> | primary | PokemonRedExperiments internals and visuals |
| <https://github.com/PWhiddy/pokerl-map-viz> | primary | PokemonRedExperiments internals and visuals |
| <https://github.com/dvruette/pygba> | primary | Emerald emulator environments |
| <https://github.com/dvruette/pokemon-emerald-experiments> | primary | Emerald emulator environments |
| <https://arxiv.org/html/2603.15563v1> | primary | Emerald emulator environments |
| <https://github.com/pret/pokeemerald/pull/2364> | primary | Emerald emulator environments |
| <https://bulbapedia.bulbagarden.net/wiki/Pok%C3%A9mon_data_structure_(Generation_III)> | secondary | Emerald emulator environments |
| <https://deepwiki.com/pret/pokeemerald> | unreliable | Emerald emulator environments |
| <https://drubinstein.github.io/pokerl/> | primary | Reward design and anti-stuck follow-up work |
| <https://github.com/drubinstein/pokemonred_puffer> | primary | Reward design and anti-stuck follow-up work |
| <https://arxiv.org/abs/2502.19920> | primary | Reward design and anti-stuck follow-up work |
| <https://arxiv.org/html/2604.10812v1> | primary | Reward design and anti-stuck follow-up work |
| <https://github.com/drubinstein/pokemonred_puffer/tree/main/pyboy_states> | primary | Reward design and anti-stuck follow-up work |
| <https://github.com/amcheste/pokemon-red-ai> | blog | Reward design and anti-stuck follow-up work |
| <https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/> | primary | PPO math and theory |
| <https://arxiv.org/pdf/1707.06347> | primary | PPO math and theory |
| <https://arxiv.org/pdf/1506.02438> | primary | PPO math and theory |
| <https://arxiv.org/pdf/2205.11104> | primary | PPO math and theory |

---

## Code-verified addendum (2026-10-02)

Checked after the research run against shallow clones in `~/refs/` and by running code on the `e-rl` VM. Where this disagrees with a finding above, this section wins.

### Corrections to the research

- **Refuted claim (a) is actually true.** `v2/baseline_fast_v2.py` exists, logs to tensorboard, and has `use_wandb_logging = False` at the top. The verifiers were wrong to kill it.
- **Red V2's real reward weights**, from `v2/red_gym_env_v2.py::get_game_state_reward`, each multiplied by `reward_scale = 0.5`:
  `event 4 × (max event flags set − flags at start)` · `heal 10 × Σ(heal fraction²)` · `badge 10 × badges` · `explore 0.25 × 0.1 × unique (x,y,map) seen` · `stuck −0.05 while standing on a tile visited > 600 times`. The level, opponent-level and death terms are commented out.
- **Red V2's PPO settings** (`baseline_fast_v2.py`): 64 envs, `ep_length = 2048 × 80 = 163,840` steps, `n_steps = ep_length // 64 = 2,560` per env per rollout (163,840 transitions per update), `batch_size 512`, `n_epochs 1`, `gamma 0.997`, `ent_coef 0.01`. Everything else is the SB3 default: `lr 3e-4`, `gae_lambda 0.95`, `clip 0.2`, `vf_coef 0.5`, `max_grad_norm 0.5`, normalized advantages.
- **Red V2 observation** (a Dict): 3 stacked 72×80 grayscale screens · HP fraction · level sum, Fourier-encoded into 8 sines · 8 badge bits · all event-flag bits · a 48×48 crop of the visited-tiles map centred on the player · the last 3 actions. 7 actions (no Select). Decisions are 24 frames apart, with the button held for 8.

### pygba / Emerald wrapper: problems found in the code

1. **The event reward is farmable by design.** `PokemonEmerald.reward` adds `popcount(old_flags XOR new_flags)` to a running total, so it counts a flag being *cleared* as well as set, and never caps it. Temporary script flags that flip on and off during a cutscene pay out again every time. This is the most likely cause of dvruette's Mr. Briney boat loop. Red V2 does it correctly: `max(events_set_now − events_set_at_start)`, which can only go up.
2. **It's slow.** Every step it reads and decrypts every PC box (14 × 30 mons), loops over all 411 species reading ROM tables, and `PyGBA._get_memory_region` copies a whole memory region (ROM is up to 32 MB) the first time it's read in each frame.
3. **Grayscale observations crash.** `_get_observation` calls `.transpose(1, 0, 2)` on a 2-D grayscale array.
4. **The action space is 35 combinations** (5 arrows × 7 buttons, including Select/L/R). Red uses 7.
5. **What it gets right:** it reaches SaveBlock1/2 through the `gSaveBlock1Ptr`/`gSaveBlock2Ptr` pointers (`0x03005d8c`/`0x03005d90`). That's the correct way to cope with Emerald moving its save blocks around (DMA protection). Its party decryption follows Bulbapedia's Gen III structure: XOR with `otId ^ personality`, then pick one of 24 substructure orders by `personality % 24`.

### mGBA build and speed on our VM

- The `mgba` pip wheel is still broken: it needs `libmgba.so.0.10`. Building mGBA **0.10.5** with `-DBUILD_PYTHON=ON` works, but the Python bindings also need **`-DUSE_FFMPEG=ON`**. Without FFmpeg, `EReaderScanLoadImageA` is never compiled, and `import mgba` fails with an undefined symbol. Full steps are in `SETUP.md`.
- `scripts/bench_emulator.py`, run on MIT-licensed test ROMs (jsmolka/gba-tests), not Emerald:

  | processes | raw fps per process | decisions/s per process (24 frames each) | decisions/s total |
  |---|---|---|---|
  | 1 | 10,577 | 404 | 404 |
  | 8 | 10,188 | 388 | 3,102 |
  | 16 | 5,249 | 204 | 3,271 |

  The 16 vCPUs are **8 physical cores × 2 hyperthreads** (Xeon Platinum 8581C), so total throughput plateaus at 8 processes. The test ROMs mostly sit idle, so **treat these as an upper bound**; Emerald does more work per frame. The README §3 guess (40–140 decisions/s per core) looks pessimistic, but this can only be confirmed with Emerald itself.
- **Real Emerald, measured later the same day from the Littleroot overworld:** 2,481 fps and 96 decisions/s for 1 process, 737 total at 8, **900 total at 16**. Emerald is about 4× heavier than the idle test ROMs, so the README §3 range (40–140 per core) was about right, not pessimistic. Hyperthreads add ~22%. pygba's `PokemonEmerald.reward()` costs 2.5 ms per call (about +25% per step).
- Savestates: save about 14 µs, load about 17 µs, about 388 KiB each, so swarming (everyone loading the best agent's state) costs nothing measurable.

### Hoenn shared map: open question resolved

`scripts/build_hoenn_map_data.py` builds global coordinates from the pokeemerald decomp **with no ROM needed**. It walks each map's `connections` outward from Littleroot and anchors indoor maps at the door that leads into them. Result: 518 maps, 49 outdoor maps stitched into an 800×383-tile world, 254 indoor maps anchored, and every map from Littleroot to Rustboro Gym placed. Map (bank, num) comes from the order in `map_groups.json`, which matches what the game stores in `SaveBlock1.location`. Known gaps: 6 of Hoenn's own connections disagree by 2 tiles; maps reached by dive, ferry or scripted warps (Sootopolis, underwater routes, Battle Frontier) aren't placed yet; and big maps like Petalburg Woods currently collapse to their entrance tile.

### Visual pipeline, confirmed in code

- **Collated replay grid**: `baselines/tile_vids_to_grid.py` builds one ffmpeg `xstack` command. It tiles 40 per-env mp4s into an 8×5 grid, then tiles those grids into an 8×8 "big_boi" mosaic.
- **Overlaid sprites on the map**: `visualization/BetterMapVis_script_version.py` loads per-env (x, y, map) logs, maps each to pixels with *hand-written* per-map offsets, and draws a 16×16 walking sprite (direction from dx/dy) with 8 interpolated sub-steps per move into a transparent ProRes `.mov`, which gets composited over the map image afterwards.
- **Live shared map**: `pokerl-map-viz` (MIT) is a static PIXI.js page plus `ws-server/index.js`, a small Node websocket relay. Both can be self-hosted on the VM for free; swap in a Hoenn background and our `map_data.json`.
- **pokemonred_puffer's version**: the `CoordinatesWriter` wrapper writes CSV every 1,000 steps, and `visualizations/coordinates_replay.py` renders them. Its `DecayWrapper` slowly decays the "seen" values (coordinates × 0.9995 every 10 steps, floor 0.15), so the exploration reward comes back over time instead of drying up.

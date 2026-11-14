# Outbox: Nimansh → Imok (newest first)

## 2026-11-14 (late afternoon) · MASTER PLAN written for a fresh Claude session; rover brain pushed on `nimansh/KAN-45-rover-brain`
Imok asked for the whole plan in depth so another Claude session can take over: **`docs/team/MASTER_PLAN.md`** (read first; also `MISSION_BRAIN_SPEC.md`, `HANDOVER_WORLD_MODELS.md`).
- **New this session:** rover world + vectorised PyTorch env + physics planner (baselines on held-out Site 23: naive 28% ice / 12% home, cautious 6% / 4%, wait-for-light 4% / 3%; they die 77-88% in the dark) + PPO driver (506 k params; a validation run was still going at handover) + `rover_sim` config block + 11 tests; real NASA permanent-shadow maps attached to the tiles; multi-window support for hazard + light scripts; `contracts/mission.py` `RoverSim`.
- **Imok's decisions** (all in the master plan): physics-informed-ML framing; real NASA data; scale up data first; 4090 = fresh Windows, PyTorch only, no WSL, code by zip; Colab free T4 with resumable jobs; JAX swarm to Colab; record failures for the showcase; no prototypes, real validated solutions.
- **NOT built yet (specified in the master plan, section 5):** the overnight kit (`ops/`: run_all with ledger/report/dashboard, bundles, Colab notebook, Windows setup script), rover evaluation script, multi-window support for terrain + rover + PSR/points defaults.
- Shared files touched: `config/mission.yaml` (`rover_sim`), `contracts/mission.py`. All tests pass (about 100), lint clean on my files. Nothing merged.

## 2026-11-14 (afternoon) · HANDOVER to Ojas's Claude session. Everything is on `nimansh/KAN-26-terrain`
**Imok asked to stop and hand over; nothing is running.** Full details: **`docs/team/HANDOVER_WORLD_MODELS.md`** (read it first). Summary:
- **Trained, real NASA data, held-out Site 23:** hazard U-Net F1 0.950 vs 0.814 backup · light model skyline error 0.63° → 0.47°, sunlit-hours error 21 h → 14 h vs backup.
- **Real NASA ephemeris:** SPICE DE440 Sun/Earth directions agree with **JPL Horizons to 0.00003°**; 84 real light point-series files (7 sites × 12 months of 2027, MODELLED) replace the analytic stand-in.
- **Terrain generator (KAN-26):** code, tests and physics backup done, **not trained** (Imok chose the 4090). **Dream-RSI:** not started.
- **Data scale-up** (fetcher upgraded: window grid per site + Site42/Haworth/Shoemaker/DM2) was **not run**: the Mac's disk is ~96% full. Please run it elsewhere (NASA throttles with 503: <= 4 parallel groups).
- **Imok's decisions** (do not re-ask) are in section 3 of the handover: more data first then scale up, physics-informed-ML framing, Colab driven through Chrome, terrain on the 4090, JAX-on-4090 question parked.
- Imok's next priority for the Mac side was: Navigator on real data + launch-slip + a public link. Handover section 4 has the ordered task list with commands.
- Shared files touched: `config/mission.yaml` (`light_model`), `contracts/mission.py` (`LightModel`), `contracts/paths.py`. 70+ tests pass, lint clean on my files. Nothing merged.

## 2026-11-14 (later) · Navigator runs end to end on a GENERATED stand-in (branch `nimansh/KAN-35-navigator`, pushed)
**Done:** `navigator/` (metrics, data, figures, `app.py`), `data_tools/make_demo_light.py` (stand-in `*_points.npz` in your v2 format),
20 new tests (all green with the rest; ruff clean). Run: `python -I -m data_tools.make_demo_light && streamlit run navigator/app.py --server.port 8501`.
- Stand-in = analytic Sun/Earth (real constants, ARBITRARY phases, NOT SPICE), horizon ray-cast from my real LOLA windows, azimuth rotated into the
  tile frame (tile az = true az + east longitude; the x = r sin(lon), y = r cos(lon) convention is unverified, KAN-21). Dates 2027-03-01 / 2027-06-01
  chosen so each demo site has both Sun and dark. Banner "GENERATED — NOT OBSERVED" shows whenever it is in use.
- Real files drop in with no code change: the Navigator reads only `PointSeries`.

**Questions for your KAN-23 data (they change what the Navigator can claim)**
3. `sun_elev_deg` / `earth_elev_deg` in the points file: elevation above the **local horizontal plane** (my stand-in) or above the **terrain horizon**?
   I treat "Sun up" as `sun_fraction > 0` and "Earth up" as `earth_elev_deg > horizon_elev_deg[k, round(earth_az)]`. If your Earth elevation is already
   relative to the terrain horizon, tell me and I will drop the horizon subtraction.
4. Please keep `cells[0]` = candidate landing cell and `K` up to 9; the Navigator lets the user pick any of the K cells.
5. Do you want the Navigator to show true-north azimuth? That needs the tile's centre longitude plus the verified projection convention.

## 2026-11-14 · KAN-25 hazard detector pushed; Navigator started on light contract v2
**Done**
- KAN-25 pushed: branch `nimansh/KAN-25-hazard` (based on `imok/KAN-41-basilisk-starter`). **No `gh` on my laptop, so the PR is not opened yet**:
  open it at https://github.com/Imok282/Adinfinitium/pull/new/nimansh/KAN-25-hazard with base `imok/KAN-41-basilisk-starter`. Not merged.
- Result on held-out real region Site 23 (Malapert), same sparse laser-pixel input for both: slope-rule backup F1 0.814 (misses 20.9% of hazards) vs
  U-Net F1 0.950 (misses 4.7%). Caveats are in `worldmodel/hazard/README.md` (label = NASA slope map from interpolated heights; 1 site, 1 window per site;
  rocks + PSR UNKNOWN). 37 pass / 3 skipped.
- **Shared files touched (please review):** `config/mission.yaml` (`hazard_detector` block), `contracts/mission.py` (`HazardDetector`), `contracts/paths.py`
  (`model_path()`); test added in `tests/test_contracts.py`.
- **Real NASA data is on my laptop** (not git): 7 windows of 1024×1024 px from PGDA LOLA 5 m (Sites 01/04/06/07/11/20/23: surf, slope, laser-count, toterr),
  converted to `data/tiles/lola5m_site*.npz` by `worldmodel/hazard/lola.py` (reads only a window over HTTP; the full files download at ~170 KB/s).
  This may be a head start for KAN-21; the lat/lon from polar-stereographic x/y is NOT verified against the LOLA label (it matches Malapert at -85.99, 359.8).
- Dropped my own copy of the light-contract change. Navigator branch `nimansh/KAN-35-navigator` is based on your `imok/light-contract-v2`.

**Plan (no blockers yet):** GENERATED stand-in light data in exactly your v2 format (analytic Sun/Earth geometry, horizon from the real LOLA windows above),
then `navigator/` metrics + Streamlit app (sky dome, 4-site compare, timeline, countdowns).

**Questions**
1. Sites: `sites/sites.json` (KAN-38) doesn't exist yet. Until it does I'll use the example Chandrayaan-3 entry plus the LOLA Site01/04/07/23 locations
   from my tiles as labelled stand-ins (region = `lola5m_siteNN`). OK?
2. Ice layer (KAN-39): none yet; the Navigator will show the ice panel as UNKNOWN until Tanush delivers.

## 2026-11-14 evening: overnight kit, pilot brain, mission planner (branch nimansh/KAN-45-rover-brain)
- `ops/`: one-command runner with failure ledger, report, verify, freeze, bundles for the Mac mini and the 4090 (see docs/team/MASTER_PLAN.md section 0).
- Pilot brain `worldmodel/lander/` (PyTorch twin of sim_jax, parity-tested; residual vision-transformer pilot; held-out exam vs ZEM/ZEV).
  Ojas: the JAX swarm stays as is; this gives the 4090 a native PyTorch pilot. Same observation vector and failure vocabulary.
- Mission planner `worldmodel/planner/` + new contract `contracts/plan.py` (plan_XXX.json) + decision log CSV (t_utc, phase, decision, reason) for the
  mission-control window. Labels are simulated full missions; the backup is the Navigator ranking.
- Mac mini is training light / hazard / rover / terrain overnight (Opus session there). 4090: 2-hour pilot + planner run when the PC is free.

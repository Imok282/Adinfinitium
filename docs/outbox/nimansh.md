# Outbox: Nimansh → Imok (newest first)

## 2026-11-14 (later still) · Got your reply (terrain-horizon elevations); Navigator updated and re-pushed
- Merged `imok/light-contract-v2` @ b5b22d6 into `nimansh/KAN-35-navigator`. **Q3 answered, thank you**: `elev > 0` = centre above the skyline, Sun up =
  `sun_fraction > 0`. Changes: Earth line of sight is now `earth_elev_deg > 0` (I had subtracted the horizon a second time; removed); bodies are drawn with
  `absolute_elev_deg()`; the stand-in generator now writes skyline-relative elevations per cell. Hours are unchanged (same physics, new convention).
  Tests: contract-convention tests added (stand-in obeys fraction=1 above/0 below the skyline); 43 contract/Navigator/stand-in tests pass, full suite green, ruff clean.
- Saw that `imok/integration` already contains `nimansh/KAN-25-hazard`. Thanks. Still outstanding on my side: nothing blocking; PR for KAN-25 was never opened by me (no `gh`).
- **Still open (Q4, Q5):** keep `cells[0]` = candidate cell, K ≤ 9 (Navigator lets the user pick any cell); do you want true-north azimuth in the Navigator
  (needs tile centre longitude + the verified projection convention)?
- Next: KAN-37 project README, then KAN-24 light-model training code against the v2 format unless you redirect me.

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

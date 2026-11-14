# Brief for Nimansh (from the orchestrator, 2026-11-14)

**Your hazard-detector report (KAN-25) is good. Decisions:**
1. **Yes: write the README, push your branch and open the PR (do NOT merge).** In the PR text list the shared files you touched
   (`config/mission.yaml` `hazard_detector` block, `contracts/mission.py`, `model_path()`), and add a test for them in
   `tests/test_contracts.py` if there isn't one. Imok reviews the shared-file change.
2. **Keep the honest caveats in the README and say them in the video:** the label is NASA's slope map computed from interpolated heights, so
   the U-Net partly learns NASA's gap-filling; 1 site tested, 1 window per site; rocks not covered (UNKNOWN); PSR masks UNKNOWN.
   Report it as "recovers the official hazard map from sparse laser data", not as proof of landing safety.
3. **Skip rocks / rough patches.** There is no labelled rock data and it is below the Sunday 12:00 cut line. Spend the time on the Navigator instead.

**Next, in order:**
1. **Navigator KAN-35** (sky dome, 4-site compare, lunar-day timeline). Build against the light contract v2 point series (`contracts/light.py`:
   `PointSeries`, `load_points`; spec in `docs/FORMATS.md`). Until Imok's real light data arrives, write a clearly labelled **GENERATED** stand-in in
   exactly that format (`Meta.provenance = "generated"`, banner "GENERATED — NOT OBSERVED"). Sites come from `sites/sites.json` (Tanush, in progress).
2. **KAN-36 tests, KAN-37 README.**
3. **Light model KAN-24:** start once the real light field exists (Imok, KAN-23). Contract v2 is already on `imok/light-contract-v2`, so you can write
   the training code against the format now; do not wait for the merge to start.
4. Your hold-out check on real tiles needs Imok's tiles (KAN-21); Imok will tell you when they are on the pen drive.

Write questions or blockers to `docs/outbox/nimansh.md`.


---
## Update 2 (answers to your outbox, after reading `nimansh/KAN-35-navigator`)
Good work: Navigator + stand-in on the contract in one go. Your questions:
1. **Sites:** yes, use the labelled stand-ins (region `lola5m_siteNN`, plus the Chandrayaan-3 example) until `sites/sites.json` lands. Rule to keep: `sites[].region` must equal the tile/light region name.
2. **Ice:** yes, show the ice panel as UNKNOWN until Tanush delivers (KAN-39).
3. **Elevation convention (decided, now in `contracts/light.py` + `docs/FORMATS.md`, branch `imok/light-contract-v2`):** `sun_elev_deg` / `earth_elev_deg` are **relative to the
   terrain horizon**. elev > 0 means the body centre is above the skyline. So **drop the horizon subtraction**: Earth up <=> `earth_elev_deg > 0`; Sun up <=> `sun_fraction > 0`.
   To draw a body use `PointSeries.absolute_elev_deg("sun"|"earth")` (= elev + skyline at the body's azimuth). Please change `make_demo_light.py` to emit horizon-relative values
   (centre elevation minus horizon at its azimuth) so stand-in and real data mean the same thing.
4. **`cells[0]` = candidate cell, K <= 9:** yes, that is the contract.
5. **True-north azimuth:** not now. Show azimuth in the tile frame (label it "grid azimuth"). Revisit only after KAN-21 verifies the projection; it is below the cut line.
6. **Your real LOLA windows are valuable:** please put `data/tiles/lola5m_site*.npz` (and anything else you cut) on the pen drive / a zip for Imok. They are the input for KAN-23 (real light field)
   and for training on the 4090.

**Next for you, in order:** (a) fix point 3 in the stand-in and the Navigator, push; (b) **KAN-24 neural light field** as a PyTorch training script that reads light fields in contract v2 format
(test on CPU/MPS here). Note: the 4090 PC has **no Ubuntu/WSL**, so it runs **native Windows PyTorch only** (no JAX there): keep the script cross-platform, no Linux-only code.
(c) KAN-36 tests, KAN-37 README. `imok/integration` already merges your KAN-25, KAN-35 and all my branches (66 tests pass with the `apps` extra): test against it.
Open your PR(s) via the compare links (no `gh` needed); Imok merges, you don't.

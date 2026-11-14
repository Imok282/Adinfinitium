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

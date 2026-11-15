# Brief for Imok (orchestrator's own checklist, 2026-11-14)

Done: swarm (--resume, --exams), exams, look-ahead branch pushed, light contract v2 implemented. Open, in order:
1. Merge PR #1 then PR #2 on GitHub (main is empty until then). Then review/merge `imok/KAN-105-tensorneat`, `imok/light-contract-v2`, `imok/KAN-100-lookahead`.
2. On the 4090: `make bench`, then `make swarm` (tight_fuel) before you sleep; watch the first generation's time and lower `--generations` if needed.
   `overnight2` (hazard exams, `--lookahead`) only after the look-ahead branch is merged. See `swarm/README.md`.
3. Real data: LOLA tiles + SPICE kernels (KAN-21/22), then the light field (KAN-23) in contract v2 format.
4. Send Arnav 3 projection test points (lat/lon -> x,y).
5. Rely on the repo, not Jira: the KAN board is stale (almost everything still shows To Do).

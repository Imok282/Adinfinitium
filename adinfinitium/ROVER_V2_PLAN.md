# Rover v2: pre-registration (written and committed before any v2 number exists)

Written Sun 15 Nov 2026, about 00:20 IST, by the rover v2 lead. Nothing below may change after a held-out number for v2 has been seen.

## Why v2 exists, and what we had already seen

Rover v1 (`rover_driver_mini`, PPO from scratch, 144 updates) failed. Its held-out result (Site23 + Site42, 8 worlds x 512 rovers) is in the main run's
`runs/rover/eval.json` and stays as it is: network reached ice 0.017 / returned 0.000 / died 0.955; naive planner 0.196 / 0.093 / 0.868; cautious planner
0.105 / 0.088 / 0.543; cautious + wait-for-light 0.069 / 0.060 / 0.739. **We have seen these numbers.** They are the reason for v2. They are not used to
choose the teacher, a threshold, a checkpoint or anything else below: every choice v2 makes from data is made by code on train sites or on the validation
site Site11. The planner numbers on the held-out worlds are deterministic and will come out the same in the v2 evaluation.

Diagnosis (records/agents/rover-diagnosis.md): exploration failure. Random driving never reaches a 20 m target 330 to 1600 m away, so PPO never sees the
+50 and collapses to "stop".

## What stays the same

Simulator physics, reward, planners, evaluator (`worldmodel/rover/*`, `config/mission.yaml`): not one byte changed (the launcher checks this against the
main run folder). Same network as v1: vision transformer `--arch former --d 384 --layers 6 --heads 8` (11,377,354 parameters). Same hazard map shown to
the rover (`hazard_unet_big`). Same train worlds as v1: centre window of Site01, 04, 06, 07, 20, Haworth, Shoemaker, DM2 x 4 dates of 2027 (32 worlds).
Same held-out evaluation command: `worldmodel.rover.evaluate --n 512 --hazard-name hazard_unet_big` on Site23 + Site42 x 4 dates.

## Method (the only thing that changes): learn from the physics planners instead of from random exploration

1. **Teacher.** Six fixed variants of the unchanged physics planner `RoverEnv.planner_action`:
   naive; cautious (reserve 0.1); cautious + wait-for-light (need 1.5); naive + wait-for-light (need 1.5); cautious (reserve 0.3);
   cautious + wait-for-light (need 2.5). For every training rover (start position, start hour, target) all six are simulated for the full episode in the
   train-site world; that rover's teacher is the variant with the highest episode return (the task's own reward; ties go to the first in the list above).
   This lookahead is used only to make training labels on train sites. The network gets no such information: it sees only the normal observation.
2. **Imitation with DAgger.** The network is trained with cross-entropy to copy the teacher's action. After the first rounds, half of the rovers in each
   data-collection episode are driven by the network itself (greedy, 5% random actions) and the states they reach are labelled with what their teacher
   would do there, so the network learns to recover from its own mistakes. Full 300-step episodes, so episodes finish.
3. **No PPO fine-tune in v2.** Reduced scope, decided now: there was no time to test a fine-tune step, and it is the step that collapsed in v1.
   Consequence, stated now: the honest ceiling of v2 is "as good as picking the best planner variant per rover", it cannot discover behaviour no planner shows.
4. **No hybrid.** The number reported as "model" is the network alone, greedy, with no planner fallback at deployment.

## Checkpoint selection (validation site Site11 only)

- Candidates: a snapshot of the network about every 12 minutes of training plus the final weights.
- Each candidate drives full episodes, greedy, in the 4 Site11 worlds (centre window x 2027-03-01, 06-01, 09-01, 12-01), 256 rovers per world, seed 4321.
- Selection score = mean over the 4 worlds of `returned` (fraction of rovers that reached ice AND came back to the lander). Ties: higher `reached_ice`,
  then lower `died`, then the earlier candidate.
- The winner is saved as `rover_driver_v2.pt`. Every candidate's Site11 numbers are kept in `runs/_rover2/selection.json` (losers included).
- Training-rollout statistics are never used for selection (that was the v1 rule that picked the update-1 network).
- Site23 and Site42 are never loaded by the training / selection program (asserted in code and tested).

## Metrics

- Primary: `returned` = fraction of rovers that reach ice and return to the lander (summary mean over the 8 held-out worlds, 512 rovers each).
- Secondary: `reached_ice` (higher is better), `died` (lower is better). `mean_return` is reported, not judged.

## The one held-out evaluation and the verdict

One run of the unchanged evaluator on Site23 + Site42 after selection is final. It is run once (the launcher refuses to repeat a completed one) and is
reported whatever it shows. "Best planner" = the planner with the highest `returned` in that same eval.json (three planners evaluated by the evaluator).

- **SUCCESS (beats the best planner):** model `returned` > best planner `returned` + 0.01, and model `died` <= that planner's `died`.
- **MATCH (as good as the best planner):** model `returned` >= best planner `returned` - 0.01, and model `died` <= that planner's `died` + 0.02.
- **PARTIAL:** not a match, but model `returned` >= half of the best planner's `returned` (the network does the mission, worse than the planner).
- **FAILURE:** anything else.

(0.01 is about two standard errors of a 9% rate over 4096 rovers.) The report shows v1, v2 and the three planners side by side and prints this verdict.
If the run is cut short, the report says so and judges whatever checkpoint Site11 selected.

## Time on the Mac mini

Training including the Site11 selection evaluations: at most 110 minutes (default), hard-stopped. Held-out evaluation: about 8 to 10 minutes.

## Known weaknesses, stated in advance

- The teacher's per-rover choice uses the future (simulation); the network may not be able to infer it from its observation, so it can end below the
  single best planner. Even the planners lose most rovers on this task.
- Site11 is one site (4 worlds); it has no rover caches yet, they are built at the start of the run.
- Throughput on the mini is an estimate (about 1000 training samples per second); nothing was measured there before the run.

## Clarifications (added 00:36 IST on 15 Nov, after an independent review of the code, before any v2 held-out number exists; nothing above is changed)

- The plan above was committed at 00:11 IST (commit d02c090), not "about 00:20".
- Time: 110 minutes is the trainer's own budget (world building, training, Site11 selection). The launcher's hard kill is 8 minutes later (118), the
  held-out evaluation has its own 25-minute hard limit. Worst case wall time is about 145 minutes; expected about 120.
- "Centre window" means the first window `regions_for_site` returns that the spatial split has not excluded (the same rule v1 and the evaluator use).
- "Best planner" on a tie in `returned`: the first of naive, cautious, cautious + wait-for-light.
- A held-out evaluation that did not finish all 8 worlds gets the verdict NO RESULT; its partial numbers are shown as provisional.
- A training crash (not the time limit) is resumed up to 3 times inside the same budget, before the held-out evaluation; once that evaluation has
  started, training is closed for good and the evaluated file must have the sha256 recorded at closing.
- Training details fixed now: 512 rovers per data-collection episode, every train world once per pass in random order, 25% of visited states stored in a
  100,000-state buffer, batch 256, Adam 2e-4, gradient steps per episode = 4 passes over the states it added (16 to 400), first 8 episodes teacher-driven,
  then half of the rovers network-driven. Seeds: training 0, Site11 selection 4321.
- Seen before the run, on the validation site and train sites only (no test site was loaded by any v2 code): a CPU-only probe on the mini printed the
  three planners on Site11 (4 worlds x 32 rovers: returned 0.336 / 0.352 / 0.250), and small MacBook smoke runs with tiny networks printed Site11
  candidate rows. They were used to check that the pipeline runs, not to change the method, metric, selection rule or verdict rule.
- Smoke runs use the launcher's test hooks; such a run writes `rover_driver_v2_smoke` and `runs/_rover2_smoke/`, never the real names.

## Amendment 1 (00:53 IST on 15 Nov, before any v2 held-out number exists; decided from train-site and Site11 evidence only)

What changed: which training states are stored. Nothing else (teacher, metric, selection rule, verdict rule are as above).

- States where the rover's planner has no route (it answers "stop" even without the wait-for-light rule) are no longer stored as training examples.
- A rover that did not move on the previous step is stored 1/8 as often as a rover that moved.
- Gradient steps per episode: 4 passes over the states it added, 64 to 400 (was 16 to 400).

Why: a small MacBook check (1 M-parameter network, 5 train sites, seed hazard map; logged in records/agents/rover-v2-lead.md at 00:41 and 00:47) showed
the copied network stalling: where its teacher said "move" it said "stop" 37% of the time on train world site01 and 84% on validation world site11,
and on Site11 it returned 0.13 against the best planner's 0.36. Cause: a planner whose target is unreachable repeats "stop" until the battery is empty,
about 50 near-identical examples per such rover, and the network cannot see reachability in its 320 m view, so "stop" swamped the examples near the
start. The planner's "stop" there is a fallback, not a driving decision. Consequence, stated now: for a target no planner can reach, the v2 network will
probably drive and die where the planner would stand still (and also die when its battery runs out, unless it started within 20 m of the lander); this
can raise v2's `died` a little, and the verdict rule above still applies unchanged.

No test site (Site23, Site42) was loaded by any v2 code before or for this change.

## Amendment 2 (01:02 IST on 15 Nov, found by the independent re-review, before any v2 held-out number exists)

Amendment 1's "planner has no route" was coded too widely: it also dropped the stops of a cautious rover that had turned back and was standing at the
lander (the re-review measured 13,997 of 14,542 states dropped in a dark flat test world). Those stops are a real decision ("stay at the lander") and
are training examples again. "No route" now means exactly: the planner answers "stop" even without the wait-for-light rule AND the rover's current goal
is unreachable from where it stands (cost-to-go at the simulator's cap). Standing rovers are still stored 1/8 as often. Also corrected: "resumed up to
3 times" in the clarifications means at most 3 attempts (2 resumes). Not listed before, unchanged since the first code: Adam eps 1e-5, gradient clip 1.0.

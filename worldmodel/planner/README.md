# worldmodel/planner: the meta layer (mission planner + orchestrator)

Chooses WHERE (landing cell) and WHEN (landing hour) a mission most likely succeeds, says go / wait, and writes the plan + decision log.

| File | What |
|---|---|
| `features.py` | per candidate (cell, hour): sunlight and darkness ahead (NASA SPICE + exact skyline), Earth link and blackouts, hazard and clear radius, slope, distance to NASA PSR ice-edge targets; `navigator_score` = the exact non-learned ranking (**backup**) |
| `labels.py`, `labels_run.py` | **simulated outcomes as labels**: the lander twin flies to the exact cell (p_land, touchdown scatter) and the rover environment drives from it at that hour with the physics planner (p_ice, p_return). p_mission = p_land x p_return-with-ice |
| `model.py` | `PlannerSetNet` (candidates of one world attend to each other: a listwise ranker, ~0.4 M) and `LogisticPlanner` (honest small baseline). Small on purpose: the data are hundreds to thousands of simulated candidates |
| `train.py`, `evaluate.py` | site-split training; held-out exam on Site23 + Site42: Brier, calibration error, rank correlation, chosen-candidate success and regret vs **Navigator backup** and random |
| `plan.py` | scores candidates, flies the lander to the chosen cell for its scatter radius, writes `runs/<run>/plan_XXX.json` (`contracts/plan.py`) and `decisions_XXX.csv` |
| `orchestrator.py` | state machine orbit -> descent -> touchdown -> surface -> return deadline -> night -> sunrise; every row has a reason |

Limits: labels come from simulation (rover kinematic, hourly), the physics rover rarely completes ice-and-return in many worlds so p_mission is often small;
PSR = dark and cold, not confirmed ice; landing hours are a few slots per month.
```bash
python -m worldmodel.planner.labels_run --max-minutes 120
python -m worldmodel.planner.train && python -m worldmodel.planner.evaluate
python -m worldmodel.planner.plan --region lola5m_site23 --date 2027-03-01 --run demo
```

## Dream-RSI: the planner keeps improving itself offline (`dream.py`)
After training stops (e.g. the planner is installed on a laptop), it can keep learning on its own, safely:
imagine new missions on training terrain -> pick the ones it is least sure about (its two predictors disagree) -> play them out in the
offline physics simulators (lander twin + rover environment, NASA LOLA terrain, SPICE light) -> warm-start retrain -> **keep the new
version only if it is better on a frozen validation set**, else roll back. Each accepted version chooses the next dreams (recursive).
Test sites are never dreamt about. State and history: `<ADI_DATA>/planner_dream/` (resumable, Ctrl-C safe).
```bash
python -m worldmodel.planner.dream --generations 3
python -m worldmodel.planner.dream --forever          # keep improving whenever the laptop is on
```
First run (Mac, 14 Nov): 120 missions imagined, 2 most-uncertain cells simulated (112 missions), validation Brier 0.0089 -> 0.0091, rejected (kept the previous version): the gate working.

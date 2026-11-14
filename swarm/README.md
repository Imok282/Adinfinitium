# swarm/

**Owner:** Imok · **Jira:** KAN-30 to KAN-32, KAN-34, KAN-105

TensorNEAT evolution of landing pilots on the JAX sim + champion export. (Dream-RSI meta layer: KAN-106, not yet.)

| | |
|---|---|
| Reads | `sim_jax` env, `config/mission.yaml`, a tile (`data/tiles/<region>.npz`) |
| Writes | `runs/<run>/generation_XXX.json`, `champion_XXX.json` (contracts/runs.py), plus `genome_XXX.npz` |

```bash
# separate venv: tensorneat is GitHub-only and pulls brax/mujoco
uv venv --python 3.12 ~/venvs/adi-swarm && source ~/venvs/adi-swarm/bin/activate
uv pip install -e ".[dev]" "git+https://github.com/EMI-Group/tensorneat.git"   # or: pip install -e ".[swarm,dev]"
make swarm-smoke                                  # laptop, ~1 min
python -m swarm.run_swarm --pop 10000 --generations 300 --run overnight1   # 4090 (make swarm)
```

## The 4090 overnight run (tight_fuel + hazard exams)

`--exams a,b,c` makes selection ALSO fly those `worldmodel.exams` scenarios (each exam env gets equal weight with
the normal env; `--exam-targets` targets per tile, `--exam-missions` missions per env per individual). The same
stress is used for training and reporting, but on **different generated terrain**: training tiles are seeded
+1000, the held-out exam tiles are the exam's own. Each `generation_XXX.json` then also has `exam_success_rate`,
`classic_exam_success_rate` and `exam_breakdown` (per exam, champion vs classic, fresh held-out missions).
Caveat: the `tight_fuel`/hazard exam tiles share seed 0 with the standard demo tile, so those are *scenario*
generalisation (new targets, new stress draws), not new terrain; only `unseen_craters` is new terrain, and it is
never trained on.

**Tonight, works from the pushed `imok/KAN-105-tensorneat` (tight_fuel only; classic = 25.7% there):**
```bash
cd ~/adinfinitium && git fetch && git checkout imok/KAN-105-tensorneat && git pull
source ~/venvs/adi-jax/bin/activate && pip install -e .
export ADI_RUNS=~/adi-runs
tmux new -s swarm          # then, inside tmux:
make swarm                 # = python -m swarm.run_swarm --pop 10000 --generations 300 --run overnight1 \
                           #     --exams tight_fuel --exam-targets 3 --exam-missions 2 --resume
```
Crashed or rebooted? Re-run the identical `make swarm` (checkpoint -> resumes). Watch the first generation's
time before sleeping: 300 generations must fit the night. If it is too slow, lower `--generations` (the resume
total) or `--exam-targets`; if the GPU runs out of memory add `SWARM_FLAGS="--pop-batch 2500"`.

**Hazard exams (needs the look-ahead; NOT usable until `imok/KAN-100-lookahead` is reviewed and merged):**
```bash
make swarm RUN=overnight2 EXAMS=tight_fuel,nav_drift_near_hazard,combined_stress SWARM_FLAGS=--lookahead
```
Without `--lookahead` the pilot only senses the cell below it and cannot divert, so the hazard exams would be
trained blind. (`--lookahead` is added to `run_swarm` on the KAN-100 branch.)

**Crash recovery:** every generation ends by writing `runs/<run>/checkpoint.pkl` (full population, species, RNG;
atomic, written after that generation's json/npz). Re-run the SAME command with `--resume` to continue from the
last finished generation; `--generations` is the total to reach. With no checkpoint, `--resume` just starts fresh,
so one command works for the first launch and every restart. Without `--resume`, an existing checkpoint is never
overwritten, and a resume with different pop/mode/seed/etc. is refused. Tested: a resumed run reproduces the
uninterrupted run exactly (`pytest -m slow`, ~90 s).

Files: `pilot.py` (net output -> thrust; **residual on classic guidance** or from scratch), `fitness.py`
(shaped selection fitness vs the gated mission score), `problem.py` (TensorNEAT `BaseProblem`; every
individual flies the same missions in a generation), `run_swarm.py` (loop, logs, champion export).

**DISCLOSURE (required in the video and on the submission page):** the default `residual` pilot is classic ZEM/ZEV
guidance plus a learned correction (<= `--residual-scale` of max thrust). It is NOT a from-scratch AI pilot; say
"evolved residual on classic guidance". `--mode scratch` has no physics prior (and does not land reliably yet).

**Reading `generation_XXX.json`:** `success_rate` is the champion on held-out missions, `classic_success_rate`
is classic guidance on the *same* missions. `mean_score` is the population's mean of the shaped selection
fitness (not the gated mission score). Extras: `population_success_rate`, `classic_best_score`, `score_basis`.

Fallback: fixed-topology network with evolved weights (KAN-106). Physics backup: `sim_jax.guidance.classic_policy`.

Rules: talk to other folders only through the files in `contracts/`; label outputs observed/modelled/generated/unknown; physics backup before AI. See `docs/SYSTEM_DESIGN.md`.

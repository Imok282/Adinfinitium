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

Files: `pilot.py` (net output -> thrust; **residual on classic guidance** or from scratch), `fitness.py`
(shaped selection fitness vs the gated mission score), `problem.py` (TensorNEAT `BaseProblem`; every
individual flies the same missions in a generation), `run_swarm.py` (loop, logs, champion export).

**What the evolved pilot is — say this in the video:** the default `residual` pilot adds a learned correction
(<= `--residual-scale` of max thrust) to classic ZEM/ZEV guidance. `--mode scratch` has no physics prior.

**Reading `generation_XXX.json`:** `success_rate` is the champion on held-out missions, `classic_success_rate`
is classic guidance on the *same* missions. `mean_score` is the population's mean of the shaped selection
fitness (not the gated mission score). Extras: `population_success_rate`, `classic_best_score`, `score_basis`.

Fallback: fixed-topology network with evolved weights (KAN-106). Physics backup: `sim_jax.guidance.classic_policy`.

Rules: talk to other folders only through the files in `contracts/`; label outputs observed/modelled/generated/unknown; physics backup before AI. See `docs/SYSTEM_DESIGN.md`.

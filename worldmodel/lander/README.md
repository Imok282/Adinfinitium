# worldmodel/lander: the pilot brain (PyTorch)

A learned lander pilot that trains natively on a CUDA / MPS GPU (no JAX needed), next to its physics backup.

| File | What |
|---|---|
| `env.py` | vectorised 3-DOF powered-descent lander, a **faithful port of `sim_jax/lander_env.py`** (same physics, observation, touchdown rules, failure injection, noise); parity tested against JAX |
| `guidance.py` | classic ZEM/ZEV guidance (the backup and the baseline) |
| `terrain.py`, `scenes.py` | real LOLA tiles; the touchdown truth is the official slope rule, the camera sees the hazard **belief** (backup map or hazard U-Net) |
| `pilot.py` | `PilotFormer` (vision transformer: camera image patches + state token, ~15 M at d 384 / 8 layers) and `PilotMLP` (small); a **residual** on classic guidance, zero-initialised, so the untrained pilot == the backup (tested) |
| `train.py` | PPO on many landing cells; best checkpoint chosen on **validation Site11** vs classic, never on training reward |
| `evaluate.py` | held-out exam on Site23 + Site42: same missions flown by classic and pilot, config failure injection, per-failure-type success, several seeds |

Honest baseline: classic guidance already lands ~95-98 % on real terrain with failures injected (Site23 97.7 %, Site11 94.9 % in a 2 x 128 test). The pilot can
only add value in the remaining cases (engine-out, navigation drift, hazard under the target). If it does not beat classic on the test sites, that is reported
and classic guidance ships.

```bash
python -m worldmodel.lander.train --arch former --d 384 --layers 8 --envs 512 --updates 1000 --max-minutes 150
python -m worldmodel.lander.evaluate --name lander_pilot --n 1024 --seeds 3
```

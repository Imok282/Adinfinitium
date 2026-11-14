# Ghost trajectories (real simulated runs, successes AND failures)

Made by `python -m worldmodel.ghosts` (rover environment + PyTorch lander twin on NASA LOLA terrain). Nothing invented or tuned; MODELLED.
Frame: tile ENU metres, origin = tile centre, x east, y north, z = LOLA height. Colour legend for every viewer: best = green, other successes =
light cyan, crash / death = red, timeout / out of fuel / out of bounds = orange; non-best runs drawn as semi-transparent ghosts that stop at `fail_t_s`.

| file | runs | outcomes |
|---|---|---|
| rover_lola5m_site06_2027-03-01.json | 48 (naive, cautious, wait_light; 16 each, same starts) | 21 returned, 18 died_battery, 9 timeout (training site) |
| rover_lola5m_site23_2027-09-01.json | 48 | 2 returned, 45 died_battery, 1 timeout (held-out test site: the hard case) |
| lander_lola5m_site06.json | 64 (32 classic ZEM/ZEV, 32 open-loop = fixed thrust, no sensors) | 31 ok, 10 hard_vertical, 17 out_of_bounds, 6 timeout |

Lander failure rates are raised (engine-out 30 %, nav drift 30 %, blind camera 15 %) so failure modes show; the config rates are lower.
Learned rover driver / lander pilot runs are added with `--learned <ckpt>` / `--pilot <ckpt>` once trained.

# Shared file formats

The **code** in `contracts/` is the authoritative definition. This page is the quick reference.
Validate any file with:

```bash
python -m data_tools.validate_formats <file> [<file> ...]
python -m data_tools.validate_formats --selftest
```

Every file carries a `meta` block:

| Field | Meaning |
|---|---|
| `provenance` | `observed` · `modelled` · `generated` · `unknown` (**required**; generated → UI shows "GENERATED — NOT OBSERVED") |
| `source` | dataset / script it came from, e.g. `PGDA 90 LDEM_80S_20MPP_ADJ` |
| `frame` | default `MOON_ME_DE440_ME421`; tile grids are local ENU metres centred on the tile |
| `units` | array/field → unit |
| `created_by`, `created_at`, `schema_version` | bookkeeping |

## `config/mission.yaml` — `contracts/mission.py`
All lander, rover, power, scenario, failure, noise, scoring and swarm numbers. Unknown keys are
rejected. Cross-checks: terminal start mass − propellant = dry mass, and the lander can hover
with one engine out.

## `data/tiles/<region>.npz` — `contracts/tiles.py`
| Array | Shape | dtype | Unit |
|---|---|---|---|
| `height` | [H,W] | float32 | m (relative to 1737.4 km sphere) |
| `slope` | [H,W] | float32 | deg, 0..90 |
| `hazard` | [H,W] | float32 | probability 0..1 |
| `uncertainty` | [H,W] | float32 | m, 1-sigma |
| `psr_mask` | [H,W] | bool | permanently shadowed |
| `real_mask` (optional) | [H,W] | bool | laser spot present |
| `header` | scalar str | JSON | region, meters_per_pixel, center_lat/lon_deg, meta |

Grid: row 0 = north (+y), col 0 = west (−x); `x = (col − (W−1)/2)·mpp`, `y = ((H−1)/2 − row)·mpp`.

## `data/light/<region>_<date>.npz` — `contracts/light.py`
Same grid as the tile. `times_utc` [T] (in header), `et_s` [T] float64 (strictly increasing),
`sun_fraction` [T,H,W] 0..1 (finite solar disk), `sun_elev_deg` and `earth_elev_deg` [T,H,W]
above the **local** horizon (terrain + curvature, no refraction).

## `sites/sites.json` — `contracts/sites.py`
`{"schema_version": 1, "sites": [{id, name, lat_deg, lon_deg, source, provenance, notes, region}]}`.
Ids are unique snake_case. See `sites/sites.example.json`.

## `runs/<run>/generation_XXX.json` — `contracts/runs.py: GenerationLog`
run, generation, population, best/mean score, success_rate, classic_success_rate,
exam_success_rate, failures [{reason, count}], champion_id, champion_genome, strategy,
wall_time_s, meta. Failure reasons come from `FAILURE_REASONS`.

## `runs/<run>/champion_XXX.json` — `Champion`
run, generation, champion_id, score, genes {site, target_x_m, target_y_m, landing_time_utc,
battery_wh, heater_w}, initial_state {pos_m[3], vel_m_s[3], mass_kg}, controls [{t_s,
thrust_n[3]}] in tile ENU newtons, outcome, meta.

## `runs/<run>/telemetry_XXX.csv` — `TELEMETRY_COLUMNS`
`t_s, phase, x_m, y_m, alt_m, vx_m_s, vy_m_s, vz_m_s, mass_kg, fuel_kg, thrust_n, engines_ok,
battery_wh, solar_w, earth_visible, decision` — exactly this order.

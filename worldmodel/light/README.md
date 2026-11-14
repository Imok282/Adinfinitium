# worldmodel/light — neural light field (KAN-24) + real Sun/Earth ephemeris

**Owner:** Nimansh · answers "how much Sun, and is Earth visible, at this point and time?" — fast, with an exact physics backup.

**Design.** Time never enters the network. The model predicts the terrain **skyline** (horizon elevation for every compass degree) at a cell. Sun and Earth
directions come from **NASA SPICE** (below), and plain geometry (`physics.py`) turns skyline + direction into Sun fraction (finite solar disk) and Earth visibility.
The model is a learned **correction on top of the exact ray-cast of coarse terrain** (the physics backup); an untrained model equals the backup exactly (tested).

## Results (real NASA data; held-out test = Site 23, never trained or tuned on)
Input is the 5 m LOLA height block-averaged 4x (a stand-in for LOLA's coarser product); truth is the exact ray-cast of the 5 m grid. 1,000 random cells.

| Site 23 (TEST) | Physics backup | Model |
|---|---|---|
| Skyline error, mean / p95 | 0.63° / 2.65° | **0.47° / 1.99°** |
| Sun up/down wrong (709 h window) | 3.0% | **2.2%** |
| Earth visible wrong | 2.3% | **1.6%** |
| Sunlit hours error, mean / p95 | 21.2 h / 94 h | **14.1 h / 62 h** |

Reproduce: `python -I -m worldmodel.light.evaluate` (checkpoint `b1fac75ad7`, 32,737 parameters, trained on Sites 01/04/06/07/20, validated on Site 11).

## Real ephemeris (`ephemeris.py`, `make_points.py`)
Sun and Earth positions are **NASA/JPL SPICE (DE440)**, topocentric, apparent, in the MOON_ME frame. Kernels (public, NAIF) go in `<ADI_DATA>/kernels/`:
`naif0012.tls`, `de440s.bsp`, `pck00011.tpc`, `moon_pa_de440_200625.bpc`, `moon_de440_250416.tf`.
**Checked against JPL Horizons** (NASA's own service) at Site 23 over two 30-day windows: elevation and azimuth agree to **0.00003°** for both Sun and Earth;
pinned in `tests/test_ephemeris.py`. `python -I -m worldmodel.light.make_points` writes real light point series (provenance MODELLED, 12 monthly windows of 2027 per site).

## Honest limits
- The model sees only a coarse 20 m terrain: detail averaged away cannot be recovered, so the gain is modest (about 25–33%). More data and a bigger model help; they cannot beat missing information.
- The skyline is ray-cast only out to 1,280 m inside the ~5 km LOLA window: far mountains are missing and the skyline is biased low. LOLA heights are mostly interpolated.
- Grid azimuth = true azimuth + east longitude assumes the LOLA x = r·sin(lon), y = r·cos(lon) convention, still unverified (KAN-21).
- 32.7 thousand parameters: a small, narrow model trained from scratch on 5 tiles. One held-out site.

## Run
```bash
pip install -e ".[hazard,dev]" spiceypy
python -I -m worldmodel.light.train                          # laptop or 4090 (native Windows PyTorch is enough); --resume to continue
python -I -m worldmodel.light.evaluate
```
Splits (whole tiles): train Site01/04/06/07/20 · val Site11 · **test Site23** (never loaded by `train.py`). Checkpoints: `<ADI_DATA>/models/light_skyline*.pt` (not in git).

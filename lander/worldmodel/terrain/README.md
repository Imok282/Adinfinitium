# worldmodel/terrain — terrain generator (KAN-26)

**Owner:** Nimansh · imagines plausible 5 m terrain where LOLA has no laser measurement (about 90% of pixels). **Every output is GENERATED — NOT OBSERVED.**

**Design.** Conditional diffusion (small U-Net, 2.04 M parameters) trained **only on real laser-measured pixels** by hide-and-check: part of the measurements are
hidden, the model must predict them back; interpolated pixels are never supervised. Physics backup (no AI): gap-fill the measurements and add noise from LOLA's own error map.

**Status: code and tests done, model NOT trained yet** (planned on the 4090 with more data). `python -I -m worldmodel.terrain.evaluate` runs the hide-and-check test:
hide half the real measurements of held-out patches, generate versions, compare with the hidden REAL heights, against the physics backup (RMSE, 90% band coverage, band width).
If the model cannot beat the backup, the backup ships (cut rule Sunday 12:00).

```bash
python -I -m worldmodel.terrain.train [--base 64 --steps 12000 --resume]
python -I -m worldmodel.terrain.evaluate
python -I -m worldmodel.terrain.generate --region lola5m_site23 --versions 4 [--backup-only] [--crop 512]
```
Splits (whole tiles): train Site01/04/06/07/20 · val Site11 · **test Site23**. Writes `data/tiles/<region>_gen<k>.npz` (provenance GENERATED, uncertainty = spread across versions).

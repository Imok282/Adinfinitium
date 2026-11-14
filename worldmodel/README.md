# worldmodel/

**Owner:** Imok · **Jira:** KAN-23 to KAN-27

Light-field teacher + neural light field, hazard U-Net, diffusion terrain versions, exam sets.

Exam sets live in `exams.py` (`python -m worldmodel.exams` prints the classic-guidance baseline). They use only the public `sim_jax` API; hard cases are overrides + target placement, never edits to the shared sim.

| | |
|---|---|
| Reads | Tiles + SPICE kernels |
| Writes | `data/light/<region>_<date>.npz` (contracts/light.py), hazard maps written back into tiles, exam sets |

Physics backups: exact ray-cast teacher; slope ≥ 12° hazard (`contracts.tiles.hazard_from_slope`); LDEM_ERR noise instead of diffusion.

Rules: talk to other folders only through the files in `contracts/`; label outputs observed/modelled/generated/unknown; physics backup before AI. See `docs/SYSTEM_DESIGN.md`.

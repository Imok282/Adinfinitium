# data_tools/

**Owner:** Imok · **Jira:** KAN-21, KAN-22

Download LOLA/ice/SPICE data and cut it into tiles.

| | |
|---|---|
| Reads | Raw PGDA/PDS files on the pen drive |
| Writes | `data/tiles/<region>.npz` (contracts/tiles.py) |

`make_demo_tile.py` writes a GENERATED crater tile so everyone can work before real tiles exist.

Rules: talk to other folders only through the files in `contracts/`; label outputs observed/modelled/generated/unknown; physics backup before AI. See `docs/SYSTEM_DESIGN.md`.

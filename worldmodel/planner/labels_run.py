"""Run the outcome simulation over many worlds (resumable: finished worlds are skipped). Test sites are labelled too (they are needed for the exam),
but the planner's train.py never reads them.

    python -m worldmodel.planner.labels_run --max-minutes 180 --cells 12
"""

from __future__ import annotations

import argparse

from contracts.mission import load_mission
from worldmodel.hazard.lola import HOLDOUT_SITES, TRAIN_SITES, VAL_SITES, regions_for_site
from worldmodel.lander.pilot import pick_device
from worldmodel.planner.labels import generate

DATES = ("2027-03-01", "2027-06-01", "2027-09-01", "2027-12-01")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites", nargs="+", default=VAL_SITES + HOLDOUT_SITES + TRAIN_SITES)
    ap.add_argument("--dates", nargs="+", default=list(DATES))
    ap.add_argument("--windows-per-site", type=int, default=1)
    ap.add_argument("--cells", type=int, default=12)
    ap.add_argument("--n-land", type=int, default=48)
    ap.add_argument("--n-rover", type=int, default=32)
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true", help="accepted for the runner; finished worlds are always skipped")
    a = ap.parse_args(argv)
    import os

    import torch

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 4))          # several label processes share the CPU with the pilot trainer
    worlds = [(r, d) for s in a.sites for r in regions_for_site(s)[: a.windows_per_site] for d in a.dates]
    generate(worlds, load_mission(), pick_device(a.device), a.max_minutes, cells=a.cells, n_land=a.n_land, n_rover=a.n_rover)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

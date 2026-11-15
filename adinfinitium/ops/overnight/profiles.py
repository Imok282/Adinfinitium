"""Stage and profile definitions for the overnight runner. Every command is a python module run as `python -s -m <module> <args>`.

Profiles (decided with Imok):
  smoke    tiny settings for every stage: proves the kit works end to end in minutes (run this FIRST on every machine)
  real4090 the 4090 PC (Windows, PyTorch only): rover driver (vision transformer) and terrain diffusion (attention U-Net, ~112 M). The lander pilot joins
           when its PyTorch version exists.
  colab    free Colab T4 (may disconnect: everything resumes): data from NASA's fast network, light skyline transformer, hazard attention U-Net
  macmini  Mac mini (64 GB unified memory): data, light transformer, hazard attention U-Net, evaluations
Budgets are wall-clock minutes per stage (UNTESTED estimates; the ledger records the real durations). A stage killed at its budget keeps its last checkpoint
and resumes next time (all training scripts save a resumable checkpoint).
"""

from __future__ import annotations

from dataclasses import dataclass, field

MONTHS = [f"2027-{m:02d}-01" for m in range(1, 13)]
SITES_GROUP_A = ["Site23", "Site42", "Site11"]
SITES_GROUP_B = ["Site01", "Site04", "Site06"]
SITES_GROUP_C = ["Site07", "Site20", "Haworth", "Shoemaker", "DM2"]


@dataclass
class Stage:
    name: str
    cmds: list[list[str]]                  # each: [module, *args]; several commands in one stage run in parallel (up to `parallel`)
    budget_min: float = 60.0
    parallel: int = 1
    train: bool = False                    # training stages get --resume appended when a checkpoint exists
    note: str = ""
    shrink_first: bool = False             # under --deadline this training stage gives up its time before the others
    skip_below_min: float = 0.0            # ... and is skipped when less than this many minutes would be left for it


@dataclass
class Profile:
    name: str
    stages: list[Stage] = field(default_factory=list)
    fresh: bool = False                    # training always starts from scratch (old checkpoints of these names are deleted)
    sandbox: str = ""                      # outputs go under <ADI_RUNS>/<sandbox>/ so they can't be mistaken for a real run's

    def names(self) -> list[str]:
        return [s.name for s in self.stages]


def _data_stages(grid: bool, sites_groups: list[list[str]], budget: float) -> list[Stage]:
    out = [Stage("kernels", [["worldmodel.light.fetch_kernels"]], 10, note="NASA/JPL SPICE kernels (46 MB)")]
    if grid:
        out.append(Stage("tiles", [["worldmodel.hazard.lola", "all", "--grid", "--workers", "3", "--sites", *[s for g in sites_groups for s in g]]], budget,
                         note="NASA LOLA 5 m windows; ONE process, at most 3 NASA connections in total (NASA answers HTTP 503 when hammered)"))
    out.append(Stage("psr", [["worldmodel.rover.psr"]], 15, note="NASA permanent-shadow map into every tile"))
    out.append(Stage("spatial_split", [["ops.spatial_split"]], 5, note="drops train/val windows that touch a test tile (+1.28 km ray range); must run before any training"))
    out.append(Stage("points", [["worldmodel.light.make_points", "--dates", *MONTHS]], 30, note="real SPICE + exact skyline light files (12 months)"))
    return out


def smoke() -> Profile:
    """About 23 min at most. Names can't collide with the seed's `*_smoke` checkpoints; always trained from scratch; outputs under runs/_smoke/."""
    sm = ["--name"]
    return Profile("smoke", [
        Stage("env_check", [["ops.overnight.env_check"]], 1),
        *[Stage(s.name, s.cmds, b) for s in _data_stages(False, [], 0) for n, b in (("kernels", 2), ("psr", 3), ("spatial_split", 1)) if s.name == n],
        Stage("hazard", [["worldmodel.hazard.train", "--arch", "attn", "--base", "16", "--depth", "4", "--blocks", "1", "--heads", "4", "--epochs", "1", "--steps", "3",
                          "--batch", "2", "--crop", "64", *sm, "smoketest_hazard"]], 2, train=True),
        Stage("hazard_eval", [["worldmodel.hazard.evaluate", *sm, "smoketest_hazard"]], 2),
        Stage("light", [["worldmodel.light.train", "--arch", "former", "--width", "8", "--d", "64", "--layers", "2", "--heads", "4", "--steps", "10", "--eval-every", "5",
                         "--cells", "200", "--val-cells", "100", "--batch", "8", *sm, "smoketest_light"]], 2, train=True),
        Stage("light_eval", [["worldmodel.light.evaluate", *sm, "smoketest_light", "--cells", "100"]], 2),
        # depth 3 like the real profile: at depth 2 self-attention runs on 64x64 tokens and validation needs 32 GiB on MPS
        Stage("terrain", [["worldmodel.terrain.train", "--arch", "attn", "--base", "16", "--depth", "3", "--blocks", "1", "--heads", "4", "--steps", "6", "--eval-every", "3",
                           "--batch", "8", *sm, "smoketest_terrain"]], 2, train=True),
        Stage("terrain_eval", [["worldmodel.terrain.evaluate", *sm, "smoketest_terrain", "--patches", "4", "--versions", "2", "--steps", "3"]], 2),
        Stage("rover", [["worldmodel.rover.train", "--arch", "former", "--d", "64", "--layers", "2", "--heads", "4", "--envs", "32", "--updates", "2", "--horizon", "8",
                         "--sites", "Site01", "--dates", "2027-03-01", "--hazard-name", "smoketest_hazard", *sm, "smoketest_rover"]], 2, train=True),
        Stage("rover_eval", [["worldmodel.rover.evaluate", *sm, "smoketest_rover", "--sites", "Site23", "--dates", "2027-03-01", "--n", "32",
                              "--hazard-name", "smoketest_hazard"]], 3),
    ], fresh=True, sandbox="_smoke")


def real4090() -> Profile:
    return Profile("real4090", [
        Stage("env_check", [["ops.overnight.env_check", "--need-cuda"]], 2),
        *_data_stages(True, [SITES_GROUP_A, SITES_GROUP_B, SITES_GROUP_C], 120),
        Stage("rover", [["worldmodel.rover.train", "--arch", "former", "--d", "512", "--layers", "10", "--heads", "8", "--envs", "1024", "--horizon", "32", "--epochs", "2",
                         "--updates", "1500", "--windows-per-site", "2", "--name", "rover_driver_big"]], 200, train=True,
              note="vision-transformer driver ~33 M; 1024 rovers x 32 steps per update (4096 x 64 would take DAYS: see FLOP math in the master plan); --max-minutes sizes the update count"),
        Stage("rover_eval", [["worldmodel.rover.evaluate", "--name", "rover_driver_big", "--n", "1024"]], 30),
        Stage("terrain", [["worldmodel.terrain.train", "--arch", "attn", "--base", "192", "--depth", "3", "--blocks", "2", "--heads", "8", "--steps", "20000",
                           "--batch", "64", "--eval-every", "500", "--name", "terrain_diffusion_big"]], 200, train=True, note="attention diffusion U-Net ~112 M"),
        Stage("terrain_eval", [["worldmodel.terrain.evaluate", "--name", "terrain_diffusion_big", "--patches", "128", "--versions", "8"]], 30),
    ])


def colab() -> Profile:
    return Profile("colab", [
        Stage("env_check", [["ops.overnight.env_check"]], 2),
        *_data_stages(True, [SITES_GROUP_A, SITES_GROUP_B, SITES_GROUP_C], 90),
        Stage("light", [["worldmodel.light.train", "--arch", "former", "--width", "64", "--d", "512", "--layers", "12", "--heads", "8", "--steps", "6000",
                         "--cells", "6000", "--name", "light_skyline_big"]], 150, train=True, note="skyline transformer ~38 M"),
        Stage("light_eval", [["worldmodel.light.evaluate", "--name", "light_skyline_big"]], 20),
        Stage("hazard", [["worldmodel.hazard.train", "--arch", "attn", "--base", "72", "--depth", "4", "--blocks", "2", "--heads", "8", "--epochs", "40", "--steps", "200",
                          "--batch", "12", "--name", "hazard_unet_big"]], 240, train=True, note="attention U-Net ~60 M"),
        Stage("hazard_eval", [["worldmodel.hazard.evaluate", "--name", "hazard_unet_big", "--figure"]], 20),
    ])


def macmini() -> Profile:
    """Mac mini (M5, 64 GB unified memory, stays on overnight): everything at MINI size, sized to finish in about 10 hours."""
    return Profile("macmini", [
        Stage("env_check", [["ops.overnight.env_check"]], 2),
        *_data_stages(True, [SITES_GROUP_A, SITES_GROUP_B, SITES_GROUP_C], 90),
        Stage("light", [["worldmodel.light.train", "--arch", "former", "--width", "64", "--d", "512", "--layers", "12", "--heads", "8", "--steps", "6000",
                         "--cells", "6000", "--name", "light_skyline_big"]], 60, train=True, note="skyline transformer ~38 M"),
        Stage("light_eval", [["worldmodel.light.evaluate", "--name", "light_skyline_big"]], 15),
        Stage("hazard", [["worldmodel.hazard.train", "--arch", "attn", "--base", "72", "--depth", "4", "--blocks", "2", "--heads", "8", "--epochs", "40", "--steps", "200",
                          "--batch", "12", "--name", "hazard_unet_big"]], 120, train=True, note="attention U-Net ~60 M"),
        Stage("hazard_eval", [["worldmodel.hazard.evaluate", "--name", "hazard_unet_big", "--figure"]], 15),
        Stage("rover", [["worldmodel.rover.train", "--arch", "former", "--d", "384", "--layers", "6", "--heads", "8", "--envs", "512", "--horizon", "32", "--epochs", "2",
                         "--updates", "800", "--windows-per-site", "1", "--hazard-name", "hazard_unet_big", "--name", "rover_driver_mini"]], 150, train=True,
              note="mini vision-transformer driver ~9 M; sees tonight's hazard map (slope-rule backup if that checkpoint is missing)"),
        Stage("rover_eval", [["worldmodel.rover.evaluate", "--name", "rover_driver_mini", "--n", "512", "--hazard-name", "hazard_unet_big"]], 30),     # measured 10.5 min for the 8 test worlds on this mini
        Stage("terrain", [["worldmodel.terrain.train", "--arch", "attn", "--base", "128", "--depth", "3", "--blocks", "2", "--heads", "8", "--steps", "8000", "--batch", "32",
                           "--eval-every", "500", "--name", "terrain_diffusion_mini"]], 125, train=True, note="mini attention diffusion U-Net ~50 M",
              shrink_first=True, skip_below_min=30),
        # measured on this mini: ~95 s per window, 25 windows tonight (19 of them test, scored first): 25 min more, taken from terrain training
        Stage("terrain_eval", [["worldmodel.terrain.evaluate", "--name", "terrain_diffusion_mini", "--patches", "96", "--versions", "6"]], 50),
    ])


PROFILES = {"smoke": smoke, "real4090": real4090, "colab": colab, "macmini": macmini}


def get_profile(name: str) -> Profile:
    if name not in PROFILES:
        raise SystemExit(f"unknown profile {name!r}; choose one of {sorted(PROFILES)}")
    return PROFILES[name]()

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


@dataclass
class Profile:
    name: str
    stages: list[Stage] = field(default_factory=list)

    def names(self) -> list[str]:
        return [s.name for s in self.stages]


def _data_stages(grid: bool, sites_groups: list[list[str]], budget: float) -> list[Stage]:
    out = [Stage("kernels", [["worldmodel.light.fetch_kernels"]], 10, note="NASA/JPL SPICE kernels (46 MB)")]
    if grid:
        out.append(Stage("tiles", [["worldmodel.hazard.lola", "all", "--grid", "--sites", *g] for g in sites_groups], budget, parallel=3,
                         note="NASA LOLA 5 m windows; NASA throttles with HTTP 503, so at most 3 parallel groups"))
    out.append(Stage("psr", [["worldmodel.rover.psr"]], 15, note="NASA permanent-shadow map into every tile"))
    out.append(Stage("points", [["worldmodel.light.make_points", "--dates", *MONTHS]], 30, note="real SPICE + exact skyline light files (12 months)"))
    return out


def smoke() -> Profile:
    sm = ["--name"]
    return Profile("smoke", [
        Stage("env_check", [["ops.overnight.env_check"]], 2),
        *[s for s in _data_stages(False, [], 0) if s.name in ("kernels", "psr")],
        Stage("hazard", [["worldmodel.hazard.train", "--arch", "attn", "--base", "16", "--depth", "4", "--blocks", "1", "--heads", "4", "--epochs", "1", "--steps", "3",
                          "--batch", "2", "--crop", "64", *sm, "hazard_smoke"]], 10, train=True),
        Stage("hazard_eval", [["worldmodel.hazard.evaluate", *sm, "hazard_smoke"]], 10),
        Stage("light", [["worldmodel.light.train", "--arch", "former", "--width", "8", "--d", "64", "--layers", "2", "--heads", "4", "--steps", "10", "--eval-every", "5",
                         "--cells", "200", "--val-cells", "100", "--batch", "8", *sm, "light_smoke"]], 10, train=True),
        Stage("light_eval", [["worldmodel.light.evaluate", *sm, "light_smoke", "--cells", "100"]], 10),
        Stage("terrain", [["worldmodel.terrain.train", "--arch", "attn", "--base", "16", "--depth", "3", "--blocks", "1", "--heads", "4", "--steps", "6", "--eval-every", "3",
                           "--batch", "8", *sm, "terrain_smoke"]], 10, train=True),
        Stage("terrain_eval", [["worldmodel.terrain.evaluate", *sm, "terrain_smoke", "--patches", "4", "--versions", "2", "--steps", "3"]], 10),
        Stage("rover", [["worldmodel.rover.train", "--arch", "former", "--d", "64", "--layers", "2", "--heads", "4", "--envs", "32", "--updates", "2", "--horizon", "8",
                         "--sites", "Site01", "--dates", "2027-03-01", *sm, "rover_smoke"]], 15, train=True),
        Stage("rover_eval", [["worldmodel.rover.evaluate", *sm, "rover_smoke", "--sites", "Site23", "--dates", "2027-03-01", "--n", "32"]], 15),
        Stage("lander", [["worldmodel.lander.train", "--arch", "mlp", "--envs", "32", "--horizon", "8", "--updates", "2", "--val-n", "16", "--eval-every", "1",
                          "--sites", "Site01", *sm, "lander_smoke"]], 10, train=True),
        Stage("lander_eval", [["worldmodel.lander.evaluate", *sm, "lander_smoke", "--n", "64", "--seeds", "1"]], 10),
        Stage("planner_labels", [["worldmodel.planner.labels_run", "--sites", "Site01", "Site23", "--dates", "2027-03-01", "--cells", "3", "--n-land", "8", "--n-rover", "8"]], 15),
        Stage("planner", [["worldmodel.planner.train", "--epochs", "20", *sm, "planner_smoke"]], 5),
        Stage("planner_eval", [["worldmodel.planner.evaluate", *sm, "planner_smoke"]], 5),
    ])


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
        Stage("lander", [["worldmodel.lander.train", "--arch", "former", "--d", "384", "--layers", "8", "--heads", "8", "--envs", "512", "--horizon", "96",
                          "--updates", "1000", "--windows-per-site", "2", "--name", "lander_pilot"]], 150, train=True,
              note="pilot brain: vision-transformer residual on ZEM/ZEV guidance (~15 M); best checkpoint by validation success vs classic"),
        Stage("lander_eval", [["worldmodel.lander.evaluate", "--name", "lander_pilot", "--n", "1024", "--seeds", "3"]], 30),
        Stage("planner_labels", [["worldmodel.planner.labels_run", "--cells", "16", "--max-minutes", "100"]], 110,
              note="meta layer labels: simulated lander + rover outcomes per landing cell and hour (resumable)"),
        Stage("planner", [["worldmodel.planner.train", "--epochs", "400"]], 15),
        Stage("planner_eval", [["worldmodel.planner.evaluate"]], 5),
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
                         "--updates", "800", "--windows-per-site", "1", "--name", "rover_driver_mini"]], 150, train=True, note="mini vision-transformer driver ~9 M"),
        Stage("rover_eval", [["worldmodel.rover.evaluate", "--name", "rover_driver_mini", "--n", "512"]], 20),
        Stage("terrain", [["worldmodel.terrain.train", "--arch", "attn", "--base", "128", "--depth", "3", "--blocks", "2", "--heads", "8", "--steps", "8000", "--batch", "32",
                           "--eval-every", "500", "--name", "terrain_diffusion_mini"]], 150, train=True, note="mini attention diffusion U-Net ~50 M"),
        Stage("terrain_eval", [["worldmodel.terrain.evaluate", "--name", "terrain_diffusion_mini", "--patches", "96", "--versions", "6"]], 25),
    ])


def brain4090() -> Profile:
    """4090 PC, 2 hours max (Imok): the pilot brain trains on the GPU WHILE the planner's outcome labels are simulated on the CPU cores (3 processes),
    then both exams run and a demo plan is written. Uses the seed data in the package (no NASA download). Test site: Site23 (Site42 tiles not in the seed)."""
    labels = ["worldmodel.planner.labels_run", "--cells", "16", "--n-land", "48", "--n-rover", "32", "--max-minutes", "88"]
    return Profile("brain4090", [
        Stage("env_check", [["ops.overnight.env_check", "--need-cuda"]], 2),
        Stage("brain", [["worldmodel.lander.train", "--arch", "former", "--d", "384", "--layers", "8", "--heads", "8", "--envs", "512", "--horizon", "96",
                         "--updates", "3000", "--val-n", "512", "--eval-every", "10", "--name", "lander_pilot"],
                        [*labels, "--sites", "Site11", "Site23", "Site01"],
                        [*labels, "--sites", "Site04", "Site06"],
                        [*labels, "--sites", "Site07", "Site20"]], 100, parallel=4, train=True,
              note="pilot (GPU, ~15 M vision transformer, residual on ZEM/ZEV) in parallel with 3 CPU label simulators; the pilot sizes itself to the budget"),
        Stage("lander_eval", [["worldmodel.lander.evaluate", "--name", "lander_pilot", "--n", "1024", "--seeds", "3"]], 10),
        Stage("planner", [["worldmodel.planner.train", "--epochs", "400"]], 4),
        Stage("planner_eval", [["worldmodel.planner.evaluate"]], 2),
        Stage("plan_demo", [["worldmodel.planner.plan", "--region", "lola5m_site23", "--date", "2027-03-01", "--run", "demo"]], 2),
    ])


PROFILES = {"smoke": smoke, "real4090": real4090, "colab": colab, "macmini": macmini, "brain4090": brain4090}


def get_profile(name: str) -> Profile:
    if name not in PROFILES:
        raise SystemExit(f"unknown profile {name!r}; choose one of {sorted(PROFILES)}")
    return PROFILES[name]()

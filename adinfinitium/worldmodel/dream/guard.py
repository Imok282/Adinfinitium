"""Safety rails for the experimental dream track: never run beside the main overnight run, never write into it, label every score honestly."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from worldmodel.hazard.lola import HOLDOUT_SITES, TRAIN_SITES, VAL_SITES

SIMULATOR_DIRS = ("contracts", "worldmodel/rover", "worldmodel/light", "worldmodel/hazard")
SIMULATOR_FILES = ("config/mission.yaml", "worldmodel/nn_blocks.py")
# The simulator never reads the site library (site lists are constants in worldmodel/hazard/lola.py; contracts.sites.load_sites has no fallback and is
# called only by the navigator and data_tools), and sites/sites.json is not in git. Compared when either side has it; absent on both is not a difference.
OPTIONAL_FILES = ("sites/sites.json", "sites/sites.example.json")
WRAPPERS = ("nohup", "sudo", "env", "caffeinate", "time", "exec")
SHELLS = ("bash", "sh", "zsh")
NOT_FROZEN_NOTE = "main run was NOT frozen (verify failed); starting driver is the trained checkpoint, not a frozen one"
NOT_FROZEN_NOTE_START_FILE = "main run was NOT frozen (verify failed)"       # with --start-driver-file the starting driver is not the main run's at all
START_DRIVER_LABEL = "rover v2, trained after the freeze"


def is_main_runner(command: str) -> bool:
    """True for the runner itself (`bash .../run_everything.sh`, `python -m ops.overnight.run_all`, also under nohup / caffeinate / env), not for a
    shell, editor or search that only mentions it."""
    tok = _program(command)
    if not tok:
        return False
    prog = Path(tok[0]).name
    if prog.lower().startswith("python"):
        return _python_module(tok) == "ops.overnight.run_all"
    rest = tok[1:]
    if prog not in SHELLS or (rest and rest[0].startswith("-") and "c" in rest[0]):       # `bash -c "..."` runs a command string, not the script
        return False
    args = [t for t in rest if not t.startswith("-")]                                      # `bash -x script` is still the script
    return bool(args) and Path(args[0]).name == "run_everything.sh"


def is_main_stage(command: str) -> bool:
    """True for a stage of the main run that may have outlived its runner: `python -m worldmodel.<family>.<anything>` (rover.train, hazard.evaluate,
    light.make_points ...). The dream's own `worldmodel.dream.*` is not one. The folder it runs from is not checked: any such process refuses."""
    tok = _program(command)
    if not tok or not Path(tok[0]).name.lower().startswith("python"):
        return False
    module = _python_module(tok) or ""
    return module.startswith("worldmodel.") and not module.startswith("worldmodel.dream.") and module != "worldmodel.dream"


def _program(command: str) -> list[str]:
    """The command's tokens without leading wrappers (nohup, caffeinate, env X=1 ...) and their options."""
    tok = command.split()
    while tok and (Path(tok[0]).name in WRAPPERS or tok[0].startswith("-") or ("=" in tok[0] and "/" not in tok[0].split("=")[0])):
        tok = tok[1:]
    return tok


def _python_module(tok: list[str]) -> str | None:
    """The module of `python ... -m module ...` (None for a script or `-c`)."""
    return next((t for i, t in enumerate(tok[1:]) if tok[i] == "-m"), None)


def main_runner_lines(ps_output: str, own_pid: int) -> list[str]:
    """Lines of `ps -axo pid=,command=` that are the main overnight runner or one of its stage processes (not this process)."""
    hits = []
    for line in ps_output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) != own_pid and (is_main_runner(parts[1]) or is_main_stage(parts[1])):
            hits.append(line.strip())
    return hits


def main_runner_now() -> list[str]:
    """Main-runner processes alive right now; a failed process listing counts as 'alive' (never continue blind)."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return [f"could not list processes ({e})"]
    return main_runner_lines(out, os.getpid())


def simulator_differences(repo: Path, main_run_dir: Path) -> list[str]:
    """Simulator files that are missing on either side or differ between this folder and the main run folder (an OPTIONAL_FILES entry that neither
    side has is not a difference)."""
    names = set(SIMULATOR_FILES) | {f for f in OPTIONAL_FILES if (repo / f).exists() or (main_run_dir / f).exists()}
    for root in (repo, main_run_dir):
        for d in SIMULATOR_DIRS:
            names |= {p.relative_to(root).as_posix() for p in (root / d).rglob("*.py")}
    out = []
    for f in sorted(names):
        a, b = sha256(repo / f), sha256(main_run_dir / f)
        if a is None or b is None or a != b:
            out.append(f)
    return out


def main_run_state(main_run_dir: Path, frozen_name: str | None = None, checkpoint: Path | None = None) -> dict:
    """Whether the main run froze. `runs/VERIFY_FAILED.txt` means it did NOT (run_everything.sh removes it before a freeze, so a FROZEN.json beside it
    is from an older run and is ignored). Otherwise FROZEN.json must carry the sha256 of the driver file this run starts from: `refuse` says why not."""
    runs = Path(main_run_dir).expanduser() / "runs"
    frozen_json, failed = runs / "FROZEN.json", runs / "VERIFY_FAILED.txt"
    state = {"frozen": False, "verify_failed": failed.exists(), "frozen_json_present": frozen_json.exists(), "frozen_at": None, "note": None, "refuse": None}
    if state["verify_failed"]:
        state["note"] = NOT_FROZEN_NOTE
        return state
    if not frozen_json.exists():
        state["refuse"] = f"{runs} has neither FROZEN.json nor VERIFY_FAILED.txt: the main run has not reached its freeze step"
        return state
    try:
        doc = json.loads(frozen_json.read_text(encoding="utf-8"))
        recorded, state["frozen_at"] = doc.get("sha256", {}), doc.get("frozen_at")
    except (OSError, ValueError, AttributeError) as e:
        state["refuse"] = f"{frozen_json} cannot be read ({e}): cannot tell whether the main run froze"
        return state
    if frozen_name is not None and checkpoint is not None and Path(checkpoint).exists():       # a missing driver file is refused by the caller
        want, have = recorded.get(f"checkpoint/{frozen_name}"), sha256(Path(checkpoint))
        if want != have:
            state["refuse"] = (f"{frozen_json} does not match the driver file {checkpoint}: recorded sha256 {str(want)[:12]}, file {str(have)[:12]}. "
                               "FROZEN.json is stale (from an older run) or the driver changed after the freeze")
            return state
    state["frozen"] = True
    return state


def start_driver_text(label: str, digest: str | None) -> str:
    """The one way a driver from --start-driver-file is named in the report, the ledger and the log."""
    return f"starting driver: {label}, sha256 {digest}"


def start_driver_state(start_driver_file: Path, main_run_dir: Path | None = None) -> dict:
    """A starting driver given as a file (trained after the freeze, outside the main run): it must exist, must not sit inside the main run folder and
    must not be a file the main run's FROZEN.json lists (that one is started with --frozen-name). Its sha256 is computed here; `refuse` says why not."""
    path = Path(start_driver_file).expanduser().resolve()
    state = {"path": str(path), "sha256": None, "refuse": None}
    if not path.is_file():
        state["refuse"] = f"the starting driver file {path} does not exist"
        return state
    state["sha256"] = sha256(path)
    if main_run_dir is not None:
        main = Path(main_run_dir).expanduser().resolve()
        if main in path.parents:
            state["refuse"] = (f"the starting driver file {path} sits inside the main run folder {main}: --start-driver-file is for a driver trained "
                               "after the freeze, kept outside it (the main run's own driver is started with --frozen-name)")
            return state
        try:
            listed = json.loads((main / "runs" / "FROZEN.json").read_text(encoding="utf-8")).get("sha256", {})
        except (OSError, ValueError, AttributeError):
            listed = {}
        names = [k for k, v in listed.items() if v == state["sha256"]] if isinstance(listed, dict) else []
        if names:
            state["refuse"] = (f"the starting driver file {path} is a copy of {names[0]} of the main run's freeze: it was not trained after the freeze "
                               "(start it with --frozen-name)")
    return state


def refusal_reason(main_run_dir: Path | None, repo: Path, ps_output: str | None = None, runs: Path | None = None, frozen_name: str | None = None,
                   checkpoint: Path | None = None, start_driver_file: Path | None = None) -> str | None:
    """Why the dream experiment must NOT start now (None = clear to start). With `frozen_name` and `checkpoint` the main run's FROZEN.json must
    match the driver file. With `start_driver_file` that one check is replaced by `start_driver_state`; every other refusal stays."""
    if start_driver_file is not None:
        frozen_name = checkpoint = None
    hits = main_runner_lines(ps_output, os.getpid()) if ps_output is not None else main_runner_now()
    if hits:
        return "the main overnight runner or one of its stage processes is still alive: " + " | ".join(h[:120] for h in hits[:3])
    if main_run_dir is not None:
        main_run_dir = Path(main_run_dir).expanduser().resolve()
        if not main_run_dir.is_dir():
            return f"main run folder {main_run_dir} does not exist"
        for what, p in (("this code", repo), ("the output folder", runs)):
            if p is not None and (Path(p).resolve() == main_run_dir or main_run_dir in Path(p).resolve().parents):
                return f"{what} sits inside the main run folder {main_run_dir}: ops.freeze hashes that folder, run the dream track from its own folder"
        if (main_run_dir / "runs" / "run_everything.lock").exists():
            return f"{main_run_dir}/runs/run_everything.lock exists: the main run has not finished"
        if not (main_run_dir / "runs" / "FROZEN.json").exists() and not (main_run_dir / "runs" / "VERIFY_FAILED.txt").exists():
            return f"{main_run_dir}/runs has neither FROZEN.json nor VERIFY_FAILED.txt: the main run has not reached its freeze step"
        state = main_run_state(main_run_dir, frozen_name, checkpoint)
        if state["refuse"]:
            return state["refuse"]
        why = start_driver_state(start_driver_file, main_run_dir)["refuse"] if start_driver_file is not None else None
        if why:
            return why
        differ = simulator_differences(repo, main_run_dir)
        if differ:
            return (f"{len(differ)} simulator files differ from the main run or are missing ({', '.join(differ[:4])}...): scores would not be comparable "
                    "with the frozen driver's")
    elif start_driver_file is not None:
        return start_driver_state(start_driver_file)["refuse"]
    return None


def site_role(site: str) -> str:
    return "train" if site in TRAIN_SITES else "validation" if site in VAL_SITES else "test" if site in HOLDOUT_SITES else "other"


def claim(site: str, real_steps_from_site: int) -> str:
    """The only claim a score on `site` may carry, given how much of its own experience on that site the driver has learned from."""
    role = site_role(site)
    if real_steps_from_site > 0:
        return f"online self-learning on that site ({real_steps_from_site} real steps there): NOT a held-out score"
    if role == "test":
        return "held-out (zero experience from this site)"
    return f"{role} site, no self-learning yet: not a held-out score"


def sha256(path: Path) -> str | None:
    if not Path(path).exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

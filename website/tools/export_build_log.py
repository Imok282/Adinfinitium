"""Export the project's git history as a small, privacy-safe JSON for the website build-log section.

Run from anywhere (python3 stdlib only, read-only on the repo):
    python3 website/tools/export_build_log.py

Writes website/assets/build_log.json. Privacy rules:
  * no e-mail addresses and no full names are written; authors are mapped to the first names used
    in branch prefixes (imok, nimansh, tanush, arnav), the coding agent's own commits to "team",
    anything else to "team";
  * commit messages are cut to their first line (<= 110 chars) and scrubbed of absolute paths,
    home directories, e-mail addresses and token-like strings;
  * only top-level module groupings, file counts and line counts are exported, never file paths.
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "website" / "assets" / "build_log.json"

FIRST_NAMES = ("imok", "nimansh", "tanush", "arnav")
SKIP_BRANCHES = {"main", "HEAD"}
MAX_MSG = 110
IST = timezone(timedelta(hours=5, minutes=30))
# Generated data / binary assets: counted as files, but their lines are not counted as written code.
DATA_EXT = re.compile(r"\.(json|obj|mtl|glb|gltf|png|jpe?g|svg|csv|npz|npy|blend\d?|bin|ipynb)$", re.I)

# (id, label) in the order the diagram shows them.
MODULES = [
    ("lander_sim", "Lander sim"),
    ("hazard", "Hazard model"),
    ("light", "Light model"),
    ("terrain", "Terrain"),
    ("planner", "Planner"),
    ("pilot", "Pilot"),
    ("rover", "Rover driver"),
    ("dream_rsi", "Dream-RSI"),
    ("apps", "Navigator / apps"),
    ("contracts", "Contracts / config"),
    ("ops", "Run kits / ops"),
    ("docs", "Docs"),
]

# Ordered path rules: first matching prefix/regex wins for a file.
PATH_RULES = [
    (r"(^|/)dream\.py$|(^|/|_)rsi[_.]|^swarm/|test_swarm", "dream_rsi"),
    (r"^worldmodel/hazard/|test_hazard|test_lola", "hazard"),
    (r"^worldmodel/light/|make_demo_light|test_light|test_ephemeris|test_demo_light|test_multiwindow", "light"),
    (r"^worldmodel/terrain/|^data_tools/|test_terrain", "terrain"),
    (r"^worldmodel/planner/|^worldmodel/budget|test_plan|test_mission_brain", "planner"),
    (r"^worldmodel/lander/(pilot|train|evaluate|benchmark|solve|analysis)\.py|test_lander_torch|^worldmodel/nn_blocks|test_large_models", "pilot"),
    (r"^worldmodel/rover/|test_rover", "rover"),
    (r"^sim_jax/|^worldmodel/lander/|^basilisk_mods/|^scripts/bench|^worldmodel/exams|test_sim_jax|test_exams|test_basilisk", "lander_sim"),
    (r"^(navigator|mission_control|brain_window|model_viewer|vizard)/|^worldmodel/ghosts|test_navigator|test_model_viewer|test_vehicles", "apps"),
    (r"^(contracts|config|sites)/|test_contracts|^pyproject|^requirements|^Makefile|^\.github/", "contracts"),
    (r"^ops/|test_ops", "ops"),
    (r"^docs/|\.md$|^LICENSE", "docs"),
]
PATH_RULES = [(re.compile(p), m) for p, m in PATH_RULES]


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout


def module_of(path: str) -> str | None:
    for rx, mod in PATH_RULES:
        if rx.search(path):
            return mod
    return None


def lane_of(author: str, is_agent_author: bool) -> str:
    low = author.lower()
    if is_agent_author:
        return "team"
    for name in FIRST_NAMES:
        if low.startswith(name):
            return name
    return "team"


_SCRUB = [
    (re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+"), "[email]"),
    (re.compile(r"(~/|/Users/|/home/|/private/|/tmp/|/var/|[A-Za-z]:\\)[^\s,;)]*"), "[path]"),
    (re.compile(r"\b(ghp|gho|github_pat|sk|hf|xox[abp])[-_][A-Za-z0-9_-]{8,}"), "[token]"),
    (re.compile(r"\b[A-Za-z0-9_-]{32,}\b"), "[token]"),
]


def scrub(msg: str) -> str:
    msg = msg.splitlines()[0].strip() if msg else ""
    for rx, rep in _SCRUB:
        msg = rx.sub(rep, msg)
    if len(msg) > MAX_MSG:
        msg = msg[: MAX_MSG - 1].rstrip() + "…"
    return msg


def branch_sets() -> dict[str, set[str]]:
    """Short branch name -> set of full shas on that branch's first-parent line but not on main."""
    refs = git("for-each-ref", "--format=%(refname:short)", "refs/heads", "refs/remotes").split()
    out: dict[str, set[str]] = {}
    for ref in refs:
        name = ref.split("/", 1)[1] if ref.startswith("origin/") else ref
        if name in SKIP_BRANCHES or ref == "origin":
            continue
        shas = set(git("rev-list", "--first-parent", ref, "--not", "main").split())
        out.setdefault(name, set()).update(shas)
    return out


def main() -> None:
    sep = "\x1e"
    raw = git("log", "--all", "--date-order", "--numstat",
              f"--format={sep}%H%x1f%aI%x1f%an%x1f%P%x1f%s%x1f%b")
    branches = branch_sets()
    commits = []
    for block in raw.split(sep)[1:]:
        head, _, rest = block.partition("\n")
        # The body may contain newlines; re-join everything up to the numstat lines.
        fields = (head + "\n" + rest).split("\x1f")
        sha, date, author, parents, subject = fields[:5]
        body_and_stat = fields[5] if len(fields) > 5 else ""
        stat_lines = [ln for ln in body_and_stat.splitlines() if re.match(r"^(\d+|-)\t(\d+|-)\t", ln)]
        body = body_and_stat
        is_agent_author = author.strip().lower() == "claude"
        co_authored = "co-authored-by: claude" in body.lower()
        add = dele = 0
        mods: dict[str, int] = {}
        for ln in stat_lines:
            a, d, path = ln.split("\t", 2)
            path = path.split(" => ")[-1].strip("{}")
            a = int(a) if a.isdigit() and not DATA_EXT.search(path) else 0
            d = int(d) if d.isdigit() and not DATA_EXT.search(path) else 0
            add += a
            dele += d
            m = module_of(path)
            if m:
                mods[m] = mods.get(m, 0) + a + d + 1
        on = sorted((b for b, s in branches.items() if sha in s), key=lambda b: len(branches[b]))
        lane = lane_of(author, is_agent_author)
        # Prefer a branch whose prefix matches the lane, then the most specific (smallest) one.
        pref = [b for b in on if b.startswith(lane + "/")] or on
        branch = pref[0] if pref else None
        if branch and "/" not in branch and branch not in ("nimwork",):
            branch = None
        commits.append({
            "t": datetime.fromisoformat(date).astimezone(IST).isoformat(),
            "lane": lane,
            "branch": branch,
            "msg": scrub(subject),
            "modules": [m for m, _ in sorted(mods.items(), key=lambda kv: -kv[1])],
            "files": len(stat_lines),
            "add": add,
            "del": dele,
            "sha": sha[:7],
            "merge": len(parents.split()) > 1,
            "agent": is_agent_author or co_authored,
        })
    commits.sort(key=lambda c: datetime.fromisoformat(c["t"]))

    mod_counts = {m: 0 for m, _ in MODULES}
    for c in commits:
        for m in c["modules"]:
            mod_counts[m] += 1

    per_day: list[dict] = []
    for c in commits:
        day = c["t"][:10]
        if not per_day or per_day[-1]["day"] != day:
            per_day.append({"day": day, "commits": 0, "agent_commits": 0})
        per_day[-1]["commits"] += 1
        per_day[-1]["agent_commits"] += int(c["agent"])

    lane_order = [n for n in FIRST_NAMES if any(c["lane"] == n for c in commits)] + (
        ["team"] if any(c["lane"] == "team" for c in commits) else [])
    lanes = []
    for ln in lane_order:
        bs = sorted({c["branch"] for c in commits if c["lane"] == ln and c["branch"]})
        label = "AI coding agents (shared skeleton)" if ln == "team" else f"{ln.capitalize()} + AI coding agents"
        lanes.append({"id": ln, "label": label, "branches": bs,
                      "commits": sum(c["lane"] == ln for c in commits)})

    data = {
        "meta": {
            "provenance": "observed",
            "source": "git log of the project repository",
            "frame": "commit author time, ISO 8601, converted to IST (+05:30)",
            "units": ("lines added/deleted per git numstat over text/code files; generated data and "
                      "binary assets (json, obj, glb, png, svg, csv, ...) count as files but 0 lines"),
            "generated_at": datetime.now().astimezone().replace(microsecond=0).isoformat(),
            "note": ("Every commit in the repository (all branches). Lanes are first names from branch "
                     "prefixes; e-mails and full names removed; messages cut to the first line. "
                     "'agent' = authored or co-authored by an AI coding agent. Modules are inferred from "
                     "the touched paths."),
            "totals": {
                "commits": len(commits),
                "agent_commits": sum(c["agent"] for c in commits),
                "add": sum(c["add"] for c in commits),
                "del": sum(c["del"] for c in commits),
                "files": sum(c["files"] for c in commits),
                "branches": len({c["branch"] for c in commits if c["branch"]}),
            },
        },
        "modules": [{"id": m, "label": lab, "commits": mod_counts[m]} for m, lab in MODULES],
        "lanes": lanes,
        "per_day": per_day,
        "commits": commits,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}: {OUT.stat().st_size / 1024:.1f} KB, "
          f"{len(commits)} commits, {len(lanes)} lanes, {len(MODULES)} modules")


if __name__ == "__main__":
    main()

"""Why rover v2 must NOT start on the Mac mini right now (exit 3 with the reason), or 'clear' (exit 0).

    python -m worldmodel.rover2.preflight --main-run-dir ~/adinfinitium-run/adinfinitium

Refuses while the main runner or any stage process is alive (also a second rover v2 run), before the main run reached its freeze step, before its results
zip exists, when this folder or its output is inside the main run folder, or when the simulator files here differ from the main run's.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from contracts.paths import data_root, repo_root, runs_root
from worldmodel.dream.guard import main_run_state, refusal_reason

RESULTS_ZIP = "adinfinitium_results_macmini.zip"


def reason(main_run_dir: Path, repo: Path, runs: Path, data: Path, ps_output: str | None = None, need_zip: bool = True) -> str | None:
    why = refusal_reason(main_run_dir, repo, ps_output=ps_output, runs=runs)
    if why:
        return why
    main = Path(main_run_dir).expanduser().resolve()
    if data.resolve() == main or main in data.resolve().parents:
        return f"the data folder {data} is inside the main run folder {main}: rover v2 would write checkpoints and caches there"
    if need_zip and not (main / RESULTS_ZIP).exists():
        return f"{main / RESULTS_ZIP} does not exist yet: the main run has not been zipped"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main-run-dir", required=True)
    ap.add_argument("--no-zip-check", action="store_true", help="do not require the main results zip (say why in the report)")
    a = ap.parse_args(argv)
    why = reason(Path(a.main_run_dir), repo_root(), runs_root(), data_root(), need_zip=not a.no_zip_check)
    if why:
        print(f"REFUSED: {why}")
        return 3
    state = main_run_state(Path(a.main_run_dir))
    print(f"clear: main run frozen={state['frozen']} frozen_at={state['frozen_at']} verify_failed={state['verify_failed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

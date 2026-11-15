# Changes for rover v2 (branch `rover-v2`, on top of f1ab06e = main code 07eb0f0 + the dream addendum)

Rover v2 is trained AFTER the main run's freeze. It is not part of the frozen main results. Rover v1's failed result stays as reported.

## Not changed (checked by the launcher against the main run folder before it starts)

`worldmodel/rover/*` (simulator physics, reward, planners, PPO trainer, evaluator), `worldmodel/light/*`, `worldmodel/hazard/*`, `contracts/*`,
`config/mission.yaml`, `worldmodel/nn_blocks.py`. The held-out evaluation is `python -m worldmodel.rover.evaluate --name rover_driver_v2 --n 512
--hazard-name hazard_unet_big`: the main run's evaluator file and arguments, only the checkpoint name differs.

## Added

| file | what |
|---|---|
| `ROVER_V2_PLAN.md` | pre-registration, committed (d02c090) before any v2 number existed |
| `worldmodel/rover2/teach.py` | the new learning method: imitation of the physics planners with DAgger; per-rover teacher = best of 6 variants of `RoverEnv.planner_action` by simulated episode return; candidates scored on Site11 by full greedy episodes; refuses Site23/Site42 |
| `worldmodel/rover2/report.py` | `ROVER_V2_REPORT.md`: v1, v2, planners side by side, Site11 selection record, pre-registered verdict |
| `worldmodel/rover2/preflight.py` | refusal checks before a run on the mini (reuses the dream guard) plus "results zip exists" |
| `ops/rover2/run_rover2_mini.sh` | the one launcher for the mini |
| `ops/rover2/make_overlay.sh` | data folder of symlinks into the main run's data, so reads come from the main run and every write lands in the v2 folder |
| `tests/test_rover2.py` | teacher assignment, test-site wall, selection rule, verdict thresholds, preflight refusals, incomplete evaluation = no verdict |

## Changed (dream addendum only)

`worldmodel/dream/guard.py`, `worldmodel/dream/run.py`, `ops/dream/run_dream_mini.sh`, `tests/test_dream.py`: the self-learning run can start from a driver
trained after the freeze (`START_DRIVER_FILE=...`); it is labelled "starting driver: rover v2, trained after the freeze, sha256 ..." and never "frozen".
Default behaviour without the option is unchanged.

## Differences from v1 training, all of them

| | v1 (`worldmodel.rover.train`) | v2 (`worldmodel.rover2.teach`) |
|---|---|---|
| learning signal | PPO reward from its own random exploration | cross-entropy to the planner teacher's action (DAgger) |
| teacher | none | best of 6 planner variants per rover, picked by simulating each on the train-site world |
| episodes | 32-step rollouts | full 300-step episodes |
| training examples | every step of its own rollouts | teacher-labelled states; planner-stuck "stop" states left out, standing rovers stored 1/8 as often (plan, Amendment 1) |
| checkpoint choice | training-rollout score (picked the update-1 network) | Site11 validation, full greedy episodes, `returned` |
| network, train worlds, hazard map, reward, physics | same | same |
| value head | trained | NOT trained (v2 uses only the action head; anything that needs the value head must train it first) |
| `TeachEnv` | - | subclass of `RoverEnv` that can skip building the observation while planners drive; dynamics and reward are the parent's code |

## Not done

No PPO fine-tune, no network + planner hybrid (see the plan). No training of the value head.

## How to deploy and run on the Mac mini (after the main run has frozen and zipped)

From the MacBook (code only; never copies data, runs, zips, .git or a venv; the mini reuses the main run's `.venv` and reads its `data/` through symlinks):

    cd ~/Documents/Ojas/Hackathon/work/adinfinitium-rover2
    rsync -a --delete --exclude '.git' --exclude '.venv' --exclude '*.zip' --exclude '__pycache__' --exclude '.pytest_cache' \
          --exclude '/data' --exclude '/runs' --exclude 'ROVER2_COMMIT.txt' ./ mini:adinfinitium-rover2/
    ssh mini "echo $(git rev-parse HEAD) > ~/adinfinitium-rover2/ROVER2_COMMIT.txt"

On the mini, in this order:

    cd ~/adinfinitium-rover2
    nohup bash ops/rover2/run_rover2_mini.sh 110 > runs/_rover2/launcher.out 2>&1 &      # refuses (exit 3) until the main run is frozen, zipped and gone
    tail -f runs/_rover2/launcher.log runs/_rover2/train.log                             # progress; candidates on Site11 appear about every 12 min
    cat runs/_rover2/ROVER_V2_REPORT.md                                                  # when "rover v2 done" is in launcher.log (about 2 h later)

    # then the self-learning addendum, starting from the v2 driver (new run id; labelled "rover v2, trained after the freeze, sha256 ..."):
    START_DRIVER_FILE=$HOME/adinfinitium-rover2/data/models/rover_driver_v2.pt nohup bash ops/dream/run_dream_mini.sh 140 selflearn_v2 > /dev/null 2>&1 &
    cat runs/_dream/selflearn_v2/report.md

Stop a run: `pkill -f worldmodel.rover2.teach` (training; the launcher then evaluates the best Site11 candidate so far, or exits 2 if there is none),
`pkill -f run_rover2_mini.sh` first to stop everything. Running the launcher again resumes. Everything rover v2 writes is under `~/adinfinitium-rover2/`:
`data/models/rover_driver_v2*.pt`, `data/rover_cache/lola5m_site11*`, `runs/_rover2/`.

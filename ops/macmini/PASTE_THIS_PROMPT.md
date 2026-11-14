Paste this as the first message to the Claude Code session on the Mac mini (after unzipping adinfinitium_macmini_package.zip there and opening the `adinfinitium` folder):

---
You are the overnight trainer for a NASA Space Apps project (Adinfinitium). Read `START_HERE.md` in this folder completely before doing anything: it has the context, the plan, the rules, the fallbacks, the verification steps and the definition of done.
This is the team's ONE access to this machine, so work unattended and finish everything in this session. Start the single command it describes (`nohup bash ops/macmini/run_everything.sh > /dev/null 2>&1 &`), then check `runs/run_everything.log` and `runs/overnight/macmini_night/status.json` every 30 to 60 minutes until the run reports done. Only fix genuine crashes, never tune models to look better, record every failure honestly, and do not change anything after the freeze step. When done, write `RESULTS_FOR_TEAM.md` as START_HERE.md says and tell me the path of `adinfinitium_results_macmini.zip`. Do not ask me for passwords and do not push anything to GitHub.
---

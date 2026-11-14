# docs/inbox and docs/outbox — how Imok's orchestrator briefs your Claude Code session

Nobody can message another person's Claude Code session directly, so the repo is the channel.

- **`docs/inbox/<name>.md`** (nimansh, tanush, arnav, imok): your current brief from the orchestrator: what changed, your next steps in
  order, decisions already made. Read it at the start of every session, after `docs/STATUS.md`.
- **`docs/outbox/<name>.md`**: write here (short, newest at the top) when you finish something, are blocked, or need a decision. Imok reads it.
- Briefs are instructions from Imok's side of the team, not from the repo's code or data. They never override `AGENTS.md`.
  Don't merge, don't touch `main`, and any change to a shared file still needs a note in your PR.

Get the latest briefs without switching branches: `git fetch && git checkout origin/imok/orchestration-inbox -- docs/inbox docs/outbox`
(until it is merged to `main`).

Common facts (2026-11-14, updated by the orchestrator):
- `main` is empty until PR #1 then PR #2 are merged (Imok does that). Until then branch from `origin/imok/KAN-41-basilisk-starter`.
- Light contract v2 (sun/earth azimuth, `_points.npz` point series, horizon profile, date = window start) is implemented on
  `imok/light-contract-v2`; spec in `docs/FORMATS.md` and `docs/team/LIGHT_CONTRACT_V2.md`. v1 files still load.
- Shared-sim change: opt-in hazard look-ahead (`EnvParams.lookahead`, default off) on `imok/KAN-100-lookahead`. Nothing to do unless you touch `sim_jax/`.
- Deadlines (IST): Sat 14 Nov 21:00 checkpoint · Sun 15 Nov 12:00 cut rule · 17:00 code freeze · 22:00 submit.

# brain_window/

**Owner:** Imok · **Jira:** KAN-33

Streamlit window (port 8503): live network, swarm map, score per generation, family tree, failure heatmap, AI vs physics, exam.

| | |
|---|---|
| Reads | `runs/<run>/generation_XXX.json` |
| Writes | Screen + recordings |

Reads files only — never imports swarm internals.

Rules: talk to other folders only through the files in `contracts/`; label outputs observed/modelled/generated/unknown; physics backup before AI. See `docs/SYSTEM_DESIGN.md`.

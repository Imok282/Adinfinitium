# navigator/

**Owner:** Nimansh · **Jira:** KAN-35

Streamlit app (port 8501): sky dome, compare sites/dates, lunar-day timeline, ice context.

| | |
|---|---|
| Reads | tiles, light fields, `sites/sites.json` |
| Writes | Screens; can send a site/date to the swarm |

Exact physics only — no AI needed for the Navigator to answer the challenge brief.

Rules: talk to other folders only through the files in `contracts/`; label outputs observed/modelled/generated/unknown; physics backup before AI. See `docs/SYSTEM_DESIGN.md`.

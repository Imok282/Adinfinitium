# mission_control/

**Owner:** Arnav · **Jira:** KAN-48

Streamlit window (port 8502): live telemetry, power/comms, phase, decision log.

| | |
|---|---|
| Reads | `runs/<run>/telemetry_XXX.csv` |
| Writes | Screen |

Decision log shows facts only (what/when/why), no commentary.

Rules: talk to other folders only through the files in `contracts/`; label outputs observed/modelled/generated/unknown; physics backup before AI. See `docs/SYSTEM_DESIGN.md`.

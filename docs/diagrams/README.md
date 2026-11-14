# Diagrams

Sources are `.dot` (Graphviz) and `.mmd` (Mermaid). Re-render all of them with `make diagrams`
(needs `graphviz` and `npm i -g @mermaid-js/mermaid-cli`).

| # | Diagram | Shows |
|---|---|---|
| 01 | [System architecture](01_system_architecture.png) | data → AI models → physics engines → screens |
| 02 | [Mission phases](02_mission_phases.png) | orbit → descent → touchdown → rover → night → sunrise |
| 03 | [Evolution loop](03_evolution_loop.png) | swarm generations + Dream-RSI meta layer |
| 04 | [Bot I/O](04_bot_io.png) | lander pilot sensors in, commands out |
| 05 | [Workflow](05_workflow.png) | laptops → GitHub → 4090, pen drive |
| 06 | [Schedule](06_schedule.png) | 48-hour Gantt with checkpoints |
| 07 | [Components & contracts](07_components_and_contracts.png) | every folder and the files between them |
| 08 | [Deployment](08_deployment.png) | machines, venvs, data paths, ports |
| 09 | [Generation sequence](09_generation_sequence.png) | one swarm generation end to end |
| 10 | [Data model](10_data_model.png) | contract schemas and how they relate |
| 11 | [Lander state machine](11_lander_state_machine.png) | guidance phases and every outcome code |
| 12 | [Ownership](12_ownership.png) | who owns which Jira tasks and the hand-off files |

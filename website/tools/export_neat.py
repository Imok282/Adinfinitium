"""Convert a TensorNEAT swarm run (swarm/run_swarm.py) into website/assets/data/neat_training.json (format: top of website/js/neat.js).

    python3 website/tools/export_neat.py runs/<run>          # writes neat_training.json with "placeholder": false

Reads every runs/<run>/generation_XXX.json (contracts/runs.py GenerationLog): generation, population, best_score, mean_score,
champion_genome {"nodes": [[...]], "conns": [[...]]} (TensorNEAT arrays; unused rows are NaN), and optional species counts
(`species`, `n_species` or strategy.species). Assumed TensorNEAT layout (DefaultGenome): node row = [key, bias, ...],
conn row = [in_key, out_key, ..., weight] (weight = last column). Input nodes = keys that are never a connection target,
output nodes = keys that are never a connection source; check these against the run's genome config before publishing.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "assets" / "data" / "neat_training.json"


def finite_rows(a):
    return [r for r in (a or []) if r and all(isinstance(v, (int, float)) and math.isfinite(v) for v in r[:2])]


def main(run_dir: str) -> None:
    run = Path(run_dir)
    files = sorted(run.glob("generation_*.json"))
    if not files:
        sys.exit(f"no generation_*.json in {run}")
    gens, last = [], None
    for f in files:
        d = json.loads(f.read_text())
        genome = d.get("champion_genome") or {}
        nodes, conns = finite_rows(genome.get("nodes")), finite_rows(genome.get("conns"))
        species = d.get("species", d.get("n_species", (d.get("strategy") or {}).get("species")))
        gens.append({"generation": d["generation"], "best_fitness": d["best_score"], "mean_fitness": d["mean_score"],
                     "species": species if isinstance(species, (int, float)) else None,
                     "best_nodes": len(nodes), "best_conns": len(conns), "population": d.get("population")})
        if nodes:
            last = (d["generation"], nodes, conns)
    champion = {"generation": None, "nodes": [], "conns": []}
    if last:
        g, nodes, conns = last
        srcs, dsts = {int(c[0]) for c in conns}, {int(c[1]) for c in conns}
        def kind(k):
            return "input" if k not in dsts else "output" if k not in srcs else "hidden"
        outs = sorted(int(n[0]) for n in nodes if kind(int(n[0])) == "output")
        labels = {k: l for k, l in zip(outs, ["Δ east", "Δ north", "Δ up"])}
        champion = {"generation": g,
                    "nodes": [{"id": int(n[0]), "type": kind(int(n[0])), "label": labels.get(int(n[0]), "")} for n in nodes],
                    "conns": [{"in": int(c[0]), "out": int(c[1]), "weight": float(c[-1])} for c in conns if math.isfinite(c[-1])]}
    doc = {"placeholder": False,
           "meta": {"provenance": "modelled", "source": f"{run}/generation_*.json (swarm/run_swarm.py, TensorNEAT)", "frame": "n/a",
                    "units": "fitness as logged (best = mission score, mean = shaped selection fitness; see swarm/README.md)",
                    "note": "Exported by website/tools/export_neat.py"},
           "run": run.name, "generations": gens, "champion": champion}
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"wrote {OUT}: {len(gens)} generations, champion {len(champion['nodes'])} nodes / {len(champion['conns'])} conns")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/overnight1")

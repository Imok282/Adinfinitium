"""Key-free Dream-RSI-style loop for the rover's re-learning (EXPERIMENTAL). After "DREAM-RSI: Recursive Self-Improvement Through Evolving Worlds"
(arXiv 2609.14858; code github.com/google-research/dream-rsi, Apache-2.0). This is a small re-implementation of the idea, not their code.

Their loop: a round of tries forms a tree (branches x refinements) with a score per node; the tree is saved as a replay world; "dreaming" replays
rewritten search policies on all saved worlds, scores them by how much of each world's best result they reach for how little budget, and keeps the best
(the current one can stay); the kept policy runs the next round.
Here: a node is a candidate rover driver (recipe = branch, refinement = attempt) scored by real episodes; the policy is a Strategy.
NOT the original: strategy variants come from a fixed grid, not from an LLM rewriting code (that needs a paid API key); tries are sequential, so their
parallelism penalty has nothing to measure and is dropped; a tie keeps the current strategy; the budget preference is applied as a hard cap on tries,
where the original hands it to the policy as a knob. Like the original, a saved world holds only the nodes that were really tried, so replay can never
credit a strategy for exploring wider than the round that recorded it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

RHOS = (0.0, 0.25, 0.5, 0.75, 1.0)          # budget preference sweep (here: a hard cap on tries, from 2 up to every recorded node)
MIN_BUDGET = 2
NEVER = 99


class SearchEnv:
    """What a strategy sees: which nodes it may try next, and the score of each node it has tried. Scores are gains over the current driver (0 = no gain)."""

    def __init__(self, n_branches: int, max_attempts: int):
        self.n_branches, self.max_attempts = n_branches, max_attempts
        self.revealed: dict[tuple[int, int], float] = {}

    @property
    def budget_spent(self) -> int:
        return len(self.revealed)

    @property
    def best_so_far(self) -> float:
        return max([0.0, *self.revealed.values()])

    def attempts(self, branch: int) -> int:
        return sum(1 for b, _ in self.revealed if b == branch)

    def opened(self) -> int:
        return len({b for b, _ in self.revealed})

    def _available(self, branch: int, attempt: int) -> bool:
        raise NotImplementedError

    def _score(self, branch: int, attempt: int) -> float:
        raise NotImplementedError

    def legal_actions(self) -> list[tuple[str, int]]:
        """("new", b): open the next recipe. ("more", b): refine an opened recipe once more."""
        acts = []
        nxt = self.opened()
        if nxt < self.n_branches and self._available(nxt, 0):
            acts.append(("new", nxt))
        for b in range(nxt):
            k = self.attempts(b)
            if k < self.max_attempts and self._available(b, k):
                acts.append(("more", b))
        return acts

    def probe(self, action: tuple[str, int]) -> float:
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {action}")
        b = action[1]
        k = self.attempts(b)
        s = float(self._score(b, k))
        self.revealed[(b, k)] = s
        return s


class TraceEnv(SearchEnv):
    """A replay world: only nodes that were really scored can be revealed (a lookup, nothing is predicted)."""

    def __init__(self, trace: dict):
        super().__init__(len(trace["branches"]), trace["max_attempts"])
        self.nodes = {(n["branch"], n["attempt"]): n["score"] for n in trace["nodes"]}

    def _available(self, branch: int, attempt: int) -> bool:
        return (branch, attempt) in self.nodes

    def _score(self, branch: int, attempt: int) -> float:
        return self.nodes[(branch, attempt)]


class LiveEnv(SearchEnv):
    """The real round: revealing a node trains and tests a candidate driver (`run_node`). `can_continue` closes the round when time runs out."""

    def __init__(self, n_branches: int, max_attempts: int, run_node, can_continue=lambda: True):
        super().__init__(n_branches, max_attempts)
        self.run_node, self.can_continue = run_node, can_continue

    def _available(self, branch: int, attempt: int) -> bool:
        return self.can_continue()

    def _score(self, branch: int, attempt: int) -> float:
        return self.run_node(branch, attempt)


@dataclass(frozen=True)
class Strategy:
    """How one re-learning round spends its tries. The default opens every recipe, refines them in turn and never stops early."""
    width: int = 3              # recipes opened before any is refined
    greedy: bool = False        # refine the best recipe so far (True) or take turns (False)
    patience: int = NEVER       # give up on a recipe after this many refinements in a row that did not beat its own best
    stop_after: int = NEVER     # end the round after this many tries in a row that did not raise the best gain

    def solve(self, env: SearchEnv, budget: int) -> None:
        best: dict[int, float] = {}
        stale: dict[int, int] = {}
        streak = 0
        while env.budget_spent < budget:
            acts = env.legal_actions()
            new = [a for a in acts if a[0] == "new"]
            more = [a for a in acts if a[0] == "more" and stale.get(a[1], 0) < self.patience]
            if new and len(best) < self.width:
                act = new[0]
            elif more:
                act = max(more, key=lambda a: best[a[1]]) if self.greedy else min(more, key=lambda a: (env.attempts(a[1]), a[1]))
            elif new:
                act = new[0]
            else:
                break
            before = env.best_so_far
            try:
                s, b = env.probe(act), act[1]
            except ValueError:               # the live round closed (time ran out) between listing the actions and taking one
                break
            if b not in best or s > best[b]:
                best[b], stale[b] = s, 0
            else:
                stale[b] += 1
            streak = 0 if env.best_so_far > before else streak + 1
            if streak >= self.stop_after:
                break


DEFAULT = Strategy()


def grid() -> list[Strategy]:
    """The candidate strategies a dream may pick from (54, the default among them). Stands in for the paper's LLM-written policy versions."""
    return [Strategy(w, g, p, s) for w in (1, 2, 3) for g in (False, True) for p in (1, 2, NEVER) for s in (2, 3, NEVER)]


def trace_auc(points: list[tuple[int, float]], ref_budget: int) -> float:
    """Area under the budget-vs-attainment frontier, in [0, 1]: reaching the world's best with few tries is near 1, needing every try is near 0.
    points = (tries used, share of the world's best gain reached)."""
    if ref_budget <= 0 or not points:
        return 0.0
    best_at: dict[int, float] = {}
    for b, a in points:
        best_at[b] = max(a, best_at.get(b, 0.0))
    area, prev = 0.0, 0.0
    for b in sorted(best_at):
        if best_at[b] > prev:
            area += (best_at[b] - prev) * max(0, ref_budget - b)
            prev = best_at[b]
    return area / ref_budget


def score_strategy(strategy: Strategy, traces: list[dict]) -> dict:
    """Replay `strategy` on every saved world at each budget preference. Worlds where no node beat the current driver, or with under 2 nodes, carry no
    signal and are left out (as in the paper's code)."""
    per = []
    for t in traces:
        scores = [n["score"] for n in t["nodes"]]
        if len(scores) < MIN_BUDGET or max(scores) <= 0:
            continue
        run_best, n_eval = max(scores), len(scores)
        pts = []
        for rho in RHOS:
            env = TraceEnv(t)
            strategy.solve(env, max(1, round(MIN_BUDGET + rho * (n_eval - MIN_BUDGET))))
            pts.append((env.budget_spent, max(0.0, min(env.best_so_far / run_best, 1.0))))
        per.append({"world": t["name"], "auc": trace_auc(pts, n_eval)})
    return {"reward": sum(p["auc"] for p in per) / len(per) if per else None, "worlds_scored": len(per), "per_world": per}


def dream(incumbent: Strategy, traces: list[dict]) -> dict:
    """One dreaming step: replay every grid strategy and the current one on all saved worlds; keep the best. The current strategy stays unless
    another is strictly better."""
    inc = score_strategy(incumbent, traces)
    if inc["reward"] is None:
        return {"selected": asdict(incumbent), "kept_incumbent": True, "incumbent_reward": None, "best_reward": None, "worlds_scored": 0,
                "worlds_saved": len(traces), "note": "no saved world has a node that beat the current driver: nothing to learn from, strategy kept"}
    ranked = sorted(((score_strategy(s, traces)["reward"], i, s) for i, s in enumerate(grid())), key=lambda x: (-x[0], x[1]))
    best_r, _, best = ranked[0]
    keep = best_r <= inc["reward"] + 1e-9
    return {"selected": asdict(incumbent if keep else best), "kept_incumbent": keep, "incumbent_reward": inc["reward"], "best_reward": best_r,
            "worlds_scored": inc["worlds_scored"], "worlds_saved": len(traces), "variants_replayed": len(ranked),
            "top": [{"reward": r, **asdict(s)} for r, _, s in ranked[:5]]}


def save_trace(pool_dir: Path, trace: dict) -> Path:
    pool_dir.mkdir(parents=True, exist_ok=True)
    p = pool_dir / f"{len(list(pool_dir.glob('*.json'))):02d}_{trace['name'].replace(':', '_').replace('@', '_')}.json"
    p.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return p

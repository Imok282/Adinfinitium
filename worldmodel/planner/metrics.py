"""Planner exam metrics: calibration (Brier, expected calibration error) and decision quality (rank correlation, top-1 regret) per world."""

from __future__ import annotations

import numpy as np


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            tot += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return float(tot)


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def decision(score: np.ndarray, y: np.ndarray) -> dict:
    """Pick the top-scored candidate: its simulated success and the regret vs the best candidate of the world."""
    i = int(np.argmax(score))
    return {"chosen_p": float(y[i]), "best_p": float(y.max()), "regret": float(y.max() - y[i]), "rank_corr": spearman(score, y)}


def summarize(per_world: list[dict]) -> dict:
    rc = [d["rank_corr"] for d in per_world if d["rank_corr"] is not None]
    return {"worlds": len(per_world), "mean_chosen_p": float(np.mean([d["chosen_p"] for d in per_world])) if per_world else None,
            "mean_best_p": float(np.mean([d["best_p"] for d in per_world])) if per_world else None,
            "mean_regret": float(np.mean([d["regret"] for d in per_world])) if per_world else None,
            "mean_rank_corr": float(np.mean(rc)) if rc else None}

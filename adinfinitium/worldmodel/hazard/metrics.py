"""Precision / recall for 'unsafe' (positive class = dangerous pixel). Safety asymmetry: a missed
hazard (false negative) is worse than a false alarm, so recall is reported next to precision."""

from __future__ import annotations

import numpy as np


def confusion(pred_unsafe: np.ndarray, label_unsafe: np.ndarray) -> dict:
    p, y = pred_unsafe.astype(bool), label_unsafe.astype(bool)
    tp, fp = int((p & y).sum()), int((p & ~y).sum())
    fn, tn = int((~p & y).sum()), int((~p & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0, "prevalence": float(y.mean()),
            "missed_hazard_rate": fn / (tp + fn) if tp + fn else 0.0}

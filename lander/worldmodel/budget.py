"""Wall-clock time budget for training loops: measures its own speed, then sizes the run to FINISH inside the budget (on any machine).

    bud = TimeBudget(max_minutes, total_units)        # units = steps / epochs / updates; max_minutes <= 0 means no budget
    bud.begin(done)                                    # right before the loop
    if bud.update(done): ...                           # True once, after ~45 s: bud.total was lowered to fit; rebuild the LR schedule from it
    if bud.expired(): ...                              # hard stop (checkpoint and finish)

The rate includes evaluation and checkpoint time, so the estimate is honest. A 10% reserve is kept for the final evaluation and report.
"""

from __future__ import annotations

import time


class TimeBudget:
    def __init__(self, max_minutes: float, total: int, warmup_s: float = 45.0, reserve: float = 0.9):
        self.max_s = max_minutes * 60.0 if max_minutes and max_minutes > 0 else None
        self.total, self.requested = int(total), int(total)
        self.warmup_s, self.reserve = warmup_s, reserve
        self.t0 = 0.0
        self.done0 = 0
        self.adjusted = False

    def begin(self, done: int = 0) -> None:
        self.t0, self.done0 = time.time(), int(done)

    def elapsed(self) -> float:
        return time.time() - self.t0

    def update(self, done: int) -> bool:
        """After the warm-up, shrink `total` so the run fits the budget. Returns True exactly once, when `total` changed."""
        if self.max_s is None or self.adjusted:
            return False
        el = self.elapsed()
        if el < self.warmup_s or done <= self.done0:
            return False
        rate = (done - self.done0) / el
        fit = int(done + max(self.max_s * self.reserve - el, 0.0) * rate)
        self.adjusted = True
        new_total = max(done + 1, min(self.total, fit))
        changed = new_total != self.total
        self.total = new_total
        return changed

    def expired(self) -> bool:
        return self.max_s is not None and self.elapsed() > self.max_s

    def summary(self) -> str:
        if self.max_s is None:
            return "no time budget"
        return f"time budget {self.max_s / 60:.0f} min: {self.total} of {self.requested} requested units fit"

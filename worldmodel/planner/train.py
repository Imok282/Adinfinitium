"""Train the mission planner on simulated outcomes (labels from worldmodel.planner.labels). Splits by whole site; TEST Site23 + Site42 never used here.

    python -m worldmodel.planner.labels_run --max-minutes 120      # first: simulate outcomes (see labels_run.py)
    python -m worldmodel.planner.train --epochs 300

Trains the set-transformer planner AND the logistic baseline on the same data; keeps the best by validation Brier score. Checkpoints in <ADI_DATA>/models/.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from contracts.paths import model_path, run_dir
from worldmodel.planner.data import load_groups, role
from worldmodel.planner.metrics import brier
from worldmodel.planner.model import build_model, save_checkpoint


def predict(model, g) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(g.x)[None]))[0].numpy()


def fit(kind: str, train, val, epochs: int, lr: float, seed: int, **kw) -> tuple[torch.nn.Module, dict]:
    torch.manual_seed(seed)
    model = build_model(kind, **kw)
    allx = np.concatenate([g.x for g in train])
    model.mu.copy_(torch.as_tensor(allx.mean(0)))
    model.sd.copy_(torch.as_tensor(allx.std(0) + 1e-6))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    rng = np.random.default_rng(seed)
    best, best_state, hist = 1e9, None, []
    for ep in range(epochs):
        model.train()
        for i in rng.permutation(len(train)):
            g = train[i]
            logit = model(torch.as_tensor(g.x)[None])[0]
            loss = (F.binary_cross_entropy_with_logits(logit, torch.as_tensor(g.y), reduction="none") * torch.as_tensor(g.n_eff)).sum() / float(g.n_eff.sum())
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        vb = float(np.mean([brier(predict(model, g), g.y) for g in val])) if val else float("nan")
        hist.append(vb)
        if val and vb < best:
            best, best_state = vb, {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_brier": best, "epochs": epochs, "val_brier_curve": hist[:: max(1, epochs // 50)]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="planner")
    a = ap.parse_args(argv)
    groups = load_groups()
    train = [g for g in groups if role(g) == "train"]
    val = [g for g in groups if role(g) == "val"]
    if not train:
        raise SystemExit("no planner labels for training sites (run python -m worldmodel.planner.labels_run first)")
    print(f"train worlds {len(train)} ({sum(len(g.y) for g in train)} candidates) | val worlds {len(val)} | test worlds never used here", flush=True)
    log = run_dir("planner") / "train_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    for kind, kw in (("set", {"d": a.d, "layers": a.layers, "heads": a.heads}), ("logistic", {})):
        model, info = fit(kind, train, val, a.epochs, a.lr, a.seed, **kw)
        info.update({"kind": kind, "trained_on": sorted({g.region for g in train}), "val_regions": sorted({g.region for g in val}),
                     "parameters": sum(p.numel() for p in model.parameters()), "label": "simulated p(mission) = p(land) x p(rover reached ice and returned)"})
        save_checkpoint(model_path(f"{a.name}_{kind}"), model, info)
        print(f"{kind:9s} params {info['parameters']:>9,}  best val Brier {info['best_val_brier']:.4f}", flush=True)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(info) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Score the U-Net against the slope-rule backup on whole real tiles (precision / recall of 'unsafe').

    python -I -m worldmodel.hazard.evaluate                 # validation (Site11) + TEST (Site23)
    python -I -m worldmodel.hazard.evaluate --figure

Both models see the same sparse (laser-pixels-only) input; the label is the official 5 m LOLA slope >=
limit. Only the TEST region numbers (Site23, never trained or tuned on) are headline numbers.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from contracts.mission import load_mission
from contracts.paths import model_path, run_dir, tile_path
from worldmodel.hazard.backup import backup_hazard, label_hazard
from worldmodel.hazard.lola import HOLDOUT_SITES, regions_for_site
from worldmodel.hazard.metrics import confusion

CAVEAT = ("The label is NASA's 5 m LOLA slope map at every pixel, but most pixels of that map (about 90%; see measured_fraction) are interpolated, not laser-measured: "
          "the all-pixel rows (backup_slope_rule, unet) mostly score agreement with NASA's interpolation. The *_measured_px rows keep only pixels that hold a real laser "
          "return and are the stricter evidence. Limits: those same pixels are the model's and the backup's INPUT (their heights are given), so the measured-pixel rows "
          "show accuracy where the input is best informed and do NOT measure how well hazards are recovered in the gaps; the slope label at a measured pixel is still "
          "computed by NASA from neighbouring pixels that may be interpolated.")


def evaluate_tile(tile, model, device, cfg) -> dict:
    from worldmodel.hazard.backup import slope_limit_deg
    from worldmodel.hazard.features import build_features
    from worldmodel.hazard.model import predict_probability

    thr = cfg.hazard_detector.decision_threshold
    label = label_hazard(tile, cfg) > 0.5
    feats, _ = build_features(tile, slope_limit_deg(cfg), cfg.hazard_detector.gap_fill_sigma_px)
    prob = predict_probability(model, feats, device)
    backup = backup_hazard(tile, cfg) > 0.5
    res = {"backup_slope_rule": confusion(backup, label), "unet": confusion(prob >= thr, label)}
    res.update(measured_px_metrics(tile.real_mask, label, backup, prob >= thr))
    return res, label, prob


def measured_px_metrics(real_mask, label, backup, unet) -> dict:
    """The same confusion metrics on laser-measured pixels only (tile.real_mask); a tile without a mask is skipped and says so."""
    if real_mask is None or not real_mask.any():
        return {"measured_px_note": "tile has no laser-pixel mask (real_mask): measured-pixel metrics skipped"}
    m = real_mask.astype(bool)
    return {"measured_fraction": float(m.mean()), "backup_slope_rule_measured_px": confusion(backup[m], label[m]), "unet_measured_px": confusion(unet[m], label[m])}


def main(argv: list[str] | None = None) -> int:
    from contracts.tiles import load_tile
    from worldmodel.hazard.model import load_checkpoint, pick_device

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="hazard_unet")
    ap.add_argument("--sites", nargs="+", default=["Site11"] + HOLDOUT_SITES)
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    cfg = load_mission()
    device = pick_device(a.device)
    model, info = load_checkpoint(model_path(a.name), device)
    report = {"checkpoint_name": a.name, "caveat": CAVEAT, "checkpoint_id": info["checkpoint_id"], "trained_on": info["trained_on"], "val_regions": info["val_regions"],
              "label": "official LOLA 5 m slope >= limit", "limit_deg": info["limit_deg"], "regions": {}}
    print(f"checkpoint {info['checkpoint_id']}  trained on {info['trained_on']}")
    print(f"{'region':14s} {'role':9s} {'model':18s} {'precision':>9s} {'recall':>7s} {'F1':>6s} {'missed':>7s}")
    jobs = sorted(((s, r) for s in a.sites for r in regions_for_site(s)), key=lambda j: j[0] not in HOLDOUT_SITES)     # test sites first
    for s in a.sites:
        if not regions_for_site(s):
            print(f"note: no tiles for {s} on this machine; skipped", flush=True)
    for s, region in jobs:
        t = load_tile(tile_path(region))
        role = "TEST" if s in HOLDOUT_SITES else "val"
        res, label, prob = evaluate_tile(t, model, device, cfg)
        report["regions"][t.region] = {"role": role, "unsafe_fraction": float(label.mean()), **res}
        for k, name in (("backup_slope_rule", "slope rule (backup)"), ("unet", "U-Net"),
                        ("backup_slope_rule_measured_px", "backup, laser px"), ("unet_measured_px", "U-Net, laser px")):
            if k not in res:
                continue
            m = res[k]
            print(f"{t.region:14s} {role:9s} {name:18s} {m['precision']:9.3f} {m['recall']:7.3f} {m['f1']:6.3f} "
                  f"{m['missed_hazard_rate']:7.3f}")
        print(f"{t.region:14s} {role:9s} " + (f"laser-measured pixels: {res['measured_fraction']:.1%} of the tile" if "measured_fraction" in res else res["measured_px_note"]))
        if a.figure:
            _figure(t, label, backup_hazard(t, cfg) > 0.5, prob >= cfg.hazard_detector.decision_threshold,
                    run_dir("hazard") / f"compare_{t.region}.png")
        _write_partial(run_dir("hazard") / "eval.json", report, len(jobs))
    out = run_dir("hazard") / "eval.json"
    _write_partial(out, report, len(jobs))
    print("caveat: " + CAVEAT)
    print(f"wrote {out}")
    return 0


def _write_partial(out, report: dict, planned: int) -> None:
    """eval.json is rewritten after every window, so a stage killed at its budget still leaves the test-site numbers."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps({**report, "windows_planned": planned, "complete": len(report["regions"]) >= planned}, indent=2), encoding="utf-8")
    tmp.replace(out)


def _figure(tile, label, backup, unet, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 4, figsize=(20, 5.4))
    ax[0].imshow(tile.height, cmap="gray")
    ax[0].imshow(np.ma.masked_where(~tile.real_mask, tile.real_mask), cmap="autumn", alpha=0.9)
    ax[0].set_title("height (red = real laser pixels)")
    for a_, arr, ttl in ((ax[1], label, "label: official slope >= limit"),
                         (ax[2], backup, "slope-rule backup (sparse input)"), (ax[3], unet, "U-Net (same input)")):
        a_.imshow(arr, cmap="RdYlGn_r", vmin=0, vmax=1)
        a_.set_title(ttl)
    for a_ in ax:
        a_.axis("off")
    fig.suptitle(f"{tile.region}  [{tile.meta.provenance.banner} LOLA data; hazard maps MODELLED]  "
                 "red = unsafe, green = safe", y=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())

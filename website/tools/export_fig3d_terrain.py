"""3D figure data for research/terrain.html: three hide-and-check patches of held-out Site 23 as surfaces.

Usage (repo root, read-only on the inputs; numpy 2.4.6 as in the frozen run):
    ~/venvs/adi/bin/python -I website/tools/export_fig3d_terrain.py

What it does
- Replays the frozen hide-and-check pixel choice on the Site 23 centre window (lola5m_site23, a held-out TEST window):
  numpy seed 11, 96 patch origins, half of each patch's laser pixels shown / half hidden, the Gaussian backup drawn from the
  same stream (worldmodel/terrain/evaluate.py::hide_and_check, as replayed by records/weak_flags/scripts/terrain/terrain_wf.py).
- Kriging = the presented estimator, setting chosen on Site 11: gp|mat32|32.0|0.01|1 (Matern 3/2, length 32 px, nugget 0.01,
  plane drift), code copied from terrain_wf.py::gp_predict. It sees only the shown pixels of the patch.
- Checks (asserted): the first 3 patches have the origins and shown/hidden counts printed in
  records/terrain_calibration/terrain_diffusion_samples_site23.png; the window totals are 21,767 hidden / 21,903 shown pixels
  (terrain.json pixel_identity); kriging and backup RMSE/MAE over the whole window match terrain.json heldout to 1e-6 m.
- Exports the first 3 patches (the same patches as the 2-D figure on the page) as uint16 heights (models3d encoding,
  per patch and surface) for truth, kriging (whole patch) and the old backup (mean of 6 versions), plus shown/hidden masks.

Diffusion network: its generated surfaces for these patches (terrain_diffusion_samples_site23.npz) and its checkpoint
(terrain_diffusion_mini.pt, 1f647a8660) are listed in the hand-off MANIFEST but are not on this machine, so no diffusion surface
is exported. Its RMSE on the same hidden pixels (0.51 / 0.74 / 0.85 m) is copied from that figure's titles.
"""
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ADI_DATA", str(ROOT / "data"))

import numpy as np                                                        # noqa: E402
from scipy.spatial.distance import cdist                                  # noqa: E402

from contracts.mission import load_mission                                # noqa: E402
from worldmodel.terrain.backup import backup_versions                     # noqa: E402
from worldmodel.terrain.data import PATCH, load_region, sample_patch_origins  # noqa: E402

REGION, PATCHES, VERSIONS, SEED, NFIG = "lola5m_site23", 96, 6, 11, 3
KRIG = ("mat32", 32.0, 0.01, 1)
WF = ROOT / "data/handoff_macmini_1115/macbook/records/weak_flags/terrain.json"
OUT = ROOT / "website/assets/fig3d/terrain"
# from the titles of records/terrain_calibration/terrain_diffusion_samples_site23.png (checkpoint 1f647a8660, torch seed 11)
FIG = [((128, 247), 244, 253, 0.51), ((123, 533), 243, 260, 0.74), ((765, 817), 245, 243, 0.85)]


def mat32(d):
    a = math.sqrt(3) * d
    return (1 + a) * np.exp(-a)


def gp_predict(X, y, Xq, length, nugget, drift):
    """terrain_wf.py::gp_predict (kernel mat32), mean only."""
    n = len(y)
    K = mat32(cdist(X, X) / length) + nugget * np.eye(n)
    Kq = mat32(cdist(Xq, X) / length)
    ctr = X.mean(0)
    if drift == 1:
        P = np.column_stack([np.ones(n), (X - ctr) / 32.0])
        Pq = np.column_stack([np.ones(len(Xq)), (Xq - ctr) / 32.0])
        if np.linalg.matrix_rank(P, tol=1e-6) < 3:
            drift = 0
    if drift == 0:
        P, Pq = np.ones((n, 1)), np.ones((len(Xq), 1))
    m = P.shape[1]
    A = np.zeros((n + m, n + m))
    A[:n, :n], A[:n, n:], A[n:, :n] = K, P, P.T
    B = np.column_stack([Kq, Pq]).T
    sol = np.linalg.solve(A, np.column_stack([np.concatenate([y, np.zeros(m)]), B]))
    return B.T @ sol[:, 0]


def krige(hp, C, Xq):
    sr, sc = np.nonzero(C)
    X = np.column_stack([sr, sc]).astype(np.float64)
    y = hp[C].astype(np.float64)
    mu = y.mean()
    _, L, g, dr = KRIG
    return gp_predict(X, y - mu, Xq, L, g, dr) + mu


def u16(a, lo, hi):
    return np.round((a - lo) / (hi - lo) * 65535).clip(0, 65535).astype("<u2")


def main():
    cfg = load_mission()
    tile = load_region(REGION)
    rng = np.random.default_rng(SEED)
    origins = sample_patch_origins(tile.real_mask, PATCHES, rng)
    h = tile.height.astype(np.float64)
    sig_px = cfg.hazard_detector.gap_fill_sigma_px
    grid = np.column_stack([a.ravel() for a in np.mgrid[0:PATCH, 0:PATCH]]).astype(np.float64)
    real_all, krig_all, bak_all, n_shown, keep = [], [], [], 0, []
    for r, c in origins:
        hp, mp = h[r:r + PATCH, c:c + PATCH], tile.real_mask[r:r + PATCH, c:c + PATCH]
        sig = tile.uncertainty[r:r + PATCH, c:c + PATCH].astype(np.float64)
        C = mp & (rng.random(mp.shape) < 0.5)
        Hm = mp & ~C
        if C.sum() < 8 or Hm.sum() < 8:
            continue
        bak = backup_versions(hp, C, sig, sig_px, VERSIONS, rng).mean(0)
        hr, hc = np.nonzero(Hm)
        kq = krige(hp, C, np.column_stack([hr, hc]).astype(np.float64))
        real_all.append(hp[Hm]); krig_all.append(kq); bak_all.append(bak[Hm]); n_shown += int(C.sum())
        if len(keep) < NFIG:
            keep.append(dict(origin=(int(r), int(c)), real=hp.copy(), C=C.copy(), Hm=Hm.copy(), bak=bak,
                             krig=krige(hp, C, grid).reshape(PATCH, PATCH)))
    real, kr, bk = (np.concatenate(v) for v in (real_all, krig_all, bak_all))
    rm = lambda p: (float(np.sqrt(np.mean((p - real) ** 2))), float(np.mean(np.abs(p - real))))
    wf = json.loads(WF.read_text())
    ident = wf["pixel_identity"]["site23"]["windows"][REGION]
    st = wf["heldout"]["sites"]["site23"]["methods"]
    k_rm, b_rm = rm(kr), rm(bk)
    print("window hidden", real.size, "shown", n_shown, "kriging rmse/mae", k_rm, "backup", b_rm)
    assert real.size == ident["n_hidden_pixels"] == 21767 and n_shown == ident["n_shown_pixels"] == 21903
    sk, sb = st["gp|mat32|32.0|0.01|1"]["windows"][REGION], st["backup_as_frozen"]["windows"][REGION]
    assert abs(k_rm[0] - sk["rmse_m"]) < 1e-6 and abs(k_rm[1] - sk["mae_m"]) < 1e-6, (k_rm, sk)
    assert abs(b_rm[0] - sb["rmse_m"]) < 1e-6 and abs(b_rm[1] - sb["mae_m"]) < 1e-6, (b_rm, sb)

    OUT.mkdir(parents=True, exist_ok=True)
    blobs, patches = [], []
    for i, (p, (o, ns, nh, net)) in enumerate(zip(keep, FIG)):
        assert p["origin"] == o and int(p["C"].sum()) == ns and int(p["Hm"].sum()) == nh, (i, p["origin"], int(p["C"].sum()), int(p["Hm"].sum()))
        Hm = p["Hm"]
        surf = {"truth": p["real"], "kriging": p["krig"], "backup": p["bak"]}
        lo = float(min(a.min() for a in surf.values())); hi = float(max(a.max() for a in surf.values()))
        errs = {}
        for k in ("kriging", "backup"):
            e = surf[k] - p["real"]
            errs[k] = {"rmse_hidden_m": round(float(np.sqrt(np.mean(e[Hm] ** 2))), 3), "mae_hidden_m": round(float(np.mean(np.abs(e[Hm]))), 3),
                       "rmse_all_not_shown_m": round(float(np.sqrt(np.mean(e[~p["C"]] ** 2))), 3)}
        errs["network"] = {"rmse_hidden_m": net, "source": "title of records/terrain_calibration/terrain_diffusion_samples_site23.png"}
        for k in ("truth", "kriging", "backup"):
            blobs.append(u16(surf[k], lo, hi).tobytes())
        mask = (p["C"].astype(np.uint8) | (Hm.astype(np.uint8) << 1))          # 1 = shown laser pixel, 2 = hidden laser pixel
        blobs.append(mask.tobytes())
        patches.append({"index": i, "origin_rc": list(o), "min_m": lo, "max_m": hi, "relief_m": round(float(p["real"].max() - p["real"].min()), 2),
                        "n_shown": ns, "n_hidden": nh, "errors": errs})
        print("patch", i, o, "kriging", errs["kriging"], "backup", errs["backup"]["rmse_hidden_m"], "network", net)
    (OUT / "patches.bin").write_bytes(b"".join(blobs))
    meta = {
        "created_by": "website/tools/export_fig3d_terrain.py",
        "region": REGION, "role": "TEST (held-out Site 23 centre window; never used for training or settings)",
        "n": PATCH, "m_per_px": 5.0, "size_m": PATCH * 5.0,
        "layout": "per patch: truth, kriging, backup as uint16 LE n*n (row 0 = north, h = min + v/65535*(max-min)), then uint8 n*n mask (1 shown, 2 hidden)",
        "surfaces": {
            "truth": {"provenance": "observed", "label": "Real height (LOLA 5 m)", "note": "away from laser pixels this is NASA's interpolation"},
            "kriging": {"provenance": "modelled", "label": "Kriging (Matern 3/2, plane mean)", "setting": "gp|mat32|32.0|0.01|1, chosen on Site 11"},
            "backup": {"provenance": "modelled", "label": "Old backup (Gaussian gap-fill)", "note": "mean of 6 noisy versions, as frozen"},
        },
        "diffusion_missing": "terrain_diffusion_samples_site23.npz and terrain_diffusion_mini.pt are in the hand-off MANIFEST but not on this machine",
        "window": {"n_hidden": int(real.size), "n_shown": n_shown, "kriging_rmse_m": round(k_rm[0], 3), "kriging_mae_m": round(k_rm[1], 3),
                   "backup_rmse_m": round(b_rm[0], 3), "network_rmse_m": round(st["network"]["windows"][REGION]["rmse_m"], 3)},
        "site23_pooled": {"kriging_rmse_m": round(st["gp|mat32|32.0|0.01|1"]["pooled"]["rmse_m"], 2), "network_rmse_m": round(st["network"]["pooled"]["rmse_m"], 2)},
        "checks": "origins and shown/hidden counts equal the 2-D figure; window totals and kriging/backup RMSE+MAE equal terrain.json to 1e-6 m",
        "source": "data/tiles/lola5m_site23.npz (NASA LRO LOLA PGDA 5 m); records/weak_flags/terrain.json; records/weak_flags/scripts/terrain/terrain_wf.py",
        "patches": patches,
    }
    (OUT / "patches.json").write_text(json.dumps(meta, indent=1))
    print("wrote", OUT, sum(len(b) for b in blobs), "bytes bin")


if __name__ == "__main__":
    main()

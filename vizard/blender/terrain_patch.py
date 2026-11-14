"""Cut a flat, safe patch of real LOLA terrain (default Site23) for the Blender hero scene: <out>.npz with height (m, centred) and mpp."""
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from contracts.paths import tile_path
from contracts.tiles import load_tile

region, out, half = (sys.argv[1] if len(sys.argv) > 1 else "lola5m_site23"), sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 80
t = load_tile(tile_path(region))
w = 21
mean_slope = sliding_window_view(np.pad(t.slope, w // 2, mode="edge"), (w, w)).mean(axis=(-1, -2))
m = half + 10
mean_slope[:m], mean_slope[-m:], mean_slope[:, :m], mean_slope[:, -m:] = 99, 99, 99, 99
r, c = np.unravel_index(np.argmin(mean_slope), mean_slope.shape)
h = t.height[r - half:r + half + 1, c - half:c + half + 1].astype(np.float32)
h -= h[half, half]
np.savez(out, height=h, mpp=t.meters_per_pixel, region=region, centre_rc=np.array([r, c]), slope_here=float(t.slope[r, c]))
print(f"{region}: patch {h.shape} at ({r},{c}), slope {t.slope[r, c]:.1f} deg, relief {h.max() - h.min():.1f} m")

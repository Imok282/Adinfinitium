#!/bin/sh
# Render the static fallback images (reduced motion / no WebGL) of the four 3D views in the Models section.
# Needs the site served locally (e.g. cd website && python3 -m http.server 8765) and assets from export_models3d.py.
#   BASE=http://localhost:8765/ sh website/tools/render_models3d_stills.sh
# ?mvshot=<view> makes js/models3d.js render that one view full-window, without motion.
set -e
CH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
BASE=${BASE:-http://localhost:8765/}
HERE=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d)
for v in elev hazard light ghosts; do
  "$CH" --headless=new --hide-scrollbars --user-data-dir="$TMP/profile" --disk-cache-size=1 --enable-unsafe-swiftshader \
    --window-size=1280,720 --virtual-time-budget=8000 --screenshot="$TMP/$v.png" "${BASE}?mvshot=$v" 2>/dev/null
  # 960 x 540 JPEG, kept under 150 KB
  sips -s format jpeg -s formatOptions 80 -z 540 960 "$TMP/$v.png" --out "$HERE/assets/models3d/still_$v.jpg" >/dev/null
done
rm -rf "$TMP"
ls -la "$HERE"/assets/models3d/still_*.jpg

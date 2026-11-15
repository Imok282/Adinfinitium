#!/bin/sh
# Headless Chrome screenshots of the running site. Start a server first:  cd website && python3 -m http.server 8780
CH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
BASE=${BASE:-http://localhost:8780/}
OUT=$(cd "$(dirname "$0")/../.." && pwd)/website_dev/screenshots
PROFILE=${PROFILE:-${TMPDIR:-/tmp}/adi-shot-chrome}
mkdir -p "$OUT"
shot() { # name query size
  perl -e "alarm 120; exec @ARGV" "$CH" --headless=new --user-data-dir="$PROFILE" --hide-scrollbars --enable-webgl --ignore-gpu-blocklist \
    --virtual-time-budget=${VT:-16000} --window-size="$3" --screenshot="$OUT/$1.png" "$BASE?$2" 2>/dev/null
}
shot 01_hero          "scroll=0"                  1440,900
shot 02_separation    "scroll=0.04"               1440,900
shot 03_mission       "scroll=problem&off=1.2"    1440,900
shot 03b_mission_grid "scroll=problem&off=2.6"    1440,900
shot 04_pilot_in      "scroll=ai&off=1.0"         1440,900
shot 04b_pilot_net    "scroll=ai&off=2.9"         1440,900
shot 04c_pilot_out    "scroll=ai&off=4.4"         1440,900
shot 05_built         "scroll=built&off=1.6"      1440,900
shot 05b_built_done   "scroll=built&off=3.6"      1440,900
shot 06_models_cards  "scroll=agents&off=3.6"     1440,900
shot 06b_model_modal  "scroll=agents&off=3.6&model=pilot" 1440,900
shot 07_models3d      "scroll=models&off=0.4" 1440,900
shot 07b_ghosts3d     "scroll=view-ghosts&off=1.8&tab=ghosts" 1440,900
shot 08_results       "scroll=results&off=1.5"    1440,900
shot 08b_results_deck "scroll=results&off=4"      1440,900
shot 09_touchdown_ghosts "scroll=landing&off=2.5" 1440,900
shot 09b_ghosts_end   "scroll=landing&off=4.5"    1440,900
shot 10_descent       "scroll=descent&off=0.3"    1440,900
shot 10b_touchdown    "scroll=descent&off=1.0"    1440,900
shot 11_rover         "scroll=rover&off=1.0"      1440,900
shot 11b_rover_dash   "scroll=rover&off=2.4"      1440,900
shot 12_cta_footer    "static&scroll=cta&off=0"                  1440,900

# phone width: headless Chrome will not go below 500 px, so the site runs in a 390 x 844 iframe and the shot is cropped
phone() {
  perl -e "alarm 120; exec @ARGV" "$CH" --headless=new --user-data-dir="$PROFILE" --hide-scrollbars \
    --virtual-time-budget=${VT:-16000} --window-size=500,844 --screenshot="$OUT/$1.png" "${BASE}tools/phone.html?$2" 2>/dev/null
  ffmpeg -v error -y -i "$OUT/$1.png" -vf crop=390:844:0:0 "$OUT/$1.tmp.png" && mv "$OUT/$1.tmp.png" "$OUT/$1.png"
}
phone 13_phone_hero    "scroll=0"
phone 13b_phone_results "scroll=results&off=1"
phone 13c_phone_models "scroll=agents&off=1"
phone 13d_phone_rover "scroll=rover&off=1"

# no-WebGL / reduced-motion path (?static): static posters, no scroll-jacking
perl -e "alarm 120; exec @ARGV" "$CH" --headless=new --user-data-dir="$PROFILE" --hide-scrollbars --virtual-time-budget=6000 --window-size=1440,900 --screenshot="$OUT/16_static_fallback.png" "$BASE?static" 2>/dev/null

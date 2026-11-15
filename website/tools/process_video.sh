#!/usr/bin/env bash
# Prepare an AI-generated cinematic clip for one of the site's media slots.
#
#   website/tools/process_video.sh <input video> <slot>        slot = hero | descent | landing
#
# Writes website/assets/video/<slot>.mp4 (H.264, yuv420p, +faststart, no audio, <= 8 MB),
# website/assets/video/<slot>.webm (VP9, no audio) and website/assets/video/<slot>.jpg (poster from ~1 s).
# js/media.js shows a slot automatically once its files exist; delete them to hide it again.
#
# The generator's visible watermark sits in the bottom-right corner. We crop CROP_PCT % (default 7) of the
# width and height, split evenly between left/right and top/bottom so the clip stays centred and keeps its aspect.
# Env overrides: CROP_PCT (0 disables cropping), MAX_MB (default 8), FFMPEG (path to ffmpeg).
set -euo pipefail

usage() { echo "usage: $0 <input video> <hero|descent|landing>" >&2; exit 2; }
[ $# -eq 2 ] || usage
IN=$1; SLOT=$2
case "$SLOT" in hero|descent|landing) ;; *) usage ;; esac
[ -f "$IN" ] || { echo "no such file: $IN" >&2; exit 1; }

FFMPEG=${FFMPEG:-$(command -v ffmpeg || echo /opt/homebrew/bin/ffmpeg)}
FFPROBE=${FFPROBE:-$(dirname "$FFMPEG")/ffprobe}
[ -x "$FFMPEG" ] || { echo "ffmpeg not found (install it or set FFMPEG=...)" >&2; exit 1; }

CROP_PCT=${CROP_PCT:-7}
MAX_MB=${MAX_MB:-8}
if [ "$SLOT" = hero ]; then MAXW=1280; else MAXW=960; fi

HERE=$(cd "$(dirname "$0")" && pwd)
OUT="$HERE/../assets/video"
mkdir -p "$OUT"
MP4="$OUT/$SLOT.mp4"; WEBM="$OUT/$SLOT.webm"; JPG="$OUT/$SLOT.jpg"

# crop (symmetric, keeps aspect) -> scale to <= MAXW wide, even dimensions
CROP="crop=trunc(iw*(1-${CROP_PCT}/100)/2)*2:trunc(ih*(1-${CROP_PCT}/100)/2)*2:(iw-trunc(iw*(1-${CROP_PCT}/100)/2)*2)/2:(ih-trunc(ih*(1-${CROP_PCT}/100)/2)*2)/2"
SCALE="scale='min(${MAXW},iw)':-2:flags=lanczos"
VF="${CROP},${SCALE},format=yuv420p"

MAX_BYTES=$((MAX_MB * 1024 * 1024))
size() { stat -f %z "$1" 2>/dev/null || stat -c %s "$1"; }

# H.264: start at CRF 23 and raise it until the file fits
CRF=23
while :; do
  "$FFMPEG" -hide_banner -loglevel error -y -i "$IN" -vf "$VF" -an \
    -c:v libx264 -preset slow -crf "$CRF" -profile:v high -pix_fmt yuv420p -movflags +faststart "$MP4"
  S=$(size "$MP4")
  echo "mp4  crf $CRF -> $((S / 1024)) KB"
  [ "$S" -le "$MAX_BYTES" ] && break
  CRF=$((CRF + 3))
  [ "$CRF" -gt 40 ] && { echo "still over ${MAX_MB} MB at CRF 40; trim the clip first" >&2; exit 1; }
done

# VP9 WebM (optional second source; browsers pick the first they can play)
WCRF=34
while :; do
  "$FFMPEG" -hide_banner -loglevel error -y -i "$IN" -vf "$VF" -an \
    -c:v libvpx-vp9 -b:v 0 -crf "$WCRF" -row-mt 1 -deadline good -cpu-used 2 "$WEBM"
  S=$(size "$WEBM")
  echo "webm crf $WCRF -> $((S / 1024)) KB"
  [ "$S" -le "$MAX_BYTES" ] && break
  WCRF=$((WCRF + 4))
  [ "$WCRF" -gt 56 ] && { echo "webm over ${MAX_MB} MB; removing it (the mp4 is enough)" >&2; rm -f "$WEBM"; break; }
done

# poster: frame at ~1 s (or the first frame of a shorter clip), same crop/scale
DUR=$("$FFPROBE" -v error -show_entries format=duration -of default=nw=1:nk=1 "$IN" 2>/dev/null || echo 0)
SS=$(awk -v d="${DUR:-0}" 'BEGIN { print (d > 1.2) ? 1 : 0 }')
"$FFMPEG" -hide_banner -loglevel error -y -ss "$SS" -i "$IN" -frames:v 1 -vf "${CROP},${SCALE}" -q:v 3 "$JPG"

echo "done: $(ls -1 "$OUT"/"$SLOT".* | sed "s#^$HERE/../##" | tr '\n' ' ')"
echo "reload the page: the '$SLOT' slot appears automatically."

# list what exists in assets/video/manifest.json (js/media.js reads it instead of probing for files)
( cd "$OUT" && printf '{ "files": [%s] }\n' "$(ls -1 *.mp4 *.webm *.jpg 2>/dev/null | sed 's/.*/"&"/' | paste -sd, -)" > manifest.json )
echo "updated $(basename "$OUT")/manifest.json"

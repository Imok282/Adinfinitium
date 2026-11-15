#!/bin/bash
# Builds a data folder for rover v2 that READS the main run's data through symlinks and WRITES only here.
#   bash ops/rover2/make_overlay.sh <main data folder> <new data folder>
# tiles/ and kernels/ are linked as folders (only read). models/ and rover_cache/ are real folders holding one symlink per existing main file, so new
# checkpoints (rover_driver_v2*.pt) and new caches (Site11 worlds) are created here and nothing is ever created or replaced in the main folder.
set -u
SRC="$1"; DST="$2"
[ -d "$SRC/tiles" ] || { echo "REFUSED: no tiles folder in $SRC"; exit 1; }
SRC="$(cd "$SRC" && pwd -P)"
mkdir -p "$DST/models" "$DST/rover_cache"
DSTP="$(cd "$DST" && pwd -P)"
case "$DSTP/" in "$SRC/"*) echo "REFUSED: $DST is inside the main data folder"; exit 3;; esac
for d in tiles kernels; do
  [ -e "$DST/$d" ] || { [ -d "$SRC/$d" ] && ln -s "$SRC/$d" "$DST/$d"; }
done
for d in models rover_cache; do
  for f in "$SRC/$d"/*; do
    [ -f "$f" ] || continue
    b="$(basename "$f")"
    case "$b" in rover_driver_v2*) continue;; esac
    [ -e "$DST/$d/$b" ] || [ -L "$DST/$d/$b" ] || ln -s "$f" "$DST/$d/$b"
  done
done
# A dangling link would let a cache write pass THROUGH it into the main folder: refuse.
for l in "$DST"/models/* "$DST"/rover_cache/* "$DST/tiles" "$DST/kernels"; do
  if [ -L "$l" ] && [ ! -e "$l" ]; then echo "REFUSED: dangling link $l (the main file is gone); remove the link and run again"; exit 4; fi
done
echo "data overlay ready: $DSTP (reads $SRC)"

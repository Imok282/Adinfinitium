#!/usr/bin/env bash
# Re-render every diagram. Needs graphviz (dot) and mermaid-cli (npm i -g @mermaid-js/mermaid-cli).
set -euo pipefail
cd "$(dirname "$0")"
for f in *.dot; do dot -Tsvg "$f" -o "${f%.dot}.svg"; dot -Tpng -Gdpi=160 "$f" -o "${f%.dot}.png"; done
for f in *.mmd; do
  npx --no-install mmdc -i "$f" -o "${f%.mmd}.svg" -b white ${PUPPETEER_CFG:+-p "$PUPPETEER_CFG"}
  npx --no-install mmdc -i "$f" -o "${f%.mmd}.png" -b white -s 2 ${PUPPETEER_CFG:+-p "$PUPPETEER_CFG"}
done
echo "rendered: $(ls *.svg | wc -l) svg"

// Frame-time probe for the #rover beat on landing.html (agent F).
//   cd website/tools/qa && node perf_rover.mjs [width] [height] [--trace]
// Wheel-scrolls 120 px every 50 ms through #rover (from 1 viewport above it to its end) and prints p50 / p95 / max
// frame time and the dropped-frame count (frames > 1.5 x 16.7 ms), plus the same for a no-scroll idle baseline.
// Caveat: headless Chrome on an 8 GB M1 runs WebGL through SwiftShader/ANGLE-Metal with no vsync, so absolute
// numbers are pessimistic and noisy; compare before/after on the same machine, not against 60 fps.
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';

const W = +(process.argv[2] || 1440), H = +(process.argv[3] || 900), TRACE = process.argv.includes('--trace');
const BASE = process.env.BASE || 'http://localhost:8780/landing.html';
const sleep = ms => new Promise(r => setTimeout(r, ms));
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: 'new', args: ['--enable-webgl', '--ignore-gpu-blocklist', `--window-size=${W},${H}`, '--hide-scrollbars'],
  defaultViewport: { width: W, height: H, deviceScaleFactor: 1, isMobile: W < 500, hasTouch: false },
});
const page = await browser.newPage();
await page.goto(BASE, { waitUntil: 'networkidle2', timeout: 60000 });
await sleep(2500);
await page.mouse.move(W / 2, H / 2);
await page.evaluate(() => {
  window.__ft = []; let last = performance.now();
  const f = now => { window.__ft.push(now - last); last = now; requestAnimationFrame(f); };
  requestAnimationFrame(f);
});
const stats = a => {
  const s = [...a].sort((x, y) => x - y), q = k => s[Math.min(s.length - 1, Math.floor(k * s.length))] || 0;
  return { frames: s.length, p50: +q(0.5).toFixed(1), p95: +q(0.95).toFixed(1), max: +(s[s.length - 1] || 0).toFixed(1), dropped: s.filter(x => x > 25).length };
};
// jump to just above the rover beat
const top = await page.evaluate(() => { const r = document.getElementById('rover'); return r.getBoundingClientRect().top + scrollY - innerHeight; });
await page.evaluate(y => scrollTo(0, y), top);
await sleep(1500);
await page.evaluate(() => { window.__ft = []; });
await sleep(2000);
const idle = stats(await page.evaluate(() => window.__ft));
await page.evaluate(() => { window.__ft = []; });
if (TRACE) await page.tracing.start({ path: new URL('./perf_rover_trace.json', import.meta.url).pathname, screenshots: false });
const end = await page.evaluate(() => { const r = document.getElementById('rover'); return r.getBoundingClientRect().bottom + scrollY; });
for (let i = 0; i < 400; i++) {
  await page.mouse.wheel({ deltaY: 120 });
  await sleep(50);
  if (await page.evaluate(e => scrollY + innerHeight >= e, end)) break;
}
if (TRACE) await page.tracing.stop();
const scroll = stats(await page.evaluate(() => window.__ft));
console.log(JSON.stringify({ viewport: `${W}x${H}`, idle, scroll }, null, 1));
await browser.close();

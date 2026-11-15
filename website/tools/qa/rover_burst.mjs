// Dense real-scroll frames through the #rover beat (agent F): wheel 60 px per step, a full frame every 4th step and a
// 240x240 close-up crop around the 3D rover every step (position from window.__qa.rover, written by js/scene.js).
//   cd website/tools/qa && node rover_burst.mjs [width] [height] [tag]   (BASE env, default landing.html)
// Output: out_rover_<tag>/ (f_*.jpg, c_*.jpg, track.jsonl, crops.jpg contact sheet, frames.jpg)
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import { execFileSync } from 'node:child_process';

const W = +(process.argv[2] || 1440), H = +(process.argv[3] || 900), TAG = process.argv[4] || 'run';
const BASE = process.env.BASE || 'http://localhost:8780/landing.html';
const OUT = new URL(`./out_rover_${TAG}/`, import.meta.url).pathname;
fs.rmSync(OUT, { recursive: true, force: true }); fs.mkdirSync(OUT, { recursive: true });
const sleep = ms => new Promise(r => setTimeout(r, ms));
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: 'new', args: ['--enable-webgl', '--ignore-gpu-blocklist', `--window-size=${W},${H}`, '--hide-scrollbars'],
  defaultViewport: { width: W, height: H, deviceScaleFactor: 1, isMobile: W < 500, hasTouch: false },
});
const page = await browser.newPage();
await page.goto(BASE, { waitUntil: 'networkidle2', timeout: 60000 });
await sleep(6000);   // loader + scene init
await page.mouse.move(W / 2, H / 2);
// real wheel scrolling all the way (jumping with scrollTo lets late layout / scripts move the page under us)
for (let i = 0; i < 2000; i++) {
  const d = await page.evaluate(() => document.getElementById('rover').getBoundingClientRect().top - innerHeight * 0.2);
  if (d <= 0) break;
  await page.mouse.wheel({ deltaY: Math.min(400, Math.max(60, d)) }); await sleep(30);
}
await sleep(800);
const endIn = () => page.evaluate(() => document.getElementById('rover').getBoundingClientRect().bottom <= innerHeight);
const track = []; let k = 0;
for (let i = 0; i < 600; i++) {
  await page.mouse.wheel({ deltaY: 60 });
  await sleep(90);
  const st = await page.evaluate(() => ({ y: Math.round(scrollY), rover: window.__qa && window.__qa.rover }));
  track.push(st);
  const name = String(k).padStart(3, '0');
  if (st.rover && st.rover.x > -200 && st.rover.x < W + 200) {
    const cx = Math.max(0, Math.min(W - 240, Math.round(st.rover.x - 120))), cy = Math.max(0, Math.min(H - 240, Math.round(st.rover.y - 160)));
    await page.screenshot({ path: OUT + `c_${name}.jpg`, type: 'jpeg', quality: 80, clip: { x: cx, y: cy + st.y, width: 240, height: 240 }, captureBeyondViewport: false });   // clip is in page coordinates
  }
  if (k % 4 === 0) await page.screenshot({ path: OUT + `f_${name}.jpg`, type: 'jpeg', quality: 70 });
  k++;
  if (await endIn()) break;
}
fs.writeFileSync(OUT + 'track.jsonl', track.map(t => JSON.stringify(t)).join('\n'));
await browser.close();
const sheet = (glob, cols, out, w) => { try {
  execFileSync('/opt/homebrew/bin/ffmpeg', ['-loglevel', 'error', '-y', '-pattern_type', 'glob', '-i', OUT + glob, '-vf', `scale=${w}:-1,tile=${cols}x8`, '-frames:v', '1', OUT + out]);
} catch (e) { /* ffmpeg optional */ } };
sheet('c_*.jpg', 10, 'crops.jpg', 160); sheet('f_*.jpg', 5, 'frames.jpg', 288);
// jitter summary: screen-space step of the rover between consecutive steps
const steps = []; for (let i = 1; i < track.length; i++) { const a = track[i - 1].rover, b = track[i].rover; if (a && b) steps.push(Math.hypot(b.x - a.x, b.y - a.y)); }
steps.sort((a, b) => a - b);
console.log(JSON.stringify({ steps: track.length, rover_frames: steps.length, step_px_p50: steps[Math.floor(steps.length / 2)] | 0, step_px_p95: steps[Math.floor(steps.length * 0.95)] | 0, step_px_max: steps[steps.length - 1] | 0, out: OUT }));

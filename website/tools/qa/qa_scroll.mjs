// Real-scroll QA: wheel events in small steps from top to bottom, a screenshot + JSON line every ~300 px,
// and a list of positions where something required is not visible.
//   cd website/tools/qa && node qa_scroll.mjs [width] [height]      (site served at http://localhost:8780/)
// Output: website/tools/qa/out_<w>x<h>/  (frames, report.jsonl, failures.txt, contact.jpg)
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import { execFileSync } from 'node:child_process';

const W = +(process.argv[2] || 1440), H = +(process.argv[3] || 900);
const BASE = process.env.BASE || 'http://localhost:8780/';
// one output folder per page and size (parallel runs no longer overwrite each other), plus an optional OUT_TAG suffix
const PAGE_NAME = ((BASE.split('/').pop() || 'index.html').replace('.html', '') || 'index');
const OUT = new URL(`./out_${PAGE_NAME}_${W}x${H}${process.env.OUT_TAG ? '_' + process.env.OUT_TAG : ''}/`, import.meta.url).pathname;
fs.rmSync(OUT, { recursive: true, force: true }); fs.mkdirSync(OUT, { recursive: true });
const sleep = ms => new Promise(r => setTimeout(r, ms));

// Required content per beat, checked only in the middle of the beat (inner progress 0.2..0.8 of its pinned range).
// min = how many of the matched elements must be visible (in viewport, effective opacity > 0.5, not display:none).
// Page-specific sets: BASE ending in landing.html -> the scroll story; index.html or "/" -> the company home.
// A missing section is reported as a note (not a failure), so sections still being built do not fail the run.
const PAGE = /landing\.html/.test(BASE) ? 'landing' : (/(index\.html|\/)$/.test(BASE) ? 'home' : 'other');
const BEATS_LANDING = [
  { id: 'top', sel: '#top h1', min: 1 },
  { id: 'problem', sel: '#problem .ft', min: 2 },
  { id: 'ai', sel: '#ai .pl-big, #ai .pl-stream, #ai .pl-result, #ai h2', min: 1 },
  { id: 'built', sel: '#built .cur-ide', min: 1 },
  { id: 'agents', sel: '#agents .tz, #agents h2', min: 2 },
  { id: 'results', sel: '#results .stat-row li, #results h2', min: 2 },
  { id: 'landing', sel: '#landing canvas#ghosts, #landing .td-legend', min: 2 },
  { id: 'rover', sel: '#rover .rv-tile, #rover h2', min: 2 },
  { id: 'neat', sel: '#neat canvas, #neat h2', min: 2 },
];
const BEATS_HOME = [
  { id: 'hero', root: '.ch-hero', sel: '.ch-hero h1, .ch-play', min: 2 },
  { id: 'products', root: '#products', sel: '.ch-prod', min: 1 },   // stacked on phones: one card fills the view
  { id: 'reel', root: '#reel', sel: '.ch-reel, .ch-reel-play', min: 2 },
  { id: 'research', root: '#research', sel: '.ch-rc', min: 2 },
];
const BEATS = PAGE === 'home' ? BEATS_HOME : PAGE === 'landing' ? BEATS_LANDING : [];
const LANDER_CHECK = PAGE === 'landing';   // the side lander exists only in the landing story

const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: 'new', timeout: 120000, protocolTimeout: 180000, args: ['--enable-webgl', '--ignore-gpu-blocklist', `--window-size=${W},${H}`, '--hide-scrollbars'],
  defaultViewport: { width: W, height: H, deviceScaleFactor: 1, isMobile: W < 500, hasTouch: false },
});
const page = await browser.newPage();
const errors = [];
page.on('pageerror', e => errors.push('pageerror: ' + e.message));
page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
await page.goto(BASE, { waitUntil: 'networkidle2', timeout: 60000 });
await sleep(2500);
await page.mouse.move(W / 2, H / 2);

const probe = (beats, landerCheck) => {
  const vis = el => {
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return false;
    const ix = Math.min(r.right, innerWidth) - Math.max(r.left, 0), iy = Math.min(r.bottom, innerHeight) - Math.max(r.top, 0);
    if (ix <= 0 || iy <= 0 || (ix * iy) / (r.width * r.height) < 0.5) return false;
    let o = 1;
    for (let e = el; e && e.nodeType === 1; e = e.parentElement) {
      const cs = getComputedStyle(e);
      if (cs.display === 'none' || cs.visibility === 'hidden') return false;
      o *= parseFloat(cs.opacity);
    }
    return o > 0.5;
  };
  const out = { y: Math.round(scrollY), active: null, inner: null, fails: [], notes: [] };
  for (const b of beats) {
    const s = b.root ? document.querySelector(b.root) : document.getElementById(b.id); if (!s) { out.notes.push(b.id + ': section missing'); continue; }
    const r = s.getBoundingClientRect();
    const range = Math.max(1, r.height - innerHeight);
    const p = -r.top / range;
    if (r.top <= 1 && r.bottom >= innerHeight - 1) { out.active = b.id; out.inner = +p.toFixed(2); }
    const pinnedMid = r.height > innerHeight * 1.3 ? (p >= 0.2 && p <= 0.8) : (r.top < innerHeight * 0.3 && r.bottom > innerHeight * 0.7);
    if (!pinnedMid) continue;
    const els = [...document.querySelectorAll(b.sel)];
    const n = els.filter(vis).length;
    if (n < b.min) out.fails.push(`${b.id} @${p.toFixed(2)}: ${n}/${b.min} visible of ${els.length} (${b.sel})`);
  }
  // the side lander: visible on screen from the hero to touchdown (3D only)
  if (!landerCheck) return out;
  const qa = window.__qa, td = document.querySelector('.landing-spacer');
  const beforeTouchdown = td ? td.getBoundingClientRect().top > innerHeight * 0.5 : true;
  const sprite = document.getElementById('side-lander');
  if (innerWidth <= 760 && sprite && !document.body.classList.contains('no-webgl') && beforeTouchdown && !document.querySelector('#gauge.finale')) {   // phones: DOM sprite above the content
    if (!vis(sprite)) out.fails.push('side lander (phone sprite) not visible');
  } else if (!document.body.classList.contains('no-webgl') && beforeTouchdown) {
    if (!qa || !qa.lander) out.fails.push('side lander: no __qa hook yet');
    else if (qa.lander.visible && qa.lander.inView && qa.lander.hPx >= 12 && (() => {   // occlusion: an opaque page element over the lander?
      const els = document.elementsFromPoint(qa.lander.x, qa.lander.y);
      for (const e of els) {
        if (e.id === 'scene') return false;
        const cs = getComputedStyle(e), bg = cs.backgroundColor, a = bg.startsWith('rgba') ? parseFloat(bg.split(',')[3]) : (bg === 'transparent' ? 0 : 1);
        if (e.tagName === 'IMG' || e.tagName === 'VIDEO' || a > 0.6) return !e.closest('#gauge, .topnav');
      }
      return false;
    })()) out.fails.push(`side lander covered by page content at (${Math.round(qa.lander.x)}, ${Math.round(qa.lander.y)})`);
    else if (!qa.lander.visible || !qa.lander.inView || qa.lander.hPx < 12) out.fails.push(`side lander not visible (x ${Math.round(qa.lander.x)}, y ${Math.round(qa.lander.y)}, h ${Math.round(qa.lander.hPx)} px, visible ${qa.lander.visible})`);
    out.lander = qa && qa.lander ? { x: Math.round(qa.lander.x), y: Math.round(qa.lander.y), h: Math.round(qa.lander.hPx) } : null;
  }
  return out;
};

const total = await page.evaluate(() => document.documentElement.scrollHeight - innerHeight);
const lines = [];
let lastShot = -1e9, k = 0;
for (let guard = 0; guard < 4000; guard++) {
  const y = await page.evaluate(() => scrollY);
  if (y - lastShot >= 300 || y >= total - 2) {
    await sleep(350);                                        // let the lerped scroll / pinned stages settle a little
    const rec = await page.evaluate(probe, BEATS, LANDER_CHECK);
    const f = `f_${String(k).padStart(3, '0')}.jpg`;
    await page.screenshot({ path: OUT + f, type: 'jpeg', quality: 60 });
    rec.frame = f; lines.push(rec); k++; lastShot = y;
    if (y >= total - 2) break;
  }
  await page.mouse.wheel({ deltaY: 120 });
  await sleep(80);
  if (guard % 50 === 0) { const t2 = await page.evaluate(() => document.documentElement.scrollHeight - innerHeight); if (t2 > 0) total; }
}
fs.writeFileSync(OUT + 'report.jsonl', lines.map(l => JSON.stringify(l)).join('\n') + '\n');
const fails = lines.filter(l => l.fails.length).map(l => `${l.frame} y=${l.y} [${l.active || '-'} ${l.inner ?? ''}] ${l.fails.join(' | ')}`);
const notes = [...new Set(lines.flatMap(l => l.notes || []))];
fs.writeFileSync(OUT + 'failures.txt', (fails.join('\n') || 'NONE') + '\n' + (notes.length ? '\nNOTES:\n' + notes.join('\n') + '\n' : '') + (errors.length ? '\nERRORS:\n' + errors.join('\n') : '') + '\n');
console.log(`${lines.length} frames, ${fails.length} with failures, ${errors.length} errors -> ${OUT}`);
try {
  const cols = 8;
  execFileSync('/opt/homebrew/bin/ffmpeg', ['-v', 'error', '-y', '-framerate', '1', '-i', OUT + 'f_%03d.jpg',
    '-vf', `scale=360:-1,tile=${cols}x${Math.ceil(lines.length / cols)}:padding=4`, '-frames:v', '1', OUT + 'contact.jpg']);
} catch (e) { console.warn('contact sheet failed', e.message); }
await browser.close();

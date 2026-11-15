// Full-page screenshots + console check of every subpage at desktop and phone width.
//   cd website/tools/qa && node pages_shots.mjs   ->  out_pages/<page>_<w>.jpg, out_pages/report.txt
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
const OUT = new URL('./out_pages/', import.meta.url).pathname; fs.mkdirSync(OUT, { recursive: true });
// pages from the command line (e.g. node pages_shots.mjs pilot research/hazard-model), else every page
const pages = process.argv.slice(2).length ? process.argv.slice(2) : ['index', 'pilot', 'navigator', 'research', 'company', 'mission-control', 'research/pilot-benchmark', 'research/hazard-model', 'research/dream-rsi', 'research/benchmarks'];
const b = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: 'new', timeout: 120000, protocolTimeout: 180000, args: ['--enable-webgl', '--ignore-gpu-blocklist', '--hide-scrollbars'] });
const rep = [];
for (const [w, h] of [[1440, 900], [390, 844]]) for (const pg of pages) {
  const p = await b.newPage(); await p.setViewport({ width: w, height: h, isMobile: w < 500 });
  const errs = [];
  p.on('pageerror', e => errs.push(e.message)); p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
  p.on('response', r => { if (r.status() >= 400) errs.push(r.status() + ' ' + r.url()); });
  await p.goto(`http://localhost:8780/${pg}.html`, { waitUntil: 'networkidle2' });
  // scroll through once so on-view reveals run, then back to the top
  const H = await p.evaluate(() => document.documentElement.scrollHeight);
  for (let y = 0; y < H; y += 400) { await p.evaluate(v => scrollTo(0, v), y); await new Promise(r => setTimeout(r, 60)); }
  await p.evaluate(() => scrollTo(0, 0)); await new Promise(r => setTimeout(r, 1200));
  const hs = await p.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
  await p.screenshot({ path: `${OUT}${pg.replace('/', '__')}_${w}.jpg`, type: 'jpeg', quality: 55, fullPage: true });
  rep.push(`${pg} @${w}: ${errs.length ? 'ERRORS ' + errs.join(' ; ') : 'no console errors'}${hs ? ' · HORIZONTAL SCROLL' : ''}`);
  await p.close();
}
fs.writeFileSync(OUT + 'report.txt', rep.join('\n') + '\n'); console.log(rep.join('\n'));
await b.close();

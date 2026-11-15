// Element screenshots of the "What the model sees and says" blocks (js/examples.js) + console errors.
//   cd website/tools/qa && node ex_shots.mjs  ->  out_ex/*.png
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
const OUT = new URL('./out_ex/', import.meta.url).pathname; fs.mkdirSync(OUT, { recursive: true });
const T = [['research/hazard-model', '#ex-hazard'], ['research/rover', '#ex-rover'], ['navigator', '#ex-planner'], ['research/training', '#ex-light'], ['pilot', '#pp-hz-teaser'], ['research/pilot-benchmark', '#f3d-pairs'], ['landing', '#cur-ide']];
const only = process.argv[2];
const b = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: 'new', args: ['--enable-webgl', '--ignore-gpu-blocklist', '--enable-unsafe-swiftshader', '--hide-scrollbars'] });
for (const w of only ? [+only] : [1440, 390]) for (const [pg, sel] of T) {
  const p = await b.newPage(); await p.setViewport({ width: w, height: 900, isMobile: w < 500 });
  const errs = [];
  p.on('pageerror', e => errs.push(e.message)); p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
  p.on('response', r => { if (r.status() >= 400) errs.push(r.status() + ' ' + r.url()); });
  await p.goto(`http://localhost:8780/${pg}.html`, { waitUntil: 'networkidle2', timeout: 60000 });
  if (pg === 'landing') { const y0 = await p.evaluate(s => document.querySelector(s).getBoundingClientRect().top + scrollY - 1400, sel); for (let k = 0; k < 40; k++) { await p.evaluate(v => scrollTo(0, v), y0 + k * 60); await new Promise(r => setTimeout(r, 120)); } }
  else await p.evaluate(s => document.querySelector(s)?.scrollIntoView({ block: 'center' }), sel);
  await new Promise(r => setTimeout(r, 3500));
  const hs = await p.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
  const el = await p.$(sel);
  const fn = OUT + pg.replace('/', '__') + '_' + w + '.png';
  if (el) await el.screenshot({ path: fn }); 
  console.log(pg, w, el ? 'shot' : 'NO ELEMENT', hs ? 'HSCROLL' : '', errs.join(' | '));
  await p.close();
}
await b.close();

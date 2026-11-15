// QA for the 3D figures of one research page, at 1440 and 390 px.
//   cd website/tools/qa && node fig3d_qa.mjs research/hazard-model [outdir]
// For each width: console errors, failed requests, horizontal scroll; then scrolls to every figure.f3d, waits for
// it to go live (1440: .f3d-ready; 390: static still) and screenshots the figure (stage + controls + caption)
// to <outdir>/<page>_<figure id>_<w>.jpg. Also clicks every control button once and reports errors after that.
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
const [page, outdir = '/tmp'] = process.argv.slice(2);
fs.mkdirSync(outdir, { recursive: true });
const b = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: 'new',
  args: ['--enable-webgl', '--ignore-gpu-blocklist', '--enable-unsafe-swiftshader', '--hide-scrollbars'] });
for (const [w, h] of [[1440, 900], [390, 844]]) {
  const p = await b.newPage(); await p.setViewport({ width: w, height: h, isMobile: w < 500, hasTouch: w < 500 });
  const errs = [];
  p.on('pageerror', e => errs.push(e.message)); p.on('console', m => { if (m.type() === 'error' || (m.type() === 'warning' && /fig3d/.test(m.text()))) errs.push(m.text()); });
  p.on('response', r => { if (r.status() >= 400) errs.push(r.status() + ' ' + r.url()); });
  await p.goto(`http://localhost:8780/${page}.html`, { waitUntil: 'networkidle2', timeout: 90000 });
  const ids = await p.$$eval('figure.f3d', fs => fs.map(f => f.id));
  for (const id of ids) {
    await p.$eval('#' + id, el => el.scrollIntoView({ block: 'center' }));
    let st = 'static';
    if (w > 1000) {
      try { await p.waitForFunction(i => { const c = document.getElementById(i).classList; return c.contains('f3d-ready') || c.contains('f3d-failed'); }, { timeout: 60000 }, id); }
      catch { st = 'TIMEOUT'; }
      if (st !== 'TIMEOUT') st = await p.$eval('#' + id, el => el.classList.contains('f3d-failed') ? 'FAILED' : 'live');
      // press each control once (then restore the first state of segmented groups)
      const n = await p.$$eval(`#${id} .f3d-ctl button`, bs => bs.length);
      for (let i = 0; i < n; i++) { await p.$$eval(`#${id} .f3d-ctl button`, (bs, k) => bs[k] && bs[k].click(), i); await new Promise(r => setTimeout(r, 250)); }
      await p.$$eval(`#${id} .f3d-seg`, gs => gs.forEach(g => g.querySelector('button')?.click()));
      await new Promise(r => setTimeout(r, 1500));
    } else await new Promise(r => setTimeout(r, 600));
    const kb = await p.$eval('#' + id, el => { const i = el.querySelector('.f3d-still'); return i ? `${i.naturalWidth}x${i.naturalHeight}` : 'NO STILL'; });
    await (await p.$('#' + id)).screenshot({ path: `${outdir}/${page.replace('/', '__')}_${id}_${w}.jpg`, type: 'jpeg', quality: 55 });
    console.log(`${page} @${w} #${id}: ${st}, still ${kb}`);
  }
  const hs = await p.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
  const ctx = await p.evaluate(() => document.querySelectorAll('canvas.f3d-canvas').length);
  console.log(`${page} @${w}: ${errs.length ? 'ERRORS ' + errs.join(' ; ') : 'no console errors'}${hs ? ' · HORIZONTAL SCROLL' : ''} · f3d canvases in DOM: ${ctx}`);
  await p.close();
}
await b.close();

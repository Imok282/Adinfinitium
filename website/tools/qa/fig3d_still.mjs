// Render the static fallback image of one 3D research figure (js/fig3d.js ?f3dshot=<figure id>).
//   cd website/tools/qa && node fig3d_still.mjs research/hazard-model f3d-light ../../assets/fig3d/hazard/still_light.jpg
// Needs the site served at http://localhost:8780/ (cd website && python3 -m http.server 8780).
// Writes a 1280 x 720 JPEG (quality 78). Fails if the figure did not render.
import puppeteer from 'puppeteer-core';
const [page, id, out, q = '78'] = process.argv.slice(2);
if (!page || !id || !out) { console.error('usage: node fig3d_still.mjs <page> <figure id> <out.jpg> [quality]'); process.exit(2); }
const b = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: 'new',
  args: ['--enable-webgl', '--ignore-gpu-blocklist', '--enable-unsafe-swiftshader', '--hide-scrollbars'] });
const p = await b.newPage(); await p.setViewport({ width: 1280, height: 720 });
const errs = [];
p.on('pageerror', e => errs.push(e.message)); p.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') errs.push(m.text()); });
await p.goto(`http://localhost:8780/${page}.html?f3dshot=${id}`, { waitUntil: 'networkidle2', timeout: 90000 });
try {
  await p.waitForFunction(i => { const f = document.getElementById(i); return f && (f.classList.contains('f3d-failed') || +f.querySelector('.f3d-stage').dataset.frames > 8); }, { timeout: 90000 }, id);
} catch (e) { errs.push('timeout waiting for frames'); }
const failed = await p.evaluate(i => document.getElementById(i)?.classList.contains('f3d-failed'), id);
await new Promise(r => setTimeout(r, 800));
if (!failed) await p.screenshot({ path: out, type: 'jpeg', quality: +q });
console.log(failed ? 'FAILED' : 'wrote ' + out, errs.length ? '\n' + errs.join('\n') : '');
await b.close();
process.exit(failed ? 1 : 0);

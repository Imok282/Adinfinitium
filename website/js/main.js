// Page orchestration: scroll progress, gauge, lazy modules, reduced-motion / no-WebGL fallbacks.
import { updateGauge } from './gauge.js';

const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const params = new URLSearchParams(location.search);
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));

function hasWebGL() {
  try { const c = document.createElement('canvas'); return !!(c.getContext('webgl2') || c.getContext('webgl')); } catch { return false; }
}
const use3D = !reduced && hasWebGL() && !params.has('static');
if (!use3D) document.body.classList.add('no-webgl');
// Tall scrubbed stage sections (css/stage.css) from the first layout, so deep links do not jump when js/stage.js loads.
if (use3D && innerWidth > 760) document.documentElement.classList.add('stage-on');

// ---------- scroll progress helpers ----------
// Screenshot mode (?scroll=...): headless Chrome cannot capture a scrolled window, so the body becomes the scroller.
const SHOT = params.has('scroll');
if (SHOT) document.documentElement.classList.add('shot');
const scroller = () => (SHOT ? document.body : document.documentElement);
const getY = () => (SHOT ? document.body.scrollTop : scrollY);
const setY = y => (SHOT ? (document.body.scrollTop = y) : scrollTo(0, y));
// Layout is cached (offsets + heights, re-measured on resize / ResizeObserver); per frame only cheap maths runs.
// `sy` is the smoothed scroll position (lerped toward the real one, snapped in screenshot / reduced-motion mode).
const SNAP = params.has('scroll') || reduced;
let vh = innerHeight, docH = 1, sy = 0, measured = false;
const rects = new Map();          // element -> { top, h } in document coordinates
function rectOf(el) {
  let r = rects.get(el);
  if (!r) { const b = el.getBoundingClientRect(); r = { top: b.top + getY(), h: b.height }; rects.set(el, r); }
  return r;
}
function measure() {
  vh = innerHeight; docH = scroller().scrollHeight; rects.clear(); measured = true;
}
const docP = () => clamp(getY() / Math.max(1, docH - vh));
function secProgress(el, y = sy) {   // 0 when the section top hits the viewport top, 1 when its bottom hits the viewport bottom
  const r = rectOf(el);
  return clamp((y - r.top) / Math.max(1, r.h - vh));
}
function secVisibility(el, y = sy) { // 0..1 how much of the viewport the section covers (eased at the edges)
  const r = rectOf(el);
  const cover = (Math.min(r.top + r.h - y, vh) - Math.max(r.top - y, 0)) / vh;
  return clamp(cover * 1.6 - 0.3);
}

// ---------- modules ----------
let scene = null;

// ---------- beat modules (round 3) ----------
// Each agent mounts its module between its own BEGIN/END markers below. Interface:
//   addBeat(sectionEl, mod)  where mod.update(p, vis, t) is called once per animation frame from the single loop,
//   p = section progress 0..1 (0: section top at viewport top, 1: section bottom at viewport bottom), vis = 0..1 on-screen.
//   ctx.getScene() returns the 3D scene API (or null): setProgress(p), setNet(q, mix), ...
const beats = [];
const ctx = { reducedMotion: reduced, get use3D() { return use3D; }, getScene: () => scene, shot: params.has('scroll') };
function addBeat(el, mod) { if (el && mod && typeof mod.update === 'function') { beats.push({ el, mod }); onScroll(); } }
const warn = name => e => console.warn(name, e);
// other modules (e.g. inline <script type=module> on landing.html) can register a beat without importing main.js:
//   document.dispatchEvent(new CustomEvent('adi:beat', { detail: { el, mod } }))   (mod.update(p, vis, t))
document.addEventListener('adi:beat', e => { try { addBeat(e.detail.el, e.detail.mod); } catch (err) { console.warn('adi:beat', err); } });
// BEGIN:A-stage
// Hero / mission / results / FAQ / CTA beats (js/stage.js; helpers exported there for reuse).
import('./stage.js').then(m => {
  for (const [id, mk] of [['top', m.mountHero], ['problem', m.mountMission]]) {
    try { const el = document.getElementById(id); const beatEl = (id === 'results' && el && el.querySelector('.strip-wrap')) || el; el && addBeat(beatEl, mk(el, ctx)); } catch (e) { console.warn('stage', id, e); }
  }
}).catch(warn('stage'));
// END:A-stage
// BEGIN:B-pilot
// #ai cinematic beat (js/pilot.js): fly into the lander, real camera patches, real-weight fly-through (drives scene.setNet), thrust arrows
import('./pilot.js').then(m => {
  const el = document.getElementById('ai');
  try { addBeat(el, m.mount(el, ctx)); } catch (e) { console.warn('pilot', e); }
}).catch(warn('pilot'));
// END:B-pilot
// BEGIN:C-built
// #built: multiplayer cursors assemble the software in a mock IDE + real git-history ticker (js/cursors.js, js/buildlog.js)
import('./cursors.js').then(m => {
  const el = document.getElementById('built');
  try { addBeat(el, m.mount(el, ctx)); } catch (e) { warn('cursors')(e); }
}).catch(warn('cursors'));
// END:C-built
// BEGIN:D-modelcards
// model cards moved to models.html (home keeps a teaser)
// END:D-modelcards
// BEGIN:E-ghosts
// Touchdown beat: every simulated descent (assets/data/ghosts_lander_site06.*) as additive lines (js/ghosts.js)
import('./ghosts.js').then(m => { const el = document.getElementById('landing'); if (el) { try { addBeat(el, m.mount(el, ctx)); } catch (e) { console.warn('ghosts', e); } } }).catch(warn('ghosts'));
// END:E-ghosts
// BEGIN:F-rover
// Rover beat (#rover): js/rover.js mounts the DOM (headline, HUD, projected labels) and drives the 3D rover via scene.setRover
try {
  const roverEl = document.getElementById('rover');
  if (roverEl) import('./rover.js').then(m => addBeat(roverEl, m.mount(roverEl, ctx))).catch(warn('rover'));
  // screenshots: late layout changes (tall sticky beats mounting) can move a ?scroll=<id> target; re-seek once more
  if (SHOT && params.get('scroll') && Number.isNaN(parseFloat(params.get('scroll')))) {
    [10500, 13000].forEach(ms => setTimeout(() => {
      const el = document.getElementById(params.get('scroll'));
      if (el) { setY(el.getBoundingClientRect().top + getY() + (parseFloat(params.get('off')) || 0) * innerHeight); onScroll(); }
    }, ms));
  }
} catch (e) { console.warn('rover', e); }
// END:F-rover
const secAI = document.getElementById('ai');


// Models section: interactive 3D terrain views, lazy per view (js/models3d.js)
if (document.getElementById('models')) import('./models3d.js').then(m => m.mountModels(document.getElementById('models'), { reducedMotion: reduced, use3D }))
  .catch(e => console.warn('models3d', e));
// Results section: every number comes from assets/data/results.json (js/results.js)
import('./results.js').then(m => m.mountResults(document.getElementById('results-grid')))
  .catch(e => console.warn('results', e));
// Cinematic media slots (hero / descent / landing videos), hidden unless the files exist (js/media.js)
import('./media.js').then(m => m.mountMedia({ reducedMotion: reduced }))
  .catch(e => console.warn('media', e));

if (use3D) {
  import('./scene.js').then(m => m.createScene(document.getElementById('scene'), { onReady: ok => { if (ok !== false) document.body.classList.add('scene-ready'); } }))
    .then(s => { scene = s; onScroll(); })
    .catch(e => { console.warn('3D disabled', e); document.body.classList.add('no-webgl'); fallbackStatic(); });
} else {
  fallbackStatic();
}

function fallbackStatic() {
  const poster = document.getElementById('poster');
  const spacer = document.querySelector('.landing-spacer');   // landing poster once the finale (touchdown) is on screen
  const late = () => (spacer ? spacer.getBoundingClientRect().top < innerHeight * 0.6 : docP() > 0.9);
  const setPoster = () => { poster.style.backgroundImage = `url(assets/img/${late() ? 'poster_landing' : 'poster_hero'}.jpg)`; };
  setPoster(); (SHOT ? document.body : window).addEventListener('scroll', setPoster, { passive: true });
  fetch('assets/net.json').then(r => r.json()).then(d => import('./net2d.js').then(m => m.drawNet2D(document.getElementById('nn-static'), d))).catch(() => {});
}

// ---------- the one animation loop ----------
// One requestAnimationFrame loop while the page is visible. Scroll is lerped (sy += (y - sy) * 0.12); beats get
// update(p, vis, t) every frame from the smoothed value; scene / gauge / nav only when something moved.
var dirty = true, raf = 0, frameN = 0, lastY = -1, lastSy = -1;   // var: onScroll() may run before this line
const navLinks = [...document.querySelectorAll('.topnav [data-nav]')].map(a => ({ a, ids: a.dataset.nav.split(' '), on: null }));
const railEl = document.getElementById('gauge');
function onScroll() { dirty = true; if (!raf && !document.hidden) raf = requestAnimationFrame(frame); }
function navAndRail() {
  const mid = sy + vh * 0.5;
  for (const n of navLinks) {   // highlight the pill / sheet link whose section(s) cross the middle of the viewport
    const on = n.ids.some(id => { const s = document.getElementById(id); if (!s) return false; const r = rectOf(s); return r.top < mid && r.top + r.h > mid; });
    if (on === n.on) continue;
    n.on = on; n.a.classList.toggle('active', on);
    if (on) n.a.setAttribute('aria-current', 'location'); else n.a.removeAttribute('aria-current');
  }
  const ft = document.querySelector('.cta-band') || document.querySelector('.footer');
  if (ft && railEl) railEl.classList.toggle('rail-off', rectOf(ft).top - sy < vh * 0.35);
}
function frame(now) {
  raf = document.hidden ? 0 : requestAnimationFrame(frame);
  if (!measured) measure();
  const y = getY();
  if (SNAP || dirty && Math.abs(y - sy) > vh * 3) sy = y;   // snap for screenshots / reduced motion / long jumps
  else { sy += (y - sy) * 0.12; if (Math.abs(y - sy) < 0.3) sy = y; }
  const yMoved = y !== lastY, sMoved = sy !== lastSy;
  if (yMoved || dirty) {       // raw page fraction: the scene and the rail damp themselves
    const p = docP();
    updateGauge(p);
    if (scene) scene.setProgress(p);
  }
  if (sMoved || dirty) {
    if (scene && secAI && !beats.some(b => b.el === secAI)) scene.setNet(secProgress(secAI), secVisibility(secAI));
    if (dirty || frameN % 6 === 0) navAndRail();
  }
  const t = now / 1000;
  for (let i = 0; i < beats.length; i++) {
    const b = beats[i];
    try { b.mod.update(secProgress(b.el), secVisibility(b.el), t); } catch (e) { console.warn('beat', b.el && b.el.id, e); beats.splice(i--, 1); }
  }
  lastY = y; lastSy = sy; dirty = false; frameN++;
}
(SHOT ? document.body : window).addEventListener('scroll', onScroll, { passive: true });
addEventListener('resize', () => { measured = false; onScroll(); });
document.addEventListener('visibilitychange', onScroll);
{ // body for normal pages; <main> too, because in screenshot mode the body is the fixed-height scroller
  const ro = new ResizeObserver(() => { measured = false; onScroll(); });
  ro.observe(document.body); const st = document.getElementById('story'); st && ro.observe(st);
}
onScroll();
if (params.has('clean')) document.documentElement.classList.add('clean');

// ?scroll=0.42 (fraction of the page) or ?scroll=results (section id): used for screenshots and deep links
addEventListener('load', () => {
  const v = params.get('scroll');
  if (!v) return;
  document.documentElement.style.scrollBehavior = 'auto';
  const go = () => {
    const f = parseFloat(v);
    if (!Number.isNaN(f)) setY(f * (scroller().scrollHeight - innerHeight));
    else { const el = document.getElementById(v); el && setY(el.getBoundingClientRect().top + getY() + (parseFloat(params.get('off')) || 0) * innerHeight); }
    onScroll();
  };
  go(); [300, 1500, 3500, 6000, 8000, 9500].forEach(ms => setTimeout(go, ms));
  document.fonts && document.fonts.ready.then(go);
  // re-seek whenever the page height changes during the first 15 s (lazy modules grow sections), so offsets are exact
  const t0 = performance.now(), st = document.getElementById('story');
  if (st) { let q = 0; const ro2 = new ResizeObserver(() => { if (performance.now() - t0 > 15000) { ro2.disconnect(); return; } if (!q) q = requestAnimationFrame(() => { q = 0; go(); }); }); ro2.observe(st); }
  addEventListener('resize', () => { if (performance.now() - t0 < 15000) go(); });
});

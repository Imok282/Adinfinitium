// Altitude rail + the shared descent timeline (also read by js/scene.js).
// Illustrative profile: the simulation itself only covers the terminal ~2 km.
//
// The profile is anchored to page sections (positions cached, refreshed at most every 400 ms / on resize):
//   page top (hero) ........ 100 km   q = 0
//   #problem top ........... 100 km   q = 0.10  (orbit ends, descent orbit starts)
//   #built top ............. 34 km    q = 0.30
//   #built bottom .......... 15 km    q = 0.45  (bottom of #built at the viewport bottom)
//   #landing top ........... 2 km     q = 0.80  (through #agents, #models, #results, #faq: below 15 km)
//   .landing-spacer ........ 2 km -> 0 m, touchdown at f = TD, then ramp + rover drive-out
//   #rover ................. 0 m      (rover beat)
// Timeline values:
//   hero  0..1  how far the hero section has scrolled away (orbit -> side band)
//   q     0..0.8 descent progress before the finale
//   f     0..1  finale: progress through the .landing-spacer (0 = its top at the viewport bottom, 1 = its bottom there)
//   foot  0..1  fraction of the viewport covered by the closing band / footer
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const TD = 0.5;             // finale fraction at touchdown

const ALT = [[0, 100], [0.1, 100], [0.3, 34], [0.45, 15], [0.8, 2]];   // q -> km
const TRK = [[0.1, 0], [0.3, 35], [0.45, 50], [0.8, 82]];             // q -> % of rail
function logInterp(a0, a1, t) { return Math.exp(Math.log(a0 + 0.001) + (Math.log(a1 + 0.001) - Math.log(a0 + 0.001)) * t) - 0.001; }
function keyed(K, x, log) {
  if (x <= K[0][0]) return K[0][1];
  for (let i = 1; i < K.length; i++) {
    const [x0, a] = K[i - 1], [x1, b] = K[i];
    if (x <= x1) { const t = (x - x0) / (x1 - x0); return log ? logInterp(a, b, t) : a + (b - a) * t; }
  }
  return K[K.length - 1][1];
}

const T = { p: 0, q: 0, f: 0, hero: 0, foot: 0, altKm: 100, track: 0, landed: false, rover: 0 };
let spacer = null, top = null, closer = null, roverSec = null;
let anchors = null, anchorT = 0;   // [[scrollY, q], ...]
const shot = () => document.documentElement.classList.contains('shot');
const scrollYNow = () => (shot() ? document.body.scrollTop : (window.scrollY || 0));
addEventListener('resize', () => { anchors = null; });
function absTop(el, y) { return el.getBoundingClientRect().top + y; }
function getAnchors(y, ih) {
  const now = performance.now();
  if (anchors && now - anchorT < 400) return anchors;
  anchorT = now;
  const P = document.getElementById('problem'), B = document.getElementById('built'), L = document.getElementById('landing');
  if (!P || !B || !L) { anchors = null; return null; }
  const bt = absTop(B, y), bb = bt + Math.max(ih, B.offsetHeight) - ih;
  const a = [[0, 0], [absTop(P, y), 0.1], [bt, 0.3], [Math.max(bt + 1, bb), 0.45], [Math.max(bb + 2, absTop(L, y)), 0.8]];
  for (let i = 1; i < a.length; i++) if (a[i][0] <= a[i - 1][0]) a[i][0] = a[i - 1][0] + 1;
  anchors = a;
  return a;
}
function qAt(y, A) {
  if (y <= A[0][0]) return 0;
  for (let i = 1; i < A.length; i++) if (y <= A[i][0]) return A[i - 1][1] + (A[i][1] - A[i - 1][1]) * (y - A[i - 1][0]) / (A[i][0] - A[i - 1][0]);
  return 0.8;
}
export function timeline(p) {
  const ih = innerHeight || 1;
  spacer = spacer || document.querySelector('.landing-spacer');
  top = top || document.getElementById('top');
  closer = closer || document.querySelector('section.cta-band') || document.querySelector('.footer');
  roverSec = roverSec || document.getElementById('rover');
  let f = 0, hero = clamp(p / 0.07), foot = 0, q;
  if (spacer) { const r = spacer.getBoundingClientRect(); f = clamp((ih - r.top) / Math.max(1, r.height)); }
  if (top) { const r = top.getBoundingClientRect(); hero = clamp(-r.top / Math.max(1, r.height * 0.8)); }
  if (closer) foot = clamp((ih - closer.getBoundingClientRect().top) / ih);
  const y = scrollYNow(), A = getAnchors(y, ih);
  q = f > 0 ? 0.8 : A ? clamp(qAt(y, A), 0, 0.8) : clamp(p, 0, 0.8);
  T.rover = 0;
  if (roverSec) { const r = roverSec.getBoundingClientRect(); T.rover = clamp(-r.top / Math.max(1, r.height - ih)); if (r.top < ih) f = Math.max(f, 1); }
  T.p = p; T.q = q; T.f = f; T.hero = hero; T.foot = foot;
  if (f <= 0) { T.altKm = keyed(ALT, q, true); T.track = q <= 0.1 ? 0 : keyed(TRK, q, false); }
  else { const t = clamp(f / TD); T.altKm = t >= 1 ? 0 : logInterp(2, 0, Math.pow(t, 0.8)); T.track = 82 + 18 * t; }
  T.landed = f >= TD;
  return T;
}

// ---------- DOM rail (marker damped, number tabular, phase crossfades) ----------
const gAlt = document.getElementById('gauge-alt'), gPhase = document.getElementById('gauge-phase');
const gFill = document.getElementById('gauge-fill'), gMark = document.getElementById('gauge-marker');
const gRoot = document.getElementById('gauge');
// 34 km tick (top of #built) added here so the rail labels match the anchors above
if (gMark && gMark.parentNode && !document.querySelector('.rail-tick[data-km="34"]')) {
  const t = document.createElement('div'); t.className = 'rail-tick'; t.dataset.km = '34'; t.style.setProperty('--at', '35%');
  t.innerHTML = '<span>34 km</span>'; gMark.parentNode.insertBefore(t, gMark);
}
const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
let phaseNow = '', phaseTimer = 0;
function fmt(km) { return km >= 10 ? `${km.toFixed(0)} km` : km >= 1 ? `${km.toFixed(1)} km` : km > 0.0005 ? `${Math.round(km * 1000)} m` : '0 m'; }
// marker damping is a CSS transition on top/height (no overshoot); reduced motion disables it in gauge.css
let placed = null, altTxt = null;
function place(v) { v = Math.round(v * 10) / 10; if (v === placed) return; placed = v; gFill.style.height = v + '%'; gMark.style.top = v + '%'; }
function setPhase(name) {
  if (name === phaseNow) return;
  phaseNow = name;
  if (reduced) { gPhase.textContent = name; return; }
  gPhase.classList.add('out');
  clearTimeout(phaseTimer);
  phaseTimer = setTimeout(() => { gPhase.textContent = phaseNow; gPhase.classList.remove('out'); }, 300);
}
export function updateGauge(p) {
  const s = timeline(p);
  const at = s.landed ? '0 m · Touchdown' : fmt(s.altKm);
  if (at !== altTxt) { altTxt = at; gAlt.textContent = at; }
  // the mission timeline, one short step caption at a time (steps 1-6 descent, 7-9 surface)
  setPhase(s.landed ? (s.rover > 0.85 ? '9 · Rover home before sunset' : s.rover > 0 ? '8 · Rover drives · scans · relays' : '7 · Lander becomes the base')
    : s.f > 0 ? '6 · Terminal descent · touchdown' : s.q < 0.1 ? '1 · Orbit · AI maps hazards' : s.q < 0.3 ? '2 · Lower orbit · AI picks when'
    : s.q < 0.36 ? '3 · Abort & retry?' : s.q < 0.6 ? '4 · Powered descent' : '5 · Hover · scan · re-target');
  place(s.track);
  if (gRoot) {
    gRoot.classList.toggle('landed', s.landed);
    gRoot.classList.toggle('finale', s.f > 0.02);
    gRoot.classList.toggle('rover-on', s.rover > 0);
  }
}

// altitude (km) for a (possibly damped) q / f pair, same profile as timeline()
export function altAt(q, f) {
  if (f <= 0) return keyed(ALT, q, true);
  const t = clamp(f / TD); return t >= 1 ? 0 : logInterp(2, 0, Math.pow(t, 0.8));
}
